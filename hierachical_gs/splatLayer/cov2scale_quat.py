import torch

# --------------------------
# utils: packing / symmetry
# --------------------------
def unpack_triu6_to_mat(cov6: torch.Tensor) -> torch.Tensor:
    """
    cov6: [N,6] with order [s00, s01, s02, s11, s12, s22] (row-major upper)
    return: [N,3,3] symmetric
    """
    N = cov6.shape[0]
    M = cov6.new_zeros((N, 3, 3))
    s00, s01, s02, s11, s12, s22 = cov6.unbind(-1)
    M[:, 0, 0] = s00
    M[:, 0, 1] = M[:, 1, 0] = s01
    M[:, 0, 2] = M[:, 2, 0] = s02
    M[:, 1, 1] = s11
    M[:, 1, 2] = M[:, 2, 1] = s12
    M[:, 2, 2] = s22
    return M

def symmetrize3(S: torch.Tensor) -> torch.Tensor:
    return 0.5 * (S + S.transpose(-1, -2))

# --------------------------
# rotmat -> quaternion (wxyz)
# --------------------------
def rotmat_to_quat(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    R: [N,3,3] proper rotations
    return: [N,4] quats (w,x,y,z), normalized, w>=0
    """
    N = R.shape[0]
    q = R.new_empty((N, 4))

    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    tpos = tr > 0
    # case trace > 0
    if tpos.any():
        t = torch.sqrt(torch.clamp(tr[tpos] + 1.0, min=eps)) * 2.0
        q[tpos, 0] = 0.25 * t
        q[tpos, 1] = (R[tpos, 2, 1] - R[tpos, 1, 2]) / t
        q[tpos, 2] = (R[tpos, 0, 2] - R[tpos, 2, 0]) / t
        q[tpos, 3] = (R[tpos, 1, 0] - R[tpos, 0, 1]) / t

    nt = ~tpos
    if nt.any():
        Rnt = R[nt]
        diag = torch.stack([Rnt[:, 0, 0], Rnt[:, 1, 1], Rnt[:, 2, 2]], dim=1)
        i = torch.argmax(diag, dim=1)

        qnt = Rnt.new_zeros((Rnt.size(0), 4))

        idx0 = i == 0
        if idx0.any():
            R0 = Rnt[idx0]
            t = torch.sqrt(torch.clamp(1.0 + R0[:, 0, 0] - R0[:, 1, 1] - R0[:, 2, 2], min=eps)) * 2.0
            qnt[idx0, 0] = (R0[:, 2, 1] - R0[:, 1, 2]) / t
            qnt[idx0, 1] = 0.25 * t
            qnt[idx0, 2] = (R0[:, 0, 1] + R0[:, 1, 0]) / t
            qnt[idx0, 3] = (R0[:, 0, 2] + R0[:, 2, 0]) / t

        idx1 = i == 1
        if idx1.any():
            R1 = Rnt[idx1]
            t = torch.sqrt(torch.clamp(1.0 + R1[:, 1, 1] - R1[:, 0, 0] - R1[:, 2, 2], min=eps)) * 2.0
            qnt[idx1, 0] = (R1[:, 0, 2] - R1[:, 2, 0]) / t
            qnt[idx1, 1] = (R1[:, 0, 1] + R1[:, 1, 0]) / t
            qnt[idx1, 2] = 0.25 * t
            qnt[idx1, 3] = (R1[:, 1, 2] + R1[:, 2, 1]) / t

        idx2 = i == 2
        if idx2.any():
            R2 = Rnt[idx2]
            t = torch.sqrt(torch.clamp(1.0 + R2[:, 2, 2] - R2[:, 0, 0] - R2[:, 1, 1], min=eps)) * 2.0
            qnt[idx2, 0] = (R2[:, 1, 0] - R2[:, 0, 1]) / t
            qnt[idx2, 1] = (R2[:, 0, 2] + R2[:, 2, 0]) / t
            qnt[idx2, 2] = (R2[:, 1, 2] + R2[:, 2, 1]) / t
            qnt[idx2, 3] = 0.25 * t

        q[nt] = qnt

    # normalize and enforce a canonical sign
    q = q / torch.clamp(q.norm(dim=-1, keepdim=True), min=eps)
    q = torch.where(q[:, :1] < 0, -q, q)  # w >= 0
    return q

# --------------------------
# main: Σ -> (quat, scale)
# --------------------------
@torch.no_grad()
def covar_to_quat_scale(
    cov: torch.Tensor,
    packed_triu: bool = True,
    sort_descending: bool = True,
    eps: float = 1e-8,
):
    """
    cov: [N,6] (packed upper) if packed_triu else [N,3,3] (full, row-major)
    returns:
      quat:  [N,4] (w,x,y,z)
      scale: [N,3] (std-devs)
    """
    if cov.dim() == 2 and cov.size(-1) == 6 and packed_triu:
        Sigma = unpack_triu6_to_mat(cov)
    elif cov.dim() == 3 and cov.size(-1) == 3 and cov.size(-2) == 3:
        Sigma = symmetrize3(cov)
    else:
        raise ValueError("cov should be [N,6] with packed_triu=True or [N,3,3].")

    # symmetric eigendecomposition
    # torch.linalg.eigh returns evals ascending, and orthonormal evecs in columns
    evals, evecs = torch.linalg.eigh(Sigma)  # evals: [N,3], evecs: [N,3,3]

    # clamp eigenvalues
    evals = torch.clamp(evals, min=eps)

    # sort to match your convention (e.g., largest axis first)
    if sort_descending:
        idx = torch.argsort(evals, dim=-1, descending=True)  # [N,3]
        batch = torch.arange(evals.shape[0], device=evals.device).unsqueeze(-1)
        evals = evals[batch, idx]                            # [N,3]
        evecs = evecs[batch.unsqueeze(-1), torch.arange(3, device=evals.device), idx]  # permute columns
        # The above fancy indexing builds evecs[:, :, idx[n]]; safer alternative:
        # evecs = torch.gather(evecs, 2, idx.unsqueeze(1).expand(-1,3,-1))

        # Prefer gather (uncomment); it’s clearer and avoids the advanced-indexing gotcha:
        evecs = torch.gather(evecs, 2, idx.unsqueeze(1).expand(-1, 3, -1))

    # scales are std-devs
    scales = torch.sqrt(evals)

    # ensure proper rotation: det(evecs) == +1; if negative, flip third column
    det = torch.det(evecs)
    neg = det < 0
    if neg.any():
        evecs[neg, :, 2] = -evecs[neg, :, 2]

    quat = rotmat_to_quat(evecs)
    return quat, scales

# --------------------------
# (Optional) quick check
# --------------------------
def _quat_scale_to_covar_ref(quat: torch.Tensor, scale: torch.Tensor, triu=True):
    """Reference Σ = R diag(s^2) R^T using PyTorch ops; for testing."""
    w, x, y, z = quat.unbind(-1)
    # rotation from unit quaternion
    R = torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y+z*w),   2*(x*z-y*w)], dim=-1),
        torch.stack([2*(x*y-z*w),   1-2*(x*x+z*z), 2*(y*z+x*w)], dim=-1),
        torch.stack([2*(x*z+y*w),   2*(y*z-x*w),   1-2*(x*x+y*y)], dim=-1),
    ], dim=-2)  # [N,3,3]
    S2 = torch.diag_embed(scale**2)
    Sigma = R @ S2 @ R.transpose(-1, -2)
    if triu:
        return torch.stack([Sigma[:,0,0], Sigma[:,0,1], Sigma[:,0,2],
                            Sigma[:,1,1], Sigma[:,1,2], Sigma[:,2,2]], dim=-1)
    return Sigma

if __name__ == "__main__":
    # quick evaluation
    torch.manual_seed(0)
    N = 4096
    quats = torch.randn(N, 4, device="cuda", dtype=torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3, device="cuda", dtype=torch.float32) * 0.8 + 0.2  # positive

    cov6 = _quat_scale_to_covar_ref(quats, scales, triu=True)  # [N,6]
    q_rec, s_rec = covar_to_quat_scale(cov6, packed_triu=True)

    # Rebuild covariance from recovered params and compare
    cov6_rec = _quat_scale_to_covar_ref(q_rec, s_rec, triu=True)
    rel_err = (cov6_rec - cov6).abs().mean() / (cov6.abs().mean() + 1e-12)
    print("relative L1 error on packed Σ:", float(rel_err))
