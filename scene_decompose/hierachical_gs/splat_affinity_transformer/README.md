# Splat Affinity Transformer
- This module is to determine the affinity matrix between multiple splats. 
- If the affinity score is above 0, we will merge them into one semantic block, otherwise, we will seperate them
- The architecture is unclear yet, but basically, we will have multiple self-attention layer with learnable parameters build on top of each other
- The training stages are mainly seperated into two parts: 1. Image Affinity Pretraining and 2. 3D envrionment self-supervised finetuning. 

## Image Affinity Pretraining
- FastSAM distillation, we utilize the fast SAM generated result as the input to segment multiple images, the pixel that with each masks are regarded as affinity score 1, otherwise, -1
- DINOv3 feature distillation, we utilize DINOv3 as the input feature to SAT, the input token length is defined as 64 by 64 
- Some tricks including perturbation on features, adding small amount of noise to original dinov3 features

## Splat Self Supervised Learning
- After using K-means to seperate our secen into different semantic region, we sample from same cluster for positive example, different cluster for negative example.
- And we still add a small perturbation for robustness

