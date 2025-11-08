"""
Simple pipeline orchestrator that parses YAML configuration files.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import argparse
import os
import subprocess
import sys


@dataclass
class TrainConfig:
    """Training configuration."""
    steps: int = 10000


@dataclass
class LiftingConfig:
    """Feature lifting configuration."""
    feature_encoder: str = "dinov3"
    model_path: str = ""
    output_postfix: str = "dinov3"


@dataclass
class HierarchicalGSConfig:
    """Hierarchical Gaussian Splat configuration."""
    attention_start_layer: int = 3
    hops_for_attention: int = 3
    output_postfix: str = "dinov3"


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    dataset_path: str = ""
    output_path: str = ""
    train: TrainConfig = field(default_factory=TrainConfig)
    lifting: LiftingConfig = field(default_factory=LiftingConfig)
    hierachical_gs: HierarchicalGSConfig = field(default_factory=HierarchicalGSConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'PipelineConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            PipelineConfig instance
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Remove hydra section if present (Hydra-specific config)
        if 'hydra' in config_dict:
            del config_dict['hydra']
        
        # Parse nested configs
        train_config = TrainConfig(**config_dict.get('train', {}))
        lifting_config = LiftingConfig(**config_dict.get('lifting', {}))
        hierarchical_gs_config = HierarchicalGSConfig(**config_dict.get('hierachical_gs', {}))
        
        return cls(
            dataset_path=config_dict.get('dataset_path', ''),
            output_path=config_dict.get('output_path', ''),
            train=train_config,
            lifting=lifting_config,
            hierachical_gs=hierarchical_gs_config
        )
    
    def get_scene_paths(self) -> list[Path]:
        """
        Get list of scene paths from dataset_path.
        Assumes dataset_path contains subdirectories for each scene.
        
        Returns:
            List of Path objects for each scene directory
        """
        dataset_dir = Path(self.dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_dir}")
        
        # Get all subdirectories as scenes
        scenes = [d for d in dataset_dir.iterdir() if d.is_dir()]
        return sorted(scenes)
    
    def get_output_dir(self, scene_name: Optional[str] = None) -> Path:
        """
        Get output directory path for a scene.
        
        Args:
            scene_name: Optional scene name. If None, returns base output path.
            
        Returns:
            Path to output directory
        """
        output_dir = Path(self.output_path)
        if scene_name:
            output_dir = output_dir / scene_name
        return output_dir


def load_config(config_path: Union[str, Path] = "eval.yml") -> PipelineConfig:
    """
    Convenience function to load pipeline configuration.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        PipelineConfig instance
    """
    return PipelineConfig.from_yaml(config_path)


class runner:
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def run(self):
        #self.train()
        #self.lifting()
        #self.hierachical_gs()
        self.eval()

    def _run_command(self, cmd: List[str], cwd: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
        """
        Run a shell command and return the result.
        
        Args:
            cmd: List of command and arguments (e.g., ['python', 'script.py', '--arg', 'value'])
            cwd: Working directory to run the command in
            check: If True, raise exception on non-zero exit code
            
        Returns:
            CompletedProcess object
        """
        print(f"Running: {' '.join(cmd)}")
        if cwd:
            print(f"Working directory: {cwd}")
        
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            capture_output=False,  # Set to True if you want to capture output
            text=True
        )
        
        if result.returncode != 0:
            print(f"Command failed with exit code {result.returncode}")
        else:
            print(f"Command completed successfully")
        
        return result

    def train(self):
        """Run training for each scene."""
        scene_names = [d for d in os.listdir(self.config.dataset_path) 
                      if os.path.isdir(os.path.join(self.config.dataset_path, d))]
        
        for scene_name in scene_names:
            scene_path = os.path.join(self.config.dataset_path, scene_name)
            output_dir = self.config.get_output_dir(scene_name)
            
            print(f"\n{'='*60}")
            print(f"Training scene: {scene_name}")
            print(f"{'='*60}")
            print(f"Scene path: {scene_path}")
            print(f"Output directory: {output_dir}")
            print(f"Steps: {self.config.train.steps}")
            
            # Example: Run training command
            # Adjust the command based on your actual training script
            cmd = [
                sys.executable,  # Use current Python interpreter
                "fix_trainer.py",  # Your training script
                "feature",
                "--data-dir", scene_path,
                "--disable-viewer",
                "--max-steps", str(self.config.train.steps),
                "--save-steps", str(self.config.train.steps),
                "--result-dir", str(output_dir),
                # Add other training arguments as needed
            ]
            
            try:
                self._run_command(cmd, cwd=os.getcwd())
                print(f"✓ Training completed for {scene_name}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Training failed for {scene_name}: {e}")
                # Decide whether to continue or stop on error
                # raise  # Uncomment to stop on error

    def lifting(self):
        """Run feature lifting for each scene."""
        scene_names = [d for d in os.listdir(self.config.dataset_path) 
                      if os.path.isdir(os.path.join(self.config.dataset_path, d))]
        
        for scene_name in scene_names:
            scene_path = os.path.join(self.config.dataset_path, scene_name)
            output_dir = self.config.get_output_dir(scene_name)
            
            # Input checkpoint (from training)
            ckpt_path = output_dir / "ckpts" / f"ckpt_{self.config.train.steps-1}_rank0.pt"
            # Output feature path
            feature_path = output_dir / "ckpts" / f"ckpt_{self.config.train.steps-1}_rank0_{self.config.lifting.output_postfix}.pt"
            
            if not ckpt_path.exists():
                print(f"Warning: Checkpoint not found for {scene_name}: {ckpt_path}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Feature lifting for scene: {scene_name}")
            print(f"{'='*60}")
            
            # Example command - adjust based on your actual feature lifting script
            model_type = self.config.lifting.feature_encoder.split("_")[0]
            cmd = [
                sys.executable,
                "distill_wrapper.py",  # Your feature lifting script
                "--data.dir", scene_path,
                "--distill.ckpt", str(ckpt_path),
                "--distill.feature-ckpt", str(feature_path),
                "--model.model-type", model_type,
                "--model.backbone", "vit_large_patch16_dinov3.lvd1689m",  # Adjust as needed
                "--model.model-path", self.config.lifting.model_path,
            ]
            
            try:
                self._run_command(cmd, cwd=os.getcwd())
                print(f"✓ Feature lifting completed for {scene_name}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Feature lifting failed for {scene_name}: {e}")
                # raise  # Uncomment to stop on error

    def hierachical_gs(self):
        """Run hierarchical GS construction for each scene."""
        scene_names = [d for d in os.listdir(self.config.dataset_path) 
                      if os.path.isdir(os.path.join(self.config.dataset_path, d))]
        
        for scene_name in scene_names:
            output_dir = self.config.get_output_dir(scene_name)
            
            # Input paths
            ckpt_path = output_dir / "ckpts" / f"ckpt_{self.config.train.steps-1}_rank0.pt"
            feature_path = output_dir / "ckpts" / f"ckpt_{self.config.train.steps-1}_rank0_{self.config.lifting.output_postfix}.pt"
            # Output path
            
            h_gs_path = output_dir / "ckpts" / f"h_gs_{self.config.hierachical_gs.output_postfix}.pt"
            
            if not ckpt_path.exists() or not feature_path.exists():
                print(f"Warning: Required files not found for {scene_name}")
                print(f"Checkpoint: {ckpt_path}")
                print(f"Feature: {feature_path}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Hierarchical GS construction for scene: {scene_name}")
            print(f"{'='*60}")
            
            # Example command - adjust based on your actual script
            cmd = [
                sys.executable,
                "construct_hierachical_gs.py",
                "-c", str(ckpt_path),
                "-f", str(feature_path),
                "-o", str(h_gs_path),
                "-asl", str(self.config.hierachical_gs.attention_start_layer),
                "-hfa", str(self.config.hierachical_gs.hops_for_attention),
                "-v"
            ]
            
            try:
                self._run_command(cmd, cwd=os.getcwd())
                print(f"✓ Hierarchical GS construction completed for {scene_name}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Hierarchical GS construction failed for {scene_name}: {e}")
                # raise  # Uncomment to stop on error
    
    def eval(self):
        """Run evaluation for each scene."""
        scene_names = [d for d in os.listdir(self.config.dataset_path) 
                      if os.path.isdir(os.path.join(self.config.dataset_path, d))]
        
        for scene_name in scene_names:
            output_dir = self.config.get_output_dir(scene_name)
            
            # Input path (could be hierarchical GS or regular checkpoint)
            h_gs_path = output_dir / "ckpts" / f"h_gs_{self.config.hierachical_gs.output_postfix}.pt"
            scene_path = os.path.join(self.config.dataset_path, scene_name)
            
            if not h_gs_path.exists():
                print(f"Warning: Hierarchical GS not found for {scene_name}: {h_gs_path}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Evaluation for scene: {scene_name}")
            print(f"{'='*60}")
            
            # Example command - adjust based on your actual evaluation script
            cmd = [
                sys.executable,
                "eval_wrapper.py",
                "--data.dir", scene_path,
                "--distill.ckpt", str(h_gs_path),
                "--distill.method", "H_GS",
                "--model.model-type", "jafar",  # or "jafar"
                "--model.backbone", "vit_large_patch16_dinov3.lvd1689m",
                "--model.model-path", self.config.lifting.model_path,
            ]
            
            try:
                self._run_command(cmd, cwd=os.getcwd())
                print(f"✓ Evaluation completed for {scene_name}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Evaluation failed for {scene_name}: {e}")
                # raise  # Uncomment to stop on error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline configuration parser and runner")
    parser.add_argument(
        "--config",
        type=str,
        default="eval.yml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--list-scenes",
        action="store_true",
        help="List all scenes in the dataset"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the complete pipeline (train -> lifting -> hierarchical_gs -> eval)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["train", "lifting", "hierachical_gs", "eval"],
        help="Run a specific pipeline stage"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("=" * 60)
    print("Pipeline Configuration")
    print("=" * 60)
    print(f"Dataset Path: {config.dataset_path}")
    print(f"Output Path: {config.output_path}")
    print(f"\nTraining:")
    print(f"  Steps: {config.train.steps}")
    print(f"\nFeature Lifting:")
    print(f"  Encoder: {config.lifting.feature_encoder}")
    print(f"  Model Path: {config.lifting.model_path}")
    print(f"  Output Postfix: {config.lifting.output_postfix}")
    print(f"\nHierarchical GS:")
    print(f"  Attention Start Layer: {config.hierachical_gs.attention_start_layer}")
    print(f"  Hops for Attention: {config.hierachical_gs.hops_for_attention}")
    print(f"  Output Postfix: {config.hierachical_gs.output_postfix}")

    if args.run:
        pipeline_runner = runner(config)
        pipeline_runner.run()
    

