"""
Python runner script for model interpretation
Provides an easy interface to run interpretations on trained models
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess


def find_best_checkpoint(checkpoint_dir, model_name):
    """Find the best checkpoint for a given model based on val_auc"""
    model_dir = Path(checkpoint_dir) / model_name
    if not model_dir.exists():
        return None

    ckpt_files = list(model_dir.glob("epoch=*-val_auc=*.ckpt"))
    if not ckpt_files:
        return None

    # Sort by val_auc value and return the best
    best_ckpt = sorted(ckpt_files, key=lambda x: float(x.stem.split("val_auc=")[1]))[-1]
    return best_ckpt


def run_interpretation(model_name=None, **kwargs):
    """Run interpretation for specified model(s)"""

    # Default parameters
    params = {
        "checkpoint_dir": "out/checkpoint",
        "fake_root": "/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/fake_images",
        "real_root": "/gpfs/milgram/scratch60/gerstein/yz2483/animel2m_dataset/real_images/resized_img",
        "n_samples": 8,
        "save_dir": "interpretations",
        "img_size": 224,
        "batch_size": 1,
    }

    # Update with provided parameters
    params.update(kwargs)

    # Build command
    cmd = ["python", "interpretation.py"]

    for key, value in params.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])

    if model_name:
        cmd.extend(["--models", model_name])

    # Run the command
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running interpretation: {result.stderr}")
        return False

    print(result.stdout)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run model interpretation analysis")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to interpret. If not specified, runs all models.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run interpretation for all available models"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available models with checkpoints"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="out/checkpoint",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--n_samples", type=int, default=8, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="interpretations",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Available models
    all_models = [
        "convnext",
        "resnet",
        "vit",
        "frequency",
        "efficientnet",
        "dualstream",
        "lightweight",
        "anixplore",
    ]

    # List available models
    if args.list:
        print("Available models with checkpoints:")
        print("-" * 40)
        for model in all_models:
            ckpt = find_best_checkpoint(args.checkpoint_dir, model)
            if ckpt:
                val_auc = float(ckpt.stem.split("val_auc=")[1])
                print(f"  ✓ {model:<15} (best val_auc: {val_auc:.4f})")
            else:
                print(f"  ✗ {model:<15} (no checkpoint found)")
        return

    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Run interpretation
    if args.model:
        # Run for specific model
        print(f"\nInterpreting model: {args.model}")
        print("=" * 60)

        if args.model not in all_models:
            print(f"Error: Unknown model '{args.model}'")
            print(f"Available models: {', '.join(all_models)}")
            return

        ckpt = find_best_checkpoint(args.checkpoint_dir, args.model)
        if not ckpt:
            print(f"No checkpoint found for {args.model}")
            return

        print(f"Using checkpoint: {ckpt.name}")
        success = run_interpretation(
            model_name=args.model,
            checkpoint_dir=args.checkpoint_dir,
            n_samples=args.n_samples,
            save_dir=args.save_dir,
        )

        if success:
            print(f"\n✓ Interpretation complete for {args.model}")
            print(f"Results saved to {args.save_dir}/{args.model}_interpretations.png")

    elif args.all:
        # Run for all available models
        print("\nRunning interpretation for all available models")
        print("=" * 60)

        successful_models = []
        failed_models = []

        for model in all_models:
            ckpt = find_best_checkpoint(args.checkpoint_dir, model)
            if not ckpt:
                print(f"\nSkipping {model} - no checkpoint found")
                continue

            print(f"\nProcessing {model}...")
            print("-" * 40)

            success = run_interpretation(
                model_name=model,
                checkpoint_dir=args.checkpoint_dir,
                n_samples=args.n_samples,
                save_dir=args.save_dir,
            )

            if success:
                successful_models.append(model)
            else:
                failed_models.append(model)

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        if successful_models:
            print(f"✓ Successfully interpreted {len(successful_models)} models:")
            for model in successful_models:
                print(f"  - {model}")

        if failed_models:
            print(f"\n✗ Failed to interpret {len(failed_models)} models:")
            for model in failed_models:
                print(f"  - {model}")

        # Create combined visualization
        if len(successful_models) > 1:
            print("\nCreating combined comparison visualization...")
            run_interpretation(
                checkpoint_dir=args.checkpoint_dir,
                n_samples=args.n_samples,
                save_dir=args.save_dir,
            )
            print(
                f"Combined visualization saved to {args.save_dir}/all_models_comparison.png"
            )

        print(f"\nAll results saved in {args.save_dir}/")

    else:
        print("Please specify --model MODEL_NAME or use --all to run all models")
        print("Use --list to see available models with checkpoints")


if __name__ == "__main__":
    main()
