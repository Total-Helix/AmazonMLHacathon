"""
ML Challenge 2026: End-to-End Pipeline Execution Entry Point
Role: The Judge (Role 3: ML & Feature Engineering)

Usage:
    python main.py --train --infer
    python main.py --use-real-data --train --infer
"""

import argparse
from src.config import CONFIG
from src.train import Trainer
from src.inference import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Run Amazon ML Challenge 2026 ER Judge Pipeline")
    parser.add_argument("--use-real-data", action="store_true", help="Toggle pipeline to use real dataset from student_resource/dataset")
    parser.add_argument("--use-fake-data", action="store_true", help="Toggle pipeline to use synthetic bootstrap dataset")
    parser.add_argument("--train", action="store_true", default=True, help="Train LightGBM model and optimize F0.5 threshold")
    parser.add_argument("--infer", action="store_true", default=True, help="Run inference and generate submission TSVs")

    args = parser.parse_args()

    if args.use_real_data:
        CONFIG.use_fake_data = False
        print("[Main] Mode set to: REAL DATA")
    elif args.use_fake_data:
        CONFIG.use_fake_data = True
        print("[Main] Mode set to: SYNTHETIC BOOTSTRAP DATA")
    else:
        print(f"[Main] Mode (from config.py): {'SYNTHETIC BOOTSTRAP DATA' if CONFIG.use_fake_data else 'REAL DATA'}")

    if args.train:
        print("\n" + "=" * 60)
        print("STAGE 1: MODEL TRAINING & F0.5 THRESHOLD OPTIMIZATION")
        print("=" * 60)
        trainer = Trainer(config=CONFIG)
        trainer.run()

    if args.infer:
        print("\n" + "=" * 60)
        print("STAGE 2: INFERENCE & SUBMISSION GENERATION")
        print("=" * 60)
        engine = InferenceEngine(config=CONFIG)
        engine.run_inference()


if __name__ == "__main__":
    main()
