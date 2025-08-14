import argparse
import os

import torch
import yaml

from efficientvit.seg_model_zoo import create_efficientvit_seg_model
from efficientvit.segcore.data_provider import create_data_loader, create_dataset
from efficientvit.segcore.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to evaluation config file"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Setup environment
    gpu_list = config["runtime"].get("gpu", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    print(f"Using GPU(s): {gpu_list}")

    # Create dataset and dataloader
    dataset = create_dataset(config)
    data_loader = create_data_loader(dataset, config)
    print(f"Loaded dataset {config['dataset']['name']} with {len(dataset)} images.")

    # Create model
    model = create_efficientvit_seg_model(
        config["model"]["name"], weight_url=config["model"]["weight_url"]
    )
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    print(f"Loaded model {config['model']['name']}.")

    # Create evaluator and run
    evaluator = Evaluator(model, data_loader, config)
    results = evaluator.evaluate()

    # Print summary
    print("\n--- Evaluation Summary ---")
    if "mIOU" in results:
        print(f"  mIOU: {results['mIOU']:.3f}%")
    if "fps" in results:
        print(f"  Inference FPS: {results['fps']:.2f}")
    
    save_path = config.get("save_path")
    if save_path:
        print(f"\nResults saved to: {save_path}")
    print("------------------------")


if __name__ == "__main__":
    main()
