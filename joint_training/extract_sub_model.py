"""Extract a sub-model from merged QwenJoint weights and save as standard Qwen3ForCausalLM.

Usage:
    python recipe/joint_training/extract_sub_model.py \
        --joint_model_path /data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100 \
        --output_path /data-1/model_weights/EXP-04_Joint-MiniRL-1.7B-MATH/step_100_model2 \
        --sub_model_index 1

    --sub_model_index 0 = model1 (anchor/frozen model)
    --sub_model_index 1 = model2 (trainable model, default)
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from verl.models.joint_model.weight_utils import extract_sub_model_weights


def main():
    parser = argparse.ArgumentParser(description="Extract sub-model from joint model weights")
    parser.add_argument("--joint_model_path", type=str, required=True, help="Path to merged joint model directory")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory for extracted model")
    parser.add_argument("--sub_model_index", type=int, default=1, choices=[0, 1],
                        help="Which sub-model to extract: 0=model1 (anchor), 1=model2 (trainable, default)")
    args = parser.parse_args()

    joint_path = Path(args.joint_model_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load joint model state dict
    safetensors_files = list(joint_path.glob("model*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No model*.safetensors found in {joint_path}")

    print(f"Loading joint model weights from {joint_path} ...")
    state_dict = {}
    for f in sorted(safetensors_files):
        state_dict.update(load_file(str(f)))

    total_params = sum(v.numel() for v in state_dict.values())
    print(f"  Joint model total parameters: {total_params:,}")

    # 2. Extract sub-model weights
    print(f"Extracting sub_models[{args.sub_model_index}] ...")
    sub_state_dict = extract_sub_model_weights(state_dict, sub_model_index=args.sub_model_index)

    if not sub_state_dict:
        raise ValueError(f"No weights found for sub_models.{args.sub_model_index}")

    sub_params = sum(v.numel() for v in sub_state_dict.values())
    print(f"  Sub-model parameters: {sub_params:,}")

    # 3. Save as safetensors
    output_safetensors = output_path / "model.safetensors"
    print(f"Saving to {output_safetensors} ...")
    save_file(sub_state_dict, str(output_safetensors))

    # 4. Create standard Qwen3 config (strip joint-specific fields)
    with open(joint_path / "config.json") as f:
        joint_config = json.load(f)

    # Remove joint-specific fields and set standard Qwen3 type
    for key in ("fusion_lambda", "freeze_model1", "num_sub_models"):
        joint_config.pop(key, None)
    joint_config["model_type"] = "qwen3"
    joint_config["architectures"] = ["Qwen3ForCausalLM"]
    joint_config.pop("auto_map", None)

    with open(output_path / "config.json", "w") as f:
        json.dump(joint_config, f, indent=2)
        f.write("\n")

    # 5. Copy tokenizer files
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = joint_path / name
        if src.exists():
            shutil.copy2(src, output_path / name)

    # 6. Copy or create generation_config.json
    gen_config_src = joint_path / "generation_config.json"
    if gen_config_src.exists():
        shutil.copy2(gen_config_src, output_path / "generation_config.json")

    print(f"\nDone. Standard Qwen3ForCausalLM saved to: {output_path}")
    print(f"  Sub-model: {'model1 (anchor)' if args.sub_model_index == 0 else 'model2 (trainable)'}")


if __name__ == "__main__":
    main()
