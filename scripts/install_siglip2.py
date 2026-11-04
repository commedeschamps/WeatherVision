"""Install pinned SigLIP 2 assets and verify real CPU/accelerator inference."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install_siglip_assets(offline=False):
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import (
        AutoImageProcessor,
        AutoTokenizer,
        SiglipConfig,
        SiglipTextModel,
        SiglipVisionModel,
    )
    from safetensors import safe_open
    from PIL import Image
    from src.siglip2 import MODEL_ID, REVISION, ASSET_DIR
    from src.utils import seed_everything, sha256_file, write_json, select_device

    seed_everything(42)
    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            cache_dir=str(ROOT / ".cache/huggingface"),
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.safetensors",
                "README.md",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "tokenizer.model",
            ],
            local_files_only=offline,
        )
    )
    expected = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
    if sha256_file(snapshot / "model.safetensors") != expected:
        raise ValueError(
            "Pinned official pretrained weights failed integrity verification"
        )
    config = SiglipConfig.from_pretrained(snapshot, local_files_only=True)
    encoder = SiglipVisionModel.from_pretrained(
        snapshot,
        config=config.vision_config,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    processor = AutoImageProcessor.from_pretrained(
        snapshot, local_files_only=True, use_fast=False
    )
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(ASSET_DIR / "vision")
    processor.save_pretrained(ASSET_DIR / "vision")
    input_tensor = processor(
        images=Image.new("RGB", (320, 180), (90, 140, 190)), return_tensors="pt"
    )["pixel_values"]
    devices = {}
    reference = None
    for name in (
        ["cpu"]
        + (["mps"] if torch.backends.mps.is_available() else [])
        + (["cuda"] if torch.cuda.is_available() else [])
    ):
        encoder.to(name)
        with torch.inference_mode():
            output = encoder(pixel_values=input_tensor.to(name))
        features = output.pooler_output.cpu()
        devices[name] = {
            "input_shape": list(input_tensor.shape),
            "patch_features": list(output.last_hidden_state.shape),
            "pooled": list(features.shape),
            "finite": bool(torch.isfinite(features).all()),
        }
        if reference is None:
            reference = features.clone()
        else:
            devices[name]["cpu_max_abs_difference"] = float(
                (features - reference).abs().max()
            )
    report = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "license": "Apache-2.0",
        "transformers": transformers.__version__,
        "model_class": type(encoder).__name__,
        "vision_config": config.vision_config.to_dict(),
        "processor": processor.to_dict(),
        "parameters": sum(p.numel() for p in encoder.parameters()),
        "devices": devices,
        "official_weights_sha256": expected,
        "files": {
            p.name: sha256_file(p)
            for p in (ASSET_DIR / "vision").iterdir()
            if p.is_file()
        },
    }
    write_json(ASSET_DIR / "provenance.json", report)
    write_json(ROOT / "outputs/siglip2/pretrained_verification.json", report)
    del encoder
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.save_pretrained(ASSET_DIR / "tokenizer")
    text_model = SiglipTextModel.from_pretrained(
        snapshot,
        config=config.text_config,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    experiment = json.loads((ROOT / "configs/siglip2.json").read_text())
    banks = []
    with torch.inference_mode():
        for prompts in experiment["prompt_sets"]:
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=config.text_config.max_position_embeddings,
                truncation=True,
                return_tensors="pt",
            )
            bank = text_model(**tokens).pooler_output
            banks.append(torch.nn.functional.normalize(bank, dim=-1).cpu())
    with safe_open(
        snapshot / "model.safetensors", framework="pt", device="cpu"
    ) as weights:
        scale = weights.get_tensor("logit_scale")
        bias = weights.get_tensor("logit_bias")
    torch.save(
        {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "prompt_sets": experiment["prompt_sets"],
            "text_embeddings": banks,
            "logit_scale": scale,
            "logit_bias": bias,
        },
        ASSET_DIR / "prompt_banks.pt",
    )
    print(
        json.dumps(
            {
                "model": MODEL_ID,
                "revision": REVISION,
                "devices": devices,
                "text_bank_shapes": [list(x.shape) for x in banks],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    install_siglip_assets(args.offline)
