#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Offline CPU asset checks. Does not allocate model parameters or use GPUs."""

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    model = Path(__file__).resolve().parent
    raw = json.loads((model / "config.json").read_text())
    manifest = json.loads((model / "trim_manifest.json").read_text())
    for name, metadata in manifest["files"].items():
        p = model / name
        assert not p.is_symlink(), p
        assert hashlib.sha256(p.read_bytes()).hexdigest() == metadata["sha256"], p
    assert not list(model.glob("*.safetensors"))
    assert not (model / "model.safetensors.index.json").exists()
    assert raw["text_config"]["num_hidden_layers"] == 3
    assert raw["text_config"]["layer_types"] == [
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert raw["text_config"]["full_attention_interval"] == 3
    assert raw["text_config"]["ple_layer_ids"] == [2]
    assert raw["vision_config"]["depth"] == 1
    assert raw["vision_config"]["deepstack_visual_indexes"] == []
    assert raw["language_model_only"] is False
    assert raw["text_config"]["mtp_num_hidden_layers"] == 1
    assert raw["text_config"]["mtp"]["num_hidden_layers"] == 1
    env = {"python": sys.version, "executable": sys.executable, "packages": {}}
    for name in ("vllm", "vllm_metax", "transformers", "torch"):
        env["packages"][name] = {
            "distribution_version": importlib.metadata.version(name.replace("_", "-")),
            "origin": importlib.util.find_spec(name).origin,
        }
    print("ENVIRONMENT " + json.dumps(env), flush=True)

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer, GenerationConfig

    normalized = {}
    try:
        # Mirror the intended config registration without loading the full
        # plugin. Plugin startup is checked independently.
        from vllm_metax.models.qwen4_exp.config import (
            Qwen4ExpConfig,
            Qwen4ExpTextConfig,
        )

        AutoConfig.register("qwen4_exp", Qwen4ExpConfig, exist_ok=True)
        AutoConfig.register("qwen4_exp_text", Qwen4ExpTextConfig, exist_ok=True)
        config = AutoConfig.from_pretrained(model, local_files_only=True)
        normalized = {
            "status": "PASS",
            "class": type(config).__module__ + "." + type(config).__name__,
        }
        assert config.text_config.short_conv_layer_ids == [1]
        assert config.text_config.short_conv_state_shape == (10240, 9)
        assert config.text_config.ngram_context_len == 2
        assert config.text_config.decoder_sparse_step == 1
        assert config.text_config.mlp_only_layers == []
    except (ImportError, ValueError) as exc:
        normalized = {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}
    text = SimpleNamespace(**raw["text_config"])
    assert text.num_hidden_layers == len(text.layer_types) == 3
    assert text.num_experts == 512 and text.num_experts_per_tok == 10
    assert text.shared_expert_intermediate_size == 640
    assert text.indexer_budget // text.indexer_compress_ratio == 512
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    sample = tokenizer.encode("Hello, world!")
    assert sample and max(sample) < text.vocab_size
    assert max(tokenizer.get_vocab().values()) < text.vocab_size
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    assert chat and max(chat) < text.vocab_size
    generation = GenerationConfig.from_pretrained(model, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    import numpy as np
    from PIL import Image

    image = processor.image_processor(
        images=[Image.new("RGB", (256, 256))], return_tensors="pt"
    )
    video = processor.video_processor(
        videos=[np.zeros((4, 256, 256, 3), dtype=np.uint8)],
        do_sample_frames=False,
        return_tensors="pt",
    )
    result = {
        "structural_and_asset_checks": "PASS",
        "normalized_config": normalized,
        "registration": "explicit MetaX config registration attempted; full plugin checked separately",
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size": len(tokenizer),
        "sample_token_ids": sample,
        "chat_tokens": len(chat),
        "processor_class": type(processor).__name__,
        "image_grid_thw": image["image_grid_thw"].tolist(),
        "video_grid_thw": video["video_grid_thw"].tolist(),
        "generation_eos_token_id": generation.eos_token_id,
        "gpu_startup": "UNVERIFIED",
        "generation": "UNVERIFIED",
        "mtp": "UNVERIFIED",
    }
    print("RESULT " + json.dumps(result, indent=2), flush=True)
    (model / "validation_cpu.json").write_text(
        json.dumps({"environment": env, "results": result}, indent=2) + "\n"
    )
    if normalized["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
