#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Recreate the config/assets without reading or copying checkpoint weights."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

SOURCE = Path("/mxstorage/pde_ai/models/llm/Qwen/Qwen3.8-Flash-Next")
ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "configuration.json",
    "LICENSE",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="A new directory; existing directories are rejected.",
    )
    args = parser.parse_args()
    original = json.loads((args.source / "config.json").read_text())
    text = original["text_config"]
    assert original["architectures"] == ["Qwen4ExpForConditionalGeneration"]
    assert text["num_hidden_layers"] == 48
    assert text["layer_types"][:4] == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert text["ple_layer_ids"] == [2]  # One-based: keep its original position.
    assert original["vision_config"]["depth"] == 27
    assert original["vision_config"]["deepstack_visual_indexes"] == []
    assert text["mtp_num_hidden_layers"] == text["mtp"]["num_hidden_layers"] == 1
    for name in ASSETS:
        if not (args.source / name).is_file():
            raise FileNotFoundError(args.source / name)
    reduced = copy.deepcopy(original)
    reduced["text_config"]["num_hidden_layers"] = 3
    reduced["text_config"]["layer_types"] = [text["layer_types"][i] for i in (0, 1, 3)]
    reduced["text_config"]["full_attention_interval"] = 3
    reduced["vision_config"]["depth"] = 1
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(reduced, indent=2) + "\n")
    for name in ASSETS:
        shutil.copyfile(args.source / name, args.output / name, follow_symlinks=True)
    manifest = {
        "source": str(args.source.resolve()),
        "mode": "dummy",
        "text_layer_mapping_zero_based": {"0": 0, "1": 1, "3": 2},
        "vision_layer_mapping_zero_based": {"0": 0},
        "mtp_mapping": {
            "checkpoint_namespace": "mtp.layers.0",
            "old_runtime_layer": 48,
            "new_runtime_layer": 3,
        },
        "changes": {
            "text_config.num_hidden_layers": [48, 3],
            "text_config.layer_types": [
                text["layer_types"],
                reduced["text_config"]["layer_types"],
            ],
            "text_config.full_attention_interval": [4, 3],
            "vision_config.depth": [27, 1],
        },
        "files": {},
    }
    for name in ("config.json", *ASSETS):
        p = args.output / name
        manifest["files"][name] = {
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    (args.output / "trim_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
