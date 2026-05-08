#!/usr/bin/env python3
"""Prepare the first controlled Qwen3-LoRA baseline suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PACKET_CONFIGS = [
    "configs/server/project_baselines/tallrec_amazon_beauty_packet.yaml",
    "configs/server/project_baselines/openp5_amazon_beauty_packet.yaml",
    "configs/server/project_baselines/dealrec_amazon_beauty_packet.yaml",
    "configs/server/project_baselines/lc_rec_amazon_beauty_packet.yaml",
]

DEFAULT_CONTROL_CONFIGS = [
    "configs/server/controlled_baselines/tallrec_qwen3_lora_amazon_beauty.yaml",
    "configs/server/controlled_baselines/openp5_style_qwen3_lora_amazon_beauty.yaml",
    "configs/server/controlled_baselines/dealrec_qwen3_lora_amazon_beauty.yaml",
    "configs/server/controlled_baselines/lc_rec_qwen3_lora_amazon_beauty.yaml",
]

ADDED_OFFICIAL_CANDIDATE_PACKET_CONFIGS = [
    "configs/server/project_baselines/llara_amazon_beauty_packet.yaml",
    "configs/server/project_baselines/llmesr_amazon_beauty_packet.yaml",
]

ADDED_OFFICIAL_CANDIDATE_CONTROL_CONFIGS = [
    "configs/server/controlled_baselines/llara_qwen3_lora_amazon_beauty.yaml",
    "configs/server/controlled_baselines/llmesr_qwen3_lora_amazon_beauty.yaml",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-name", default="qwen3_base_adapter_main4_amazon_beauty")
    parser.add_argument("--include-added-official-candidates", action="store_true")
    args = parser.parse_args()
    manifests = []
    packet_configs = list(DEFAULT_PACKET_CONFIGS)
    control_configs = list(DEFAULT_CONTROL_CONFIGS)
    if args.include_added_official_candidates:
        packet_configs.extend(ADDED_OFFICIAL_CANDIDATE_PACKET_CONFIGS)
        control_configs.extend(ADDED_OFFICIAL_CANDIDATE_CONTROL_CONFIGS)
    for config in packet_configs:
        _run(["python", "scripts/prepare_project_baseline_packet.py", "--config", config])
    for config in control_configs:
        payload = _run(["python", "scripts/prepare_qwen_lora_controlled_baseline.py", "--config", config])
        manifests.append(json.loads(payload))
    suite_dir = ROOT / "outputs" / "server_training" / "controlled_baselines" / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_manifest = {
        "suite_name": args.suite_name,
        "purpose": "Controlled comparison suite with shared Qwen3-8B base model and LoRA adaptation; baselines use official source/default-or-reported-optimal hyperparameters when promoted beyond adapter-pilot status.",
        "base_model_policy": "shared_qwen3_8b_base_model",
        "adaptation": "lora",
        "lora_required": True,
        "adapter_training_policy": "baseline_official_lora_with_project_modules",
        "hyperparameter_policy": "official_default_or_reported_optimal_for_baselines",
        "baseline_hparam_tuning_allowed": False,
        "ours_hparam_policy": "validation_tunable_no_test_tuning",
        "baseline_count": len(manifests),
        "baselines": manifests,
        "is_experiment_result": False,
        "is_paper_result": False,
    }
    (suite_dir / "suite_manifest.json").write_text(json.dumps(suite_manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    _write_run_queue(suite_dir / "server_run_queue.sh", manifests)
    print(json.dumps(suite_manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, check=True, capture_output=True)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.stdout


def _write_run_queue(path: Path, manifests: list[dict]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd ~/projects/TRUCE-Rec",
        "source ${QWEN_LORA_ENV:-$HOME/projects/TALLRec/.venv_tallrec/bin/activate}",
        "",
        "# Smoke each baseline first. Remove --max-* flags for full runs after the smoke succeeds.",
    ]
    for manifest in manifests:
        name = manifest["controlled_baseline_name"]
        manifest_path = Path(manifest["output_dir"]) / "controlled_baseline_manifest.json"
        lines.extend([
            "",
            f"echo '===== smoke {name} ====='",
            "python scripts/run_qwen_lora_controlled_baseline.py \\",
            f"  --manifest {manifest_path} \\",
            "  --max-train-examples 128 \\",
            "  --max-steps 5 \\",
            "  --max-score-rows 2 \\",
            "  --trust-remote-code",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
