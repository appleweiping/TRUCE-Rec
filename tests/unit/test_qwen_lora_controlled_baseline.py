import csv
import json
from pathlib import Path
import pytest

from scripts.prepare_qwen_lora_controlled_baseline import prepare_controlled_baseline


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _tallrec_packet(tmp_path: Path) -> Path:
    packet = tmp_path / "tallrec_packet"
    project = packet / "tallrec"
    project.mkdir(parents=True)
    rows = [
        {
            "instruction": "Answer Yes. or No.",
            "input": "User history: One\nCandidate item: Two",
            "output": "Yes.",
            "truce_row_id": "e1::i2",
        },
        {
            "instruction": "Answer Yes. or No.",
            "input": "User history: One\nCandidate item: Three",
            "output": "No.",
            "truce_row_id": "e1::i3",
        },
    ]
    (project / "train.json").write_text(json.dumps(rows), encoding="utf-8")
    (project / "valid.json").write_text(json.dumps(rows[:1]), encoding="utf-8")
    (project / "test.json").write_text(json.dumps(rows), encoding="utf-8")
    _write_jsonl(
        project / "test_row_map.jsonl",
        [
            {"example_id": "e1", "user_id": "u1", "item_id": "i2", "split": "test"},
            {"example_id": "e1", "user_id": "u1", "item_id": "i3", "split": "test"},
        ],
    )
    (project / "candidate_scores_template.csv").write_text("example_id,user_id,item_id,score\n", encoding="utf-8")
    return packet


def _openp5_packet(tmp_path: Path) -> Path:
    packet = tmp_path / "openp5_packet"
    project = packet / "openp5"
    project.mkdir(parents=True)
    with (project / "item_id_mapping.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "openp5_token", "title"])
        writer.writeheader()
        writer.writerow({"item_id": "i1", "openp5_token": "<item_1>", "title": "One"})
        writer.writerow({"item_id": "i2", "openp5_token": "<item_2>", "title": "Two"})
        writer.writerow({"item_id": "i3", "openp5_token": "<item_3>", "title": "Three"})
    task = {
        "example_id": "e1",
        "user_id": "u1",
        "history_item_ids": ["i1"],
        "target_item_id": "i2",
        "candidate_item_ids": ["i2", "i3"],
        "split": "test",
    }
    _write_jsonl(project / "train_sequential_tasks.jsonl", [{**task, "split": "train"}])
    _write_jsonl(project / "valid_sequential_tasks.jsonl", [{**task, "split": "valid"}])
    _write_jsonl(project / "test_sequential_tasks.jsonl", [task])
    (project / "candidate_scores_template.csv").write_text("example_id,user_id,item_id,score\n", encoding="utf-8")
    return packet


def test_tallrec_qwen_lora_controlled_contract(tmp_path: Path) -> None:
    packet = _tallrec_packet(tmp_path)
    config = {
        "project": "tallrec",
        "controlled_baseline_name": "tallrec_qwen3_lora_tiny",
        "base_model": "/models/qwen",
        "packet_dir": str(packet),
        "output_dir": str(tmp_path / "out"),
        "seed": 13,
        "lora": {"r": 16, "alpha": 32},
        "training": {"num_train_epochs": 1},
        "scoring": {"type": "pairwise_yes_no_likelihood"},
    }
    manifest = prepare_controlled_baseline(config=config, config_path=tmp_path / "cfg.yaml")
    train = Path(manifest["files"]["train_sft"]).read_text(encoding="utf-8").splitlines()
    first = json.loads(train[0])
    assert first["messages"][0]["role"] == "user"
    assert first["messages"][1]["content"] == "Yes."
    score = json.loads(Path(manifest["files"]["test_score_plan"]).read_text(encoding="utf-8").splitlines()[0])
    assert score["prompt"].startswith("Answer Yes. or No.")
    assert score["candidate_outputs"] == ["Yes."]
    assert manifest["implementation_fidelity"] == "controlled_adapter_pilot"
    assert manifest["official_fidelity_audit_required"] is True
    assert manifest["base_model_policy"] == "shared_qwen3_8b_base_model"
    assert manifest["adaptation"] == "lora"
    assert manifest["lora_required"] is True
    assert manifest["adapter_training_policy"] == "baseline_official_lora_with_project_modules"
    assert manifest["hyperparameter_policy"] == "official_default_or_reported_optimal_for_baselines"
    assert manifest["baseline_hparam_tuning_allowed"] is False
    assert manifest["provenance"]["claim_label"] == "controlled_adapter_pilot"
    assert manifest["provenance"]["official_hyperparameters_reused"] is False
    assert "pilot" in manifest["paper_table_policy"]


def test_openp5_qwen_lora_controlled_contract(tmp_path: Path) -> None:
    packet = _openp5_packet(tmp_path)
    config = {
        "project": "openp5",
        "controlled_baseline_name": "openp5_style_qwen3_lora_tiny",
        "base_model": "/models/qwen",
        "packet_dir": str(packet),
        "output_dir": str(tmp_path / "out"),
        "seed": 13,
        "lora": {"r": 16, "alpha": 32},
        "training": {"num_train_epochs": 1},
        "scoring": {"type": "candidate_token_likelihood"},
    }
    manifest = prepare_controlled_baseline(config=config, config_path=tmp_path / "cfg.yaml")
    train = Path(manifest["files"]["train_sft"]).read_text(encoding="utf-8").splitlines()
    first = json.loads(train[0])
    assert "<item_1>" in first["messages"][0]["content"]
    assert first["messages"][1]["content"] == "<item_2>"
    score = json.loads(Path(manifest["files"]["test_score_plan"]).read_text(encoding="utf-8").splitlines()[0])
    assert score["candidate_outputs"] == ["<item_2>", "<item_3>"]
    assert "candidate_scores.csv" in Path(tmp_path / "out" / "server_command_plan.md").read_text(encoding="utf-8")
    assert manifest["implementation_fidelity"] == "controlled_adapter_pilot"


def test_dealrec_qwen_lora_controlled_contract(tmp_path: Path) -> None:
    packet = tmp_path / "dealrec_packet"
    project = packet / "dealrec"
    project.mkdir(parents=True)
    task = {
        "example_id": "e1",
        "user_id": "u1",
        "history_text": "One",
        "target_item_id": "i2",
        "candidate_item_ids": ["i2", "i3"],
        "candidate_texts": [{"item_id": "i2", "text": "Two"}, {"item_id": "i3", "text": "Three"}],
    }
    _write_jsonl(project / "train_project_tasks.jsonl", [task])
    _write_jsonl(project / "valid_project_tasks.jsonl", [task])
    _write_jsonl(project / "test_project_tasks.jsonl", [task])
    (project / "candidate_scores_template.csv").write_text("example_id,user_id,item_id,score\n", encoding="utf-8")
    (project / "item_id_mapping.csv").write_text("item_id,title,category,domain,text\n", encoding="utf-8")
    config = {
        "project": "dealrec",
        "controlled_baseline_name": "dealrec_qwen3_lora_tiny",
        "base_model": "/models/qwen",
        "packet_dir": str(packet),
        "output_dir": str(tmp_path / "out"),
        "seed": 13,
        "lora": {"r": 16, "alpha": 32},
        "training": {"num_train_epochs": 1},
        "scoring": {"type": "candidate_id_likelihood"},
    }
    manifest = prepare_controlled_baseline(config=config, config_path=tmp_path / "cfg.yaml")
    first = json.loads(Path(manifest["files"]["train_sft"]).read_text(encoding="utf-8").splitlines()[0])
    assert "DEALRec controlled recommendation task" in first["messages"][0]["content"]
    assert first["messages"][1]["content"] in {"Yes.", "No."}
    score = json.loads(Path(manifest["files"]["test_score_plan"]).read_text(encoding="utf-8").splitlines()[0])
    assert score["candidate_outputs"] == ["Yes."]
    assert len(Path(manifest["files"]["test_score_plan"]).read_text(encoding="utf-8").splitlines()) == 2


def test_official_native_requires_fidelity_provenance(tmp_path: Path) -> None:
    packet = _openp5_packet(tmp_path)
    config = {
        "project": "openp5",
        "controlled_baseline_name": "openp5_official_native_tiny",
        "base_model": "/models/qwen",
        "packet_dir": str(packet),
        "output_dir": str(tmp_path / "out"),
        "implementation_fidelity": "official_native_controlled",
        "official_repo": "https://github.com/agiresearch/OpenP5",
        "provenance": {"official_algorithm_reused": False},
    }
    with pytest.raises(SystemExit):
        prepare_controlled_baseline(config=config, config_path=tmp_path / "cfg.yaml")
