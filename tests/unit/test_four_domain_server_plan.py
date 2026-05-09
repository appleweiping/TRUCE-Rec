from scripts.plan_four_domain_server_runs import build_commands


def test_four_domain_plan_preserves_week8_paths() -> None:
    commands = build_commands(
        source_root="SRC",
        output_root="OUT",
        domains=["beauty", "books"],
        splits=["valid", "test"],
    )
    assert len(commands) == 4
    assert "SRC/beauty_supplementary_smallerN_100neg_valid_same_candidate" in commands[0]
    assert "--output-dir OUT/beauty_supplementary_smallerN_100neg/valid" in commands[0]
    assert "--strict-target-in-candidates" in commands[0]
    assert "SRC/books_large10000_100neg_test_same_candidate" in commands[-1]
    assert "--domain books" in commands[-1]


def test_four_domain_plan_can_emit_ours_adapter_prep() -> None:
    commands = build_commands(
        source_root="SRC",
        output_root="OUT",
        domains=["beauty"],
        splits=["test"],
        include_ours_adapter_prep=True,
    )
    assert any("prepare_ours_qwen_adapter_training.py" in command for command in commands)
    assert any("--processed-root OUT/beauty_supplementary_smallerN_100neg" in command for command in commands)


def test_four_domain_plan_accepts_task_slug_overrides() -> None:
    commands = build_commands(
        source_root="SRC",
        output_root="OUT",
        domains=["beauty"],
        task_slugs={"beauty": "beauty_custom_100neg"},
        splits=["test"],
    )
    assert "SRC/beauty_custom_100neg_test_same_candidate" in commands[0]
    assert "--output-dir OUT/beauty_custom_100neg/test" in commands[0]
