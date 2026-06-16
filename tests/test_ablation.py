from devjepa.ablation import _apply_overrides


def test_apply_overrides_accepts_nested_and_dotted_values() -> None:
    config = {
        "data": {"encoded_path": "old.npz", "tasks": ["reach-v3"]},
        "loss": {"decision_weight": 0.0},
    }

    _apply_overrides(
        config,
        {
            "data": {"encoded_path": "/root/data/new.npz"},
            "loss.decision_weight": 0.01,
        },
    )

    assert config["data"] == {
        "encoded_path": "/root/data/new.npz",
        "tasks": ["reach-v3"],
    }
    assert config["loss"]["decision_weight"] == 0.01
