from pathlib import Path

from demo_app import get_available_checkpoints, infer_model_type_from_checkpoint


def test_available_checkpoints_prefers_experiment_order(tmp_path: Path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    for filename in [
        "best_model_iou_heavy.pth",
        "best_model_custom_run.pth",
        "best_model_unet_small.pth",
        "best_model_improved.pth",
        "best_model_baseline.pth",
    ]:
        (checkpoints_dir / filename).touch()

    checkpoints = get_available_checkpoints(checkpoints_dir)

    assert [path.name for path in checkpoints] == [
        "best_model_improved.pth",
        "best_model_baseline.pth",
        "best_model_iou_heavy.pth",
        "best_model_custom_run.pth",
    ]


def test_available_checkpoints_uses_generic_alias_when_it_is_the_only_checkpoint(
    tmp_path: Path,
):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "best_model.pth").touch()

    checkpoints = get_available_checkpoints(checkpoints_dir)

    assert [path.name for path in checkpoints] == ["best_model.pth"]


def test_infer_model_type_from_checkpoint_name():
    assert infer_model_type_from_checkpoint(Path("best_model_improved.pth")) == "unet_small"
    assert infer_model_type_from_checkpoint(Path("best_model_baseline.pth")) == "baseline"
    assert infer_model_type_from_checkpoint(Path("best_model_baseline_no_bn.pth")) == "baseline_no_bn"
    assert infer_model_type_from_checkpoint(Path("best_model_iou_heavy.pth")) == "baseline"
