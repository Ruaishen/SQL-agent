import json

from sft.train_config import SftTrainConfig
from sft.train_status import read_status


def test_sft_train_status(tmp_path) -> None:
    model = tmp_path / "model"
    data = tmp_path / "data"
    output = tmp_path / "output"
    model.mkdir()
    data.mkdir()
    output.mkdir()
    config = SftTrainConfig(model, data, output)
    metric = {
        "step": 1,
        "total_steps": 10,
        "progress_percent": 10.0,
        "epoch": 1,
        "elapsed_seconds": 5,
        "eta_seconds": 45,
        "estimated_finish_utc": "soon",
        "loss": 1.25,
        "grad_norm_before_clip": 2.5,
        "micro_batch_size": 8,
    }
    (output / "metrics.jsonl").write_text(json.dumps(metric) + "\n")
    status = read_status(config)
    assert "step 1/10" in status
    assert "loss 1.2500" in status
