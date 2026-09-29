from sft.status import _duration


def test_sft_duration_format() -> None:
    assert _duration(7322) == "02:02:02"
