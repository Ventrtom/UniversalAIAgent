import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import cli.audio_utils as audio_utils


def test_cancel_recording():
    audio_update, status_update = audio_utils.cancel_recording()
    assert audio_update["visible"] is False
    assert audio_update["value"] is None
    assert status_update["visible"] is False
