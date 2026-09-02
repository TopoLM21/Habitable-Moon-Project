from PIL import Image
import pytest

from analysis.assemble_saved_gifs import assemble


def test_intermediate_gif_filters_future_frames_and_preserves_inputs(tmp_path):
    source = tmp_path / "run"
    frames = source / "hydrosphere_frames"
    frames.mkdir(parents=True)
    for time, color in [(0, "red"), (20, "green"), (40, "blue")]:
        Image.new("RGB", (8, 8), color).save(frames / f"surface_{time:06}.0_Myr.png")
    before = {path.name: path.read_bytes() for path in frames.iterdir()}
    output = tmp_path / "gif"
    files = assemble(source, output, 20.0, ["surface"])
    assert len(files) == 1
    with Image.open(files[0]) as gif:
        assert gif.n_frames == 2
        gif.seek(1)
        assert gif.convert("RGB").getpixel((0, 0)) == (0, 128, 0)
    assert {path.name: path.read_bytes() for path in frames.iterdir()} == before
    with pytest.raises(FileExistsError):
        assemble(source, output, 20.0, ["surface"])
    with pytest.raises(ValueError, match="outside"):
        assemble(source, source / "new_gif", 20.0, ["surface"])
