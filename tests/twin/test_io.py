from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def test_twin_stable_read_matches_product(tmp_path):
    from fluidmix.core.contracts._common import stable_read_path as product_read
    from twin.io import stable_read_path as twin_read

    source = tmp_path / "source.bin"
    source.write_bytes(b"bounded source")
    assert twin_read(source, "source") == product_read(source, "source")


@pytest.mark.parametrize("reader_name", ["product", "twin"])
def test_twin_stable_read_preserves_bound_and_symlink_rejection(tmp_path, reader_name):
    from fluidmix.core.contracts._common import stable_read_path as product_read
    from twin.io import stable_read_path as twin_read

    reader = {"product": product_read, "twin": twin_read}[reader_name]
    source = tmp_path / "source.bin"
    source.write_bytes(b"too large")
    with pytest.raises(ValueError, match="bounded regular non-symlink file"):
        reader(source, "source", max_bytes=2)
    link = tmp_path / "link.bin"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="unavailable or unsafe"):
        reader(link, "source")
