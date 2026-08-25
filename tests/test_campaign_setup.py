from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
PREPARE_MONOFONIC = (
    ROOT
    / "validation"
    / "monofonic-agreement"
    / "scripts"
    / "prepare-monofonic-source.py"
)


@pytest.mark.parametrize(
    ("source_text", "expected_text"),
    [
        (
            """\
void grid::read() {
    this->reset();
    this->n_ = input_dims;
    this->space_ = rspace_id;
    this->allocate();
}
""",
            None,
        ),
        (
            """\
void grid::read() {
    for (size_t i = 0; i < 3; ++i)
        this->n_[i] = dimsize[i];
    this->space_ = rspace_id;

    this->reset();
    this->allocate();
}
""",
            """\
void grid::read() {
    this->reset();
    for (size_t i = 0; i < 3; ++i)
        this->n_[i] = dimsize[i];
    this->space_ = rspace_id;

    this->allocate();
}
""",
        ),
    ],
    ids=("current-upstream-layout", "known-old-layout"),
)
def test_prepare_monofonic_source(tmp_path, source_text, expected_text):
    source = tmp_path / "grid_fft.cc"
    source.write_text(source_text)

    completed = subprocess.run(
        ["python3", str(PREPARE_MONOFONIC), str(source)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert source.read_text() == (source_text if expected_text is None else expected_text)


def test_spherical_evolution_defaults_to_generated_glass():
    config = ROOT / "validation" / "spherical-evolution" / "config.env"
    completed = subprocess.run(
        [
            "env",
            "-u",
            "GLASS_INPUT_MODE",
            "-u",
            "GLASS_SNAP",
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "$GLASS_INPUT_MODE"',
            "bash",
            str(config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == "generate\n"
