"""Output defaults belong to the caller's workspace, never the installation."""
from pathlib import Path

import pytest

from network_builder import cli


@pytest.mark.parametrize('selected', [None, 'chosen output'])
def test_output_is_resolved_from_invocation_directory(monkeypatch, tmp_path, selected):
    # Import precedes chdir: avoid freezing the caller directory at import time.
    monkeypatch.chdir(tmp_path)
    seen = []

    class StopBeforeBuild(Exception):
        pass

    def capture(**kwargs):
        seen.append(Path(kwargs['out_dir']).resolve())
        raise StopBeforeBuild

    monkeypatch.setattr(cli, 'build_dataset', capture)
    with pytest.raises(StopBeforeBuild):
        cli.main([] if selected is None else ['--out-dir', selected])
    assert seen == [tmp_path / (selected or 'network_inputs')]
    assert not list(tmp_path.iterdir())


def test_default_and_explicit_destinations_produce_same_dataset(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert cli.main([]) == 0
    assert cli.main(['--out-dir', str(tmp_path / 'explicit')]) == 0
    default = tmp_path / 'network_inputs'
    explicit = tmp_path / 'explicit'
    assert (default / cli.DEFAULT_YAML_NAME).read_bytes() == (explicit / cli.DEFAULT_YAML_NAME).read_bytes()
    assert (default / 'MANIFEST.json').is_file()
