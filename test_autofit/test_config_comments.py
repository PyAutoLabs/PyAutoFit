"""The comment audit distinguishes YAML values from explanatory comments."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_config_comments.py"
spec = importlib.util.spec_from_file_location("check_config_comments", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_yaml_comments_and_terminal_values(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(
        'quoted: "# literal"\n'
        "single: '# literal' # Explanation.\n"
        "empty_comment: true #   \n"
        "# Nullable setting.\n"
        "nullable:\n"
        "# Sequence setting.\n"
        "sequence: [one, two]\n"
        "block: |\n"
        "  # This is scalar content, not a comment.\n"
        "  fake_key: true\n"
        "# Separated prose does not meet the adjacent-line rule.\n"
        "\n"
        "nested:\n"
        "  missing: false\n"
    )
    assert [(line, key) for _, line, key in audit.check_roots([tmp_path])] == [
        (1, "quoted"),
        (3, "empty_comment"),
        (8, "block"),
        (14, "nested.missing"),
    ]


def test_priors_excluded_and_yml_included(tmp_path):
    priors = tmp_path / "priors"
    priors.mkdir()
    (priors / "model.yaml").write_text("uncommented: true\n")
    (tmp_path / "settings.yml").write_text("flag: true # Enable feature.\n")
    assert audit.check_roots([tmp_path]) == []


def test_missing_and_empty_roots_fail(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        audit.check_roots([tmp_path / "missing"])
    with pytest.raises(ValueError, match="No non-prior YAML"):
        audit.check_roots([tmp_path])


def test_cli_exit_status_and_packaged_defaults(tmp_path, capsys):
    assert audit.main([]) == 0
    (tmp_path / "settings.yaml").write_text("flag: true\n")
    assert audit.main([str(tmp_path)]) == 1
    assert "settings.yaml:1: flag" in capsys.readouterr().out
