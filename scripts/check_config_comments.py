"""Report config leaves lacking a comment on their line or immediately above.

Run with one or more config directories; priors/ is excluded. Without arguments,
check this checkout's packaged autofit/config directory. Only YAML is loaded:
the scientific packages and their configuration stack are never imported.
"""

import argparse
from pathlib import Path

import yaml


def comment_lines(text):
    """Find nonempty comments, masking YAML scalar tokens (including quoted #)."""
    masked = list(text)
    for token in yaml.scan(text):
        if isinstance(token, yaml.tokens.ScalarToken):
            for index in range(token.start_mark.index, token.end_mark.index):
                if masked[index] != "\n":
                    masked[index] = " "
    return {
        index
        for index, line in enumerate("".join(masked).splitlines())
        if "#" in line and line.split("#", 1)[1].strip()
    }


def uncommented_leaves(path):
    text = path.read_text()
    comments = comment_lines(text)
    lines = text.splitlines()
    preceding_comments = {
        line for line in comments if lines[line].lstrip().startswith("#")
    }
    findings = []

    def walk(node, keys=()):
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                path_keys = keys + (key.value,)
                if isinstance(value, yaml.MappingNode):
                    walk(value, path_keys)
                else:
                    line = key.start_mark.line
                    if line not in comments and line - 1 not in preceding_comments:
                        findings.append((path, line + 1, ".".join(path_keys)))

    for document in yaml.compose_all(text):
        walk(document)
    return findings


def check_roots(roots):
    findings = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"Config directory does not exist: {root}")
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in {".yaml", ".yml"}
            and "priors" not in path.relative_to(root).parts
        )
        if not files:
            raise ValueError(f"No non-prior YAML files found: {root}")
        for path in files:
            findings.extend(uncommented_leaves(path))
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", type=Path)
    args = parser.parse_args(argv)
    roots = args.roots or [Path(__file__).resolve().parents[1] / "autofit/config"]
    try:
        findings = check_roots(roots)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    for path, line, key in findings:
        print(f"{path}:{line}: {key}")
    print(f"Uncommented config leaves: {len(findings)}")
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
