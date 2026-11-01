"""Validate skill YAML, shared pointers and local Markdown file targets (not anchors)."""
from pathlib import Path
import re
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[3]


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys rather than silently keeping the last value."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key, value in self.construct_pairs(node, deep=deep):
            try:
                duplicate = key in result
            except TypeError as error:
                raise yaml.constructor.ConstructorError(
                    None, None, "unhashable YAML mapping key", node.start_mark
                ) from error
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate YAML key: {key}", node.start_mark
                )
            result[key] = value
        return result


def main():
    errors = []
    canonical = ROOT / ".agents/skills"
    pointer = ROOT / ".claude/skills"
    if not pointer.is_symlink() or pointer.resolve() != canonical.resolve():
        errors.append(".claude/skills must link to .agents/skills")
    folders = sorted(p for p in canonical.iterdir() if p.is_dir()) if canonical.is_dir() else []
    skills = []
    if not folders:
        errors.append(".agents/skills must contain at least one skill folder")
    for folder in folders:
        path = folder / "SKILL.md"
        if not path.is_file():
            errors.append(f"{folder.relative_to(ROOT)}: missing SKILL.md")
            continue
        skills.append(path)
        match = re.match(r"\A---[ \t]*\r?\n(.*?)^---[ \t]*(?:\r?\n|\Z)", path.read_text(), re.S | re.M)
        if not match:
            errors.append(f"{path.relative_to(ROOT)}: missing YAML frontmatter")
            continue
        try:
            fields = yaml.load(match[1], Loader=UniqueKeyLoader)
        except yaml.YAMLError as error:
            errors.append(f"{path.relative_to(ROOT)}: invalid YAML: {error}")
            continue
        if not isinstance(fields, dict):
            errors.append(f"{path.relative_to(ROOT)}: frontmatter must be a mapping")
            continue
        name, description = fields.get("name"), fields.get("description")
        if (not isinstance(name, str) or name != path.parent.name
                or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
                or len(name) > 64 or not isinstance(description, str) or not description.strip()):
            errors.append(f"{path.relative_to(ROOT)}: invalid name/description")
    docs = [ROOT / "AGENTS.md", ROOT / "CLAUDE.md"]
    docs += sorted((ROOT / "docs").glob("*.md"))
    docs += skills
    for path in docs:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            target = target.split("#", 1)[0]
            if not target or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            if not (path.parent / unquote(target)).exists():
                errors.append(f"{path.relative_to(ROOT)}: missing link target {target}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"shared pointer, {len(skills)} skills and {len(docs)} document file-target checks passed")


if __name__ == "__main__":
    main()
