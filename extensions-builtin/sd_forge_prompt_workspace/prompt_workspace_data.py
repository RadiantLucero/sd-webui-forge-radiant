import json
import os
import re
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import ParamSpec, TypeVar

FORMAT_VERSION = 1
ENTRY_KINDS = {"snippet", "prompt"}
MAX_SECTIONS = 10
DEFAULT_SECTION_NAMES = ("Subject", "Action", "Environment", "Framing", "Lighting", "Details", "Style")
SECTION_PATTERN = re.compile(r"<<([^<>]*)>>")
# A single pair of angle brackets without a colon, so LoRA syntax such as <lora:name:1> is not matched.
REFERENCE_PATTERN = re.compile(r"(?<!<)<([\w-]+)>(?!>)")
REFERENCE_NAME_PATTERN = re.compile(r"[\w-]+")
_write_lock = RLock()
P = ParamSpec("P")
T = TypeVar("T")


class PromptResolutionError(ValueError):
    pass


def _lock_writes(function: Callable[P, T]) -> Callable[P, T]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        with _write_lock:
            return function(*args, **kwargs)

    return wrapped


def library_path(extension_dir: Path, configured_directory: str) -> Path:
    directory = Path(configured_directory.strip()).expanduser() if configured_directory.strip() else extension_dir / "data"
    return directory.resolve() / "library.json"


def normalize_prompt_text(text: str) -> str:
    return re.sub(r"\s+([,.;:])", r"\1", " ".join(text.split()))


def _punctuation_boundary(left: str, right: str) -> tuple[str, str]:
    left_end = len(left.rstrip())
    right_start = len(right) - len(right.lstrip())
    if not left_end or right_start == len(right):
        return left, right

    previous, following = left[left_end - 1], right[right_start]
    if following in ",;" and previous in ".,;:!?":
        right = right[:right_start] + right[right_start + 1 :]
    elif following == "." and previous == ".":
        # Preserve a deliberate ellipsis when another period meets it.
        left_dots = len(left[:left_end]) - len(left[:left_end].rstrip("."))
        right_dots = len(right[right_start:]) - len(right[right_start:].lstrip("."))
        right = right[:right_start] + right[right_start + min(left_dots, right_dots) :]
    elif following == "." and previous in ",;":
        left = left[: left_end - 1] + "." + left[left_end:]
        right = right[:right_start] + right[right_start + 1 :]
    elif following == "." and previous in "!?" and right[right_start + 1 : right_start + 2] != ".":
        right = right[:right_start] + right[right_start + 1 :]
    return left, right


def join_prompt_parts(left: str, right: str, separator: str) -> str:
    left, right = left.rstrip(), right.lstrip()
    if not left or not right:
        return left or right
    left, right = _punctuation_boundary(left, right)
    if not right:
        return left
    if right[0].isspace() or right[0] in ",.;:!?":
        return left + right
    return left + (" " if left[-1] in ",.;:!?" else separator) + right


def substitute_prompt(pattern: re.Pattern[str], text: str, replace: Callable[[re.Match[str]], str]) -> str:
    result = ""
    end = 0
    empty_at_start = False
    for match in pattern.finditer(text):
        literal = text[end : match.start()]
        if empty_at_start:
            # Remove an orphan separator, but not a decimal such as .5.
            literal = re.sub(r"^(\s*)[.,;](?=\s|$)", r"\1", literal, count=1)
        left, right = _punctuation_boundary(result, literal)
        result = left + right
        expansion = replace(match)
        left, right = _punctuation_boundary(result, expansion)
        result = left + right
        empty_at_start = not result.strip() and not expansion.strip()
        end = match.end()

    literal = text[end:]
    if empty_at_start:
        literal = re.sub(r"^(\s*)[.,;](?=\s|$)", r"\1", literal, count=1)
    left, right = _punctuation_boundary(result, literal)
    return left + right


@dataclass(frozen=True)
class PromptEntry:
    id: str
    kind: str
    name: str
    category: str
    tags: list[str]
    prompt: str
    negative_prompt: str
    variant_group: str | None = None
    variant_name: str = ""
    reference: str = ""
    section: str = ""
    template: str | None = None
    sections: list[dict[str, str]] | None = None

    def source_text(self) -> str:
        return self.prompt if self.template is None else self.template

    def mentioned_text(self) -> str:
        return " ".join([self.source_text(), *(section["text"] for section in self.sections or []), self.negative_prompt])


def reference_map(entries: list[PromptEntry]) -> dict[str, str]:
    return {entry.reference.casefold(): entry.prompt for entry in entries if entry.kind == "snippet" and entry.reference}


def resolve_references(text: str, references: dict[str, str], trail: tuple[str, ...] = (), allow_unknown: bool = False) -> str:
    def replace(match):
        name = match.group(1)
        key = name.casefold()
        if key in trail:
            raise PromptResolutionError(f"Circular reference: {' -> '.join(f'<{item}>' for item in (*trail, key))}")
        if key not in references:
            if allow_unknown:
                return match.group(0)
            raise PromptResolutionError(f"Unknown reference <{name}>")
        return resolve_references(references[key], references, (*trail, key), allow_unknown)

    return substitute_prompt(REFERENCE_PATTERN, text, replace)


def compose_prompt(template: str, sections: list[dict[str, str]], references: dict[str, str]) -> str:
    texts = {section["name"].casefold(): section["text"] for section in sections}

    def replace(match):
        name = match.group(1).strip()
        if name.casefold() not in texts:
            raise PromptResolutionError(f"Unknown section <<{name}>>")
        return f" {texts[name.casefold()]} "

    return normalize_prompt_text(resolve_references(substitute_prompt(SECTION_PATTERN, template, replace), references))


def used_section_names(template: str, section_names: list[str]) -> list[str]:
    used = {match.group(1).strip().casefold() for match in SECTION_PATTERN.finditer(template)}
    return [name for name in section_names if name.casefold() in used]


def validate_section_names(names: list[str]) -> list[str]:
    cleaned = [name.strip() for name in names if name.strip()]
    if len(cleaned) > MAX_SECTIONS:
        raise ValueError(f"At most {MAX_SECTIONS} sections are supported")
    if any("<" in name or ">" in name for name in cleaned):
        raise ValueError("Section names cannot contain < or >")
    if len({name.casefold() for name in cleaned}) != len(cleaned):
        raise ValueError("Section names must be unique")
    return cleaned


class PromptWorkspaceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[PromptEntry]:
        return self._read()[0]

    def section_names(self) -> list[str]:
        return self._read()[1]

    @_lock_writes
    def save_section_names(self, names: list[str]) -> list[str]:
        entries, _ = self._read()
        cleaned = validate_section_names(names)
        if not cleaned:
            raise ValueError("Define at least one section")
        self._write(entries, cleaned)
        return cleaned

    def preview(self, kind: str, text: str, sections: list[dict[str, str]] | None = None) -> str:
        references = reference_map(self.load())
        if kind == "prompt":
            return compose_prompt(text, sections or [], references)
        return normalize_prompt_text(resolve_references(text, references))

    def select_entry(self, entry_id: str, *, as_reference: bool = False) -> tuple[PromptEntry, str, str]:
        entries = self.load()
        entry = next((item for item in entries if item.id == entry_id), None)
        if entry is None:
            raise ValueError("The selected prompt entry no longer exists")
        if as_reference and entry.kind == "snippet" and entry.reference:
            prompt = f"<{entry.reference}>" if entry.prompt else ""
            return entry, prompt, entry.negative_prompt
        references = reference_map(entries)
        if entry.kind == "prompt":
            prompt = compose_prompt(entry.source_text(), entry.sections or [], references)
        else:
            prompt = normalize_prompt_text(resolve_references(entry.prompt, references))
        negative = normalize_prompt_text(resolve_references(entry.negative_prompt, references))
        return entry, prompt, negative

    def resolve_entry(self, entry_id: str) -> tuple[str, str, str]:
        entry, prompt, negative = self.select_entry(entry_id)
        return entry.kind, prompt, negative

    @_lock_writes
    def save_entry(
        self,
        entry_id: str | None,
        *,
        kind: str,
        name: str,
        category: str,
        tags: list[str],
        prompt: str,
        negative_prompt: str,
        variant_name: str = "",
        reference: str = "",
        section: str = "",
        sections: list[dict[str, str]] | None = None,
    ) -> PromptEntry:
        entries, section_names = self._read()
        existing = next((entry for entry in entries if entry.id == entry_id), None)
        if entry_id is not None and existing is None:
            raise ValueError("The selected prompt entry no longer exists")
        return self._save_entry(
            entries, section_names, existing, kind=kind, name=name, category=category, tags=tags,
            prompt=prompt, negative_prompt=negative_prompt, variant_name=variant_name,
            reference=reference, section=section, sections=sections,
        )

    @_lock_writes
    def upsert_entry(
        self,
        *,
        kind: str,
        name: str,
        category: str,
        tags: list[str],
        prompt: str,
        negative_prompt: str,
        variant_name: str = "",
        reference: str = "",
        section: str = "",
        sections: list[dict[str, str]] | None = None,
        new_variant: bool = False,
        selected_entry_id: str | None = None,
    ) -> PromptEntry:
        entries, section_names = self._read()
        if kind not in ENTRY_KINDS:
            raise ValueError(f"Unsupported prompt entry kind: {kind}")
        name = self._required_text(name, "Name")
        category = self._required_text(category, "Category")
        selected = next((entry for entry in entries if entry.id == selected_entry_id), None)
        if selected_entry_id and selected is None:
            raise ValueError("The selected prompt entry no longer exists")
        if selected and not new_variant and selected.kind == kind and selected.name == name and selected.category != category:
            if any(entry.name == name and entry.category == category for entry in entries):
                raise ValueError(f"The name '{name}' is already used in '{category}'; cannot move the selected entry")
            if kind == "prompt":
                moving = [entry for entry in entries if entry.kind == "prompt" and entry.variant_group == selected.variant_group]
                for entry in moving:
                    self._ensure_unique(entries, entry)
            else:
                moving = [selected]
                if sum(entry.name == name and entry.category == selected.category for entry in entries) > 1:
                    raise ValueError(f"Multiple entries named '{name}' exist in '{selected.category}'; resolve them before moving")
            moving_ids = {entry.id for entry in moving}
            if sum(entry.id in moving_ids for entry in entries) != len(moving_ids):
                raise ValueError("Multiple entries share an ID; resolve them before moving")
            entries = [replace(entry, category=category) if entry.id in moving_ids else entry for entry in entries]
        matches = [entry for entry in entries if entry.name == name and entry.category == category]
        if any(entry.kind != kind for entry in matches):
            raise ValueError(f"The name '{name}' is already used by another entry type in '{category}'")

        existing = None
        variant_group = None
        if kind == "snippet":
            if new_variant:
                raise ValueError("Only a prompt can have variants")
            if len(matches) > 1:
                raise ValueError(f"Multiple entries named '{name}' exist in '{category}'; resolve them before saving")
            existing = matches[0] if matches else None
        else:
            variant_name = self._required_text(variant_name or "Default", "Variant name")
            groups = {entry.variant_group for entry in matches}
            if len(groups) > 1:
                raise ValueError(f"Multiple prompts named '{name}' exist in '{category}'; resolve them before saving")
            variant_group = next(iter(groups), None)
            if variant_group and any(
                entry.variant_group == variant_group and (entry.name != name or entry.category != category)
                for entry in entries
            ):
                raise ValueError("This prompt's variants have different names or categories; resolve them before saving")
            variant_matches = [entry for entry in matches if entry.variant_name.casefold() == variant_name.casefold()]
            if len(variant_matches) > 1:
                raise ValueError(f"Multiple variants named '{variant_name}' exist; resolve them before saving")
            existing = variant_matches[0] if variant_matches else None
            if new_variant and not variant_group:
                raise ValueError("Save the prompt before adding a variant")
            if new_variant and existing:
                raise ValueError(f"A variant named '{variant_name}' already exists")

        return self._save_entry(
            entries, section_names, existing, kind=kind, name=name, category=category, tags=tags,
            prompt=prompt, negative_prompt=negative_prompt, variant_name=variant_name,
            reference=reference, section=section, sections=sections, variant_group=variant_group,
        )

    def _save_entry(
        self,
        entries: list[PromptEntry],
        section_names: list[str],
        existing: PromptEntry | None,
        *,
        kind: str,
        name: str,
        category: str,
        tags: list[str],
        prompt: str,
        negative_prompt: str,
        variant_name: str,
        reference: str,
        section: str,
        sections: list[dict[str, str]] | None,
        variant_group: str | None = None,
    ) -> PromptEntry:
        is_prompt = kind == "prompt"
        entry = self._make_entry(
            id=existing.id if existing else str(uuid.uuid4()),
            kind=kind,
            name=name,
            category=category,
            tags=tags,
            prompt=prompt,
            negative_prompt=negative_prompt,
            variant_group=(existing.variant_group if existing and existing.variant_group else variant_group or str(uuid.uuid4())) if is_prompt else None,
            variant_name=(variant_name or "Default") if is_prompt else "",
            reference="" if is_prompt else reference,
            section="" if is_prompt else section,
            template=prompt if is_prompt else None,
            sections=(sections or []) if is_prompt else None,
        )
        if existing and existing.reference and existing.reference.casefold() != entry.reference.casefold():
            self._ensure_reference_unused(entries, existing)
            if any(match.group(1).casefold() == existing.reference.casefold() for match in REFERENCE_PATTERN.finditer(entry.mentioned_text())):
                raise ValueError(f"<{existing.reference}> is still used by: {entry.name}")
        return self._store(entries, section_names, entry, replace_id=existing.id if existing else None)

    @_lock_writes
    def save_variant(
        self,
        entry_id: str,
        *,
        variant_name: str,
        prompt: str,
        negative_prompt: str,
        sections: list[dict[str, str]] | None = None,
    ) -> PromptEntry:
        entries, section_names = self._read()
        source = next((entry for entry in entries if entry.id == entry_id), None)
        if source is None:
            raise ValueError("The selected prompt entry no longer exists")
        if source.kind != "prompt" or not source.variant_group:
            raise ValueError("Only a saved prompt can have variants")

        entry = self._make_entry(
            id=str(uuid.uuid4()),
            kind="prompt",
            name=source.name,
            category=source.category,
            tags=source.tags,
            prompt=prompt,
            negative_prompt=negative_prompt,
            variant_group=source.variant_group,
            variant_name=variant_name,
            template=prompt,
            sections=sections or [],
        )
        return self._store(entries, section_names, entry, replace_id=None)

    @_lock_writes
    def delete_entry(self, entry_id: str) -> bool:
        entries, section_names = self._read()
        entry = next((item for item in entries if item.id == entry_id), None)
        if entry is None:
            return False
        if entry.reference:
            self._ensure_reference_unused(entries, entry)
        self._write([item for item in entries if item.id != entry_id], section_names)
        return True

    def variants(self, entry_id: str) -> list[PromptEntry]:
        entries = self.load()
        source = next((entry for entry in entries if entry.id == entry_id), None)
        if source is None:
            raise ValueError("The selected prompt entry no longer exists")
        if source.kind != "prompt" or not source.variant_group:
            return []
        return sorted(
            (entry for entry in entries if entry.variant_group == source.variant_group),
            key=lambda entry: (entry.variant_name.casefold(), entry.id),
        )

    def _store(self, entries: list[PromptEntry], section_names: list[str], entry: PromptEntry, replace_id: str | None) -> PromptEntry:
        self._ensure_unique(entries, entry)
        others = [item for item in entries if item.id != replace_id]
        references = reference_map([*others, entry])
        if entry.kind == "prompt":
            composed = compose_prompt(entry.template, entry.sections, references)
        else:
            if SECTION_PATTERN.search(entry.prompt):
                raise ValueError("Section placeholders can only be used in prompts")
            # Snippets may reference parts that are defined later; cycles are still rejected.
            composed = normalize_prompt_text(resolve_references(entry.prompt, references, allow_unknown=True))
        resolve_references(entry.negative_prompt, references, allow_unknown=entry.kind == "snippet")
        if not composed and not entry.negative_prompt:
            raise ValueError("Enter a prompt or a negative prompt before saving")
        if entry.kind == "prompt":
            entry = PromptEntry(**{**asdict(entry), "prompt": composed})

        if replace_id:
            entries = [entry if item.id == replace_id else item for item in entries]
        else:
            entries = [*entries, entry]
        self._write(entries, section_names)
        return entry

    def _read(self) -> tuple[list[PromptEntry], list[str]]:
        if not self.path.exists():
            return [], list(DEFAULT_SECTION_NAMES)

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid prompt library JSON in {self.path}: {error}") from error

        if not isinstance(data, dict) or data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported prompt library format in {self.path}")

        raw_entries = data.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError(f"Prompt library entries must be a list in {self.path}")

        names = data.get("section_names", list(DEFAULT_SECTION_NAMES))
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"Section names must be a list of text in {self.path}")
        return [self._parse_entry(raw, index) for index, raw in enumerate(raw_entries)], validate_section_names(names) or list(DEFAULT_SECTION_NAMES)

    @classmethod
    def _parse_entry(cls, raw, index: int) -> PromptEntry:
        if not isinstance(raw, dict):
            raise ValueError(f"Prompt library entry {index} must be an object in the library file")

        try:
            entry = cls._make_entry(**raw)
        except TypeError as error:
            raise ValueError(f"Invalid fields in prompt library entry {index}: {error}") from error
        return entry

    @classmethod
    def _make_entry(cls, **values) -> PromptEntry:
        entry = PromptEntry(**values)
        entry_id = cls._required_text(entry.id, "Entry ID")
        if not isinstance(entry.kind, str) or entry.kind not in ENTRY_KINDS:
            raise ValueError(f"Unsupported prompt entry kind: {entry.kind}")
        if not isinstance(entry.tags, list) or not all(isinstance(tag, str) for tag in entry.tags):
            raise ValueError("Tags must be a list of strings")
        if not isinstance(entry.prompt, str) or not isinstance(entry.negative_prompt, str):
            raise ValueError("Prompt and negative prompt must be text")
        if not isinstance(entry.reference, str) or not isinstance(entry.section, str):
            raise ValueError("Reference and default section must be text")
        name = cls._required_text(entry.name, "Name")
        category = cls._required_text(entry.category, "Category")
        tags = [tag.strip() for tag in entry.tags if tag.strip()]
        reference = entry.reference.strip()
        variant_group = None
        variant_name = ""
        template = None
        sections = None
        if entry.kind == "prompt":
            variant_group = cls._required_text(entry.variant_group, "Variant group")
            variant_name = cls._required_text(entry.variant_name, "Variant name")
            if reference or entry.section:
                raise ValueError("Only text snippets can have a reference name or default section")
            if entry.template is not None:
                if not isinstance(entry.template, str):
                    raise ValueError("Prompt template must be text")
                template = entry.template.strip()
            sections = cls._normalize_sections(entry.sections)
        else:
            if entry.variant_group is not None or entry.variant_name:
                raise ValueError("Text snippets cannot have prompt variants")
            if entry.template is not None or entry.sections is not None:
                raise ValueError("Text snippets cannot have a template or sections")
            if reference and not REFERENCE_NAME_PATTERN.fullmatch(reference):
                raise ValueError("Reference names may only contain letters, digits, '_' and '-'")
        return PromptEntry(
            id=entry_id,
            kind=entry.kind,
            name=name,
            category=category,
            tags=tags,
            prompt=normalize_prompt_text(entry.prompt),
            negative_prompt=normalize_prompt_text(entry.negative_prompt),
            variant_group=variant_group,
            variant_name=variant_name,
            reference=reference,
            section=entry.section.strip(),
            template=template,
            sections=sections,
        )

    @staticmethod
    def _normalize_sections(sections) -> list[dict[str, str]] | None:
        if sections is None:
            return None
        if not isinstance(sections, list) or not all(
            isinstance(section, dict) and set(section) == {"name", "text"} and all(isinstance(value, str) for value in section.values())
            for section in sections
        ):
            raise ValueError("Sections must be a list of objects with a name and text")
        names = validate_section_names([section["name"] for section in sections])
        if len(names) != len(sections):
            raise ValueError("Section names cannot be empty")
        return [{"name": section["name"].strip(), "text": section["text"].strip()} for section in sections]

    @staticmethod
    def _required_text(value, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} cannot be empty")
        return value.strip()

    @staticmethod
    def _ensure_unique(entries: list[PromptEntry], candidate: PromptEntry) -> None:
        for entry in entries:
            if entry.id == candidate.id:
                continue
            same_name = entry.name == candidate.name and entry.category == candidate.category
            if (
                candidate.kind == "prompt"
                and entry.kind == "prompt"
                and entry.variant_group == candidate.variant_group
            ):
                if not same_name:
                    raise ValueError("This prompt's variants have different names or categories; resolve them before saving")
                if entry.variant_name.casefold() == candidate.variant_name.casefold():
                    raise ValueError(f"A variant named '{candidate.variant_name}' already exists")
            if same_name:
                if entry.kind != candidate.kind:
                    raise ValueError(f"The name '{candidate.name}' is already used by another entry type in '{candidate.category}'")
                if candidate.kind == "snippet" or entry.variant_group != candidate.variant_group:
                    raise ValueError(f"The name '{candidate.name}' is already used in '{candidate.category}'")
            if candidate.reference and entry.reference.casefold() == candidate.reference.casefold():
                raise ValueError(f"The reference name <{candidate.reference}> is already used by '{entry.name}'")

    @staticmethod
    def _ensure_reference_unused(entries: list[PromptEntry], entry: PromptEntry) -> None:
        key = entry.reference.casefold()
        users = [
            item.name
            for item in entries
            if item.id != entry.id and any(match.group(1).casefold() == key for match in REFERENCE_PATTERN.finditer(item.mentioned_text()))
        ]
        if users:
            raise ValueError(f"<{entry.reference}> is still used by: {', '.join(sorted(set(users)))}")

    def _write(self, entries: list[PromptEntry], section_names: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            file_descriptor, temp_name = tempfile.mkstemp(prefix=".prompt-library-", suffix=".tmp", dir=self.path.parent)
            temp_path = Path(temp_name)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "format_version": FORMAT_VERSION,
                        "section_names": section_names,
                        "entries": [asdict(entry) for entry in entries],
                    },
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, self.path)
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
