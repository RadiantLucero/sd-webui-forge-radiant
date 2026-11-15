import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

FORMAT_VERSION = 1
ENTRY_KINDS = {"snippet", "prompt"}


def library_path(extension_dir: Path, configured_directory: str) -> Path:
    directory = Path(configured_directory.strip()).expanduser() if configured_directory.strip() else extension_dir / "data"
    return directory.resolve() / "library.json"


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


class PromptWorkspaceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[PromptEntry]:
        if not self.path.exists():
            return []

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid prompt library JSON in {self.path}: {error}") from error

        if not isinstance(data, dict) or data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported prompt library format in {self.path}")

        raw_entries = data.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError(f"Prompt library entries must be a list in {self.path}")

        return [self._parse_entry(raw, index) for index, raw in enumerate(raw_entries)]

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
    ) -> PromptEntry:
        entries = self.load()
        existing = next((entry for entry in entries if entry.id == entry_id), None)
        if entry_id is not None and existing is None:
            raise ValueError("The selected prompt entry no longer exists")

        group = None
        variant = ""
        if kind == "prompt":
            group = existing.variant_group if existing else str(uuid.uuid4())
            variant = self._required_text(variant_name or "Default", "Variant name")

        entry = self._make_entry(
            id=existing.id if existing else str(uuid.uuid4()),
            kind=kind,
            name=name,
            category=category,
            tags=tags,
            prompt=prompt,
            negative_prompt=negative_prompt,
            variant_group=group,
            variant_name=variant,
        )
        self._ensure_variant_name_available(entries, entry)

        if existing:
            entries = [entry if item.id == existing.id else item for item in entries]
        else:
            entries.append(entry)

        self._write(entries)
        return entry

    def save_variant(
        self,
        entry_id: str,
        *,
        variant_name: str,
        prompt: str,
        negative_prompt: str,
    ) -> PromptEntry:
        entries = self.load()
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
        )
        self._ensure_variant_name_available(entries, entry)
        entries.append(entry)
        self._write(entries)
        return entry

    def delete_entry(self, entry_id: str) -> bool:
        entries = self.load()
        remaining = [entry for entry in entries if entry.id != entry_id]
        if len(remaining) == len(entries):
            return False
        self._write(remaining)
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
        if not entry.prompt.strip() and not entry.negative_prompt.strip():
            raise ValueError("Enter a prompt or a negative prompt before saving")
        name = cls._required_text(entry.name, "Name")
        category = cls._required_text(entry.category, "Category")
        tags = [tag.strip() for tag in entry.tags if tag.strip()]
        variant_group = None
        variant_name = ""
        if entry.kind == "prompt":
            variant_group = cls._required_text(entry.variant_group, "Variant group")
            variant_name = cls._required_text(entry.variant_name, "Variant name")
        elif entry.variant_group is not None or entry.variant_name:
            raise ValueError("Text snippets cannot have prompt variants")
        return PromptEntry(
            id=entry_id,
            kind=entry.kind,
            name=name,
            category=category,
            tags=tags,
            prompt=entry.prompt,
            negative_prompt=entry.negative_prompt,
            variant_group=variant_group,
            variant_name=variant_name,
        )

    @staticmethod
    def _required_text(value, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} cannot be empty")
        return value.strip()

    @staticmethod
    def _ensure_variant_name_available(entries: list[PromptEntry], candidate: PromptEntry) -> None:
        if candidate.kind != "prompt":
            return
        for entry in entries:
            if (
                entry.id != candidate.id
                and entry.variant_group == candidate.variant_group
                and entry.variant_name.casefold() == candidate.variant_name.casefold()
            ):
                raise ValueError(f"A variant named '{candidate.variant_name}' already exists")

    def _write(self, entries: list[PromptEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            file_descriptor, temp_name = tempfile.mkstemp(prefix=".prompt-library-", suffix=".tmp", dir=self.path.parent)
            temp_path = Path(temp_name)
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
                json.dump(
                    {"format_version": FORMAT_VERSION, "entries": [asdict(entry) for entry in entries]},
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
