import json
import os
import re
import tempfile
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps, PngImagePlugin
from prompt_workspace_data import (
    REFERENCE_PATTERN,
    compose_prompt,
    join_prompt_parts,
    reference_map,
    resolve_references,
    substitute_prompt,
)

_lock = threading.RLock()
_version = 1
ASSET_REFERENCE_PATTERN = re.compile(r"<asset:\s*([^<>]+?)\s*>")


def asset_reference(name):
    return f"<asset:{name}>"


def draft_texts(scene_draft):
    return [scene_draft["template"], *(section["text"] for section in scene_draft["sections"]), scene_draft["negative"]]


def referenced_asset_names(scene_draft):
    return {match.group(1).casefold() for text in draft_texts(scene_draft) for match in ASSET_REFERENCE_PATTERN.finditer(text)}


def remove_asset_references(scene_draft, name):
    def remove(text):
        return ASSET_REFERENCE_PATTERN.sub(lambda match: "" if match.group(1).casefold() == name.casefold() else match.group(0), text)

    scene_draft["template"] = remove(scene_draft["template"])
    scene_draft["negative"] = remove(scene_draft["negative"])
    for section in scene_draft["sections"]:
        section["text"] = remove(section["text"])


def new_id():
    return uuid.uuid4().hex


def required_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")
    return value.strip()


def draft(section_names):
    return {
        "template": " ".join(f"<<{name}>>" for name in section_names),
        "sections": [{"name": name, "text": ""} for name in section_names],
        "negative": "",
        "selections": [],
        "context": None,
        "asset_placement": "explicit",
    }


class StoryboardStore:
    def __init__(self, directory, library):
        self.directory = Path(directory)
        self.library = library

    def _path(self, project_id):
        if not isinstance(project_id, str) or len(project_id) != 32 or any(c not in "0123456789abcdef" for c in project_id):
            raise ValueError("Invalid storyboard ID")
        return self.directory / f"{project_id}.json"

    def load(self, project_id):
        path = self._path(project_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError("Storyboard no longer exists") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid storyboard JSON: {path}") from error
        if not isinstance(data, dict) or data.get("format_version") != _version or data.get("id") != project_id:
            raise ValueError(f"Unsupported or invalid storyboard: {path}")
        if not isinstance(data.get("name"), str) or not isinstance(data.get("image_directory"), str):
            raise ValueError(f"Invalid storyboard fields: {path}")
        if not isinstance(data.get("assets"), list) or not isinstance(data.get("scenes"), list):
            raise ValueError(f"Invalid storyboard assets or scenes: {path}")
        return data

    def list_projects(self):
        if not self.directory.exists():
            return []
        projects = [self.load(path.stem) for path in self.directory.glob("*.json")]
        return sorted(projects, key=lambda project: (project["name"].casefold(), project["id"]))

    def _write(self, project):
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self._path(project["id"])
        temporary = None
        try:
            fd, name = tempfile.mkstemp(prefix=".storyboard-", suffix=".tmp", dir=self.directory)
            temporary = Path(name)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(project, file, ensure_ascii=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    def create(self, name, image_directory):
        with _lock:
            project = {
                "format_version": _version,
                "id": new_id(),
                "name": required_text(name, "Storyboard name"),
                "image_directory": required_text(image_directory, "Image directory"),
                "assets": [],
                "scenes": [],
            }
            self._write(project)
            return project

    def rename(self, project_id, name):
        with _lock:
            project = self.load(project_id)
            project["name"] = required_text(name, "Storyboard name")
            self._write(project)

    def delete(self, project_id):
        with _lock:
            self.load(project_id)
            self._path(project_id).unlink()

    @staticmethod
    def _find(items, item_id, label):
        item = next((item for item in items if item["id"] == item_id), None)
        if item is None:
            raise ValueError(f"{label} no longer exists")
        return item

    def add_scene(self, project_id, name):
        with _lock:
            project = self.load(project_id)
            scene_draft = deepcopy(project["scenes"][-1]["draft"]) if project["scenes"] else draft(self.library.section_names())
            scene = {"id": new_id(), "name": required_text(name, "Scene name"), "draft": scene_draft, "variants": [], "selected_variant": None}
            project["scenes"].append(scene)
            self._write(project)
            return scene

    def set_image_directory(self, project_id, image_directory):
        with _lock:
            project = self.load(project_id)
            project["image_directory"] = required_text(image_directory, "Image directory")
            self._write(project)

    def delete_scene(self, project_id, scene_id):
        with _lock:
            project = self.load(project_id)
            scene = self._find(project["scenes"], scene_id, "Scene")
            project["scenes"].remove(scene)
            self._write(project)

    def rename_scene(self, project_id, scene_id, name):
        with _lock:
            project = self.load(project_id)
            scene = self._find(project["scenes"], scene_id, "Scene")
            scene["name"] = required_text(name, "Scene name")
            self._write(project)

    def move_scene(self, project_id, scene_id, direction):
        with _lock:
            project = self.load(project_id)
            scene = self._find(project["scenes"], scene_id, "Scene")
            index = project["scenes"].index(scene)
            target = max(0, min(index + direction, len(project["scenes"]) - 1))
            project["scenes"].insert(target, project["scenes"].pop(index))
            self._write(project)

    def update_draft(self, project_id, scene_id, field, value, section_index=None):
        if not isinstance(value, str):
            raise ValueError("Prompt text must be text")
        with _lock:
            project = self.load(project_id)
            scene = self._find(project["scenes"], scene_id, "Scene")
            if field == "section":
                if not isinstance(section_index, int) or not 0 <= section_index < len(scene["draft"]["sections"]):
                    raise ValueError("Invalid section")
                scene["draft"]["sections"][section_index]["text"] = value
            elif field in ("template", "negative"):
                scene["draft"][field] = value
            else:
                raise ValueError("Invalid draft field")
            placed = referenced_asset_names(scene["draft"])
            names = {asset["id"]: asset["name"].casefold() for asset in self._draft_assets(project, scene["draft"])}
            for selection in scene["draft"]["selections"]:
                if names.get(selection["asset_id"]) in placed:
                    selection["placement"] = "explicit"
            self._write(project)

    @staticmethod
    def _draft_assets(project, scene_draft):
        context = scene_draft["context"]
        return context["assets"] if context is not None else project["assets"]

    @staticmethod
    def _named(items, name, preferred_id=None):
        matches = [item for item in items if item["name"].casefold() == name.casefold()]
        return next((item for item in matches if item["id"] == preferred_id), matches[0] if matches else None)

    @classmethod
    def _upsert_outfit(cls, asset, name, text):
        if asset["kind"] != "Character":
            raise ValueError("Only characters can have outfits")
        name = required_text(name, "Outfit name")
        text = required_text(text, "Outfit text")
        outfit = cls._named(asset["outfits"], name)
        if outfit is None:
            outfit = {"id": new_id(), "name": name, "text": text}
            asset["outfits"].append(outfit)
        else:
            outfit["name"], outfit["text"] = name, text
        return outfit

    def save_asset(self, project_id, selected_id, kind, name, text, outfit_name="", outfit_text=""):
        if kind not in ("Character", "Location", "Object"):
            raise ValueError("Invalid asset type")
        name = required_text(name, "Asset name")
        if "<" in name or ">" in name:
            raise ValueError("Asset names cannot contain < or >")
        text = required_text(text, "Asset text")
        with _lock:
            project = self.load(project_id)
            asset = self._named(project["assets"], name, selected_id)
            if asset is None:
                asset = {"id": new_id(), "kind": kind, "name": name, "text": text, "outfits": []}
                project["assets"].append(asset)
            elif asset["kind"] != kind:
                raise ValueError(f"An asset named '{asset['name']}' already exists as {asset['kind']}")
            else:
                asset["name"], asset["text"] = name, text
            outfit = None
            if (outfit_name or "").strip() or (outfit_text or "").strip():
                outfit = self._upsert_outfit(asset, outfit_name, outfit_text)
            self._write(project)
            return asset, outfit

    def save_outfit(self, project_id, asset_id, name, text):
        with _lock:
            project = self.load(project_id)
            outfit = self._upsert_outfit(self._find(project["assets"], asset_id, "Character"), name, text)
            self._write(project)
            return outfit

    def delete_asset(self, project_id, asset_id):
        with _lock:
            project = self.load(project_id)
            name = self._find(project["assets"], asset_id, "Asset")["name"]
            project["assets"] = [item for item in project["assets"] if item["id"] != asset_id]
            for scene in project["scenes"]:
                scene_draft = scene["draft"]
                if scene_draft["context"] is None:
                    scene_draft["selections"] = [item for item in scene_draft["selections"] if item["asset_id"] != asset_id]
                    remove_asset_references(scene_draft, name)
            self._write(project)

    def select_asset(self, project_id, scene_id, asset_id, outfit_id):
        with _lock:
            project = self.load(project_id)
            scene = self._find(project["scenes"], scene_id, "Scene")
            context = scene["draft"]["context"]
            assets = context["assets"] if context is not None else project["assets"]
            asset = self._find(assets, asset_id, "Asset in this draft; refresh the draft context")
            if outfit_id:
                self._find(asset["outfits"], outfit_id, "Outfit")
            selections = scene["draft"]["selections"]
            previous = next((item for item in selections if item["asset_id"] == asset_id), {})
            selections[:] = [item for item in selections if item["asset_id"] != asset_id]
            selections.append({**previous, "asset_id": asset_id, "outfit_id": outfit_id or None})
            self._write(project)

    def remove_asset(self, project_id, scene_id, asset_id):
        with _lock:
            project = self.load(project_id)
            scene = self._find(project["scenes"], scene_id, "Scene")
            asset = next((item for item in self._draft_assets(project, scene["draft"]) if item["id"] == asset_id), None)
            scene["draft"]["selections"] = [item for item in scene["draft"]["selections"] if item["asset_id"] != asset_id]
            if asset is not None:
                remove_asset_references(scene["draft"], asset["name"])
            self._write(project)

    def refresh_draft_context(self, project_id, scene_id):
        with _lock:
            project = self.load(project_id)
            scene = self._find(project["scenes"], scene_id, "Scene")
            scene["draft"]["context"] = None
            self._write(project)

    def resolve(self, project, scene_draft):
        sections = deepcopy(scene_draft["sections"])
        context = scene_draft["context"]
        assets = self._draft_assets(project, scene_draft)
        placed = referenced_asset_names(scene_draft)
        asset_texts = {}
        ambiguous = set()
        for selected in scene_draft["selections"]:
            asset = self._find(assets, selected["asset_id"], "Asset")
            text = asset["text"]
            if selected["outfit_id"]:
                outfit = self._find(asset["outfits"], selected["outfit_id"], "Outfit")
                text = join_prompt_parts(text, outfit["text"], ", ")
            if asset["kind"] == "Character":
                text = f"{asset['name']}: {text}"
            key = asset["name"].casefold()
            if key in asset_texts:
                ambiguous.add(key)
            asset_texts[key] = text
            if scene_draft.get("asset_placement") != "explicit" and selected.get("placement") != "explicit" and key not in placed:
                section_name = "Subject" if asset["kind"] in ("Character", "Object") else "Environment"
                fallback_index = 0 if section_name == "Subject" else min(2, len(sections) - 1)
                target = next((section for section in sections if section["name"].casefold() == section_name.casefold()), sections[fallback_index])
                target["text"] = join_prompt_parts(target["text"].strip(), text, "; ")

        def expand_assets(text):
            def replace(match):
                key = match.group(1).casefold()
                if key in ambiguous:
                    raise ValueError(f"Asset reference {match.group(0)} is ambiguous; delete the duplicate asset")
                if key not in asset_texts:
                    raise ValueError(f"Asset reference {match.group(0)} is not assigned to this scene")
                return asset_texts[key]

            return substitute_prompt(ASSET_REFERENCE_PATTERN, text, replace)

        for section in sections:
            section["text"] = expand_assets(section["text"])
        references = context["references"] if context is not None else reference_map(self.library.load())
        positive = compose_prompt(expand_assets(scene_draft["template"]), sections, references)
        negative = resolve_references(expand_assets(scene_draft["negative"]), references)
        return positive, negative

    def snapshot_context(self, project, scene_draft):
        selected_ids = {item["asset_id"] for item in scene_draft["selections"]}
        assets = deepcopy([asset for asset in project["assets"] if asset["id"] in selected_ids])
        available = reference_map(self.library.load())
        texts = [scene_draft["template"], scene_draft["negative"]]
        texts.extend(section["text"] for section in scene_draft["sections"])
        for asset in assets:
            texts.append(asset["text"])
            texts.extend(outfit["text"] for outfit in asset["outfits"])
        references = {}
        while texts:
            for match in REFERENCE_PATTERN.finditer(texts.pop()):
                key = match.group(1).casefold()
                if key in available and key not in references:
                    references[key] = available[key]
                    texts.append(available[key])
        return {"assets": assets, "references": references}

    def save_variant(self, project_id, scene_id, name):
        with _lock:
            project = self.load(project_id)
            scene = self._find(project["scenes"], scene_id, "Scene")
            name = required_text(name, "Variant name")
            if any(item["name"].casefold() == name.casefold() for item in scene["variants"]):
                raise ValueError("A variant with this name already exists")
            snapshot = deepcopy(scene["draft"])
            if snapshot["context"] is None:
                snapshot["context"] = self.snapshot_context(project, snapshot)
            positive, negative = self.resolve(project, snapshot)
            if not positive and not negative:
                raise ValueError("Enter a prompt before saving a variant")
            variant = {
                "id": new_id(), "name": name, "draft": snapshot,
                "prompt": positive, "negative": negative, "image": None, "infotext": "",
            }
            scene["variants"].append(variant)
            scene["selected_variant"] = variant["id"]
            self._write(project)
            return variant

    def select_variant(self, project_id, scene_id, variant_id):
        with _lock:
            project = self.load(project_id)
            scene = self._find(project["scenes"], scene_id, "Scene")
            variant = self._find(scene["variants"], variant_id, "Variant")
            scene["selected_variant"] = variant_id
            scene["draft"] = deepcopy(variant["draft"])
            self._write(project)
            return scene

    def delete_variant(self, project_id, scene_id, variant_id):
        with _lock:
            project = self.load(project_id)
            scene = self._find(project["scenes"], scene_id, "Scene")
            variant = self._find(scene["variants"], variant_id, "Variant")
            scene["variants"].remove(variant)
            if scene["selected_variant"] == variant_id:
                scene["selected_variant"] = None
            self._write(project)

    def assign_image(self, project_id, scene_id, variant_id, image, infotext):
        if not isinstance(image, Image.Image):
            raise ValueError("Select an image in the gallery")
        with _lock:
            project = self.load(project_id)
            scene = self._find(project["scenes"], scene_id, "Scene")
            variant = self._find(scene["variants"], variant_id, "Variant")
            if scene["selected_variant"] != variant_id:
                raise ValueError("Select this variant in the Storyboard before assigning an image")
            directory = Path(project["image_directory"]).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            prefix = f"{project_id}_{scene_id}_{variant_id}_"
            destination = directory / f"{prefix}{new_id()}.png"
            previous = Path(variant["image"]) if variant["image"] else None
            temporary = None
            try:
                fd, name = tempfile.mkstemp(prefix=".storyboard-image-", suffix=".png", dir=directory)
                os.close(fd)
                temporary = Path(name)
                oriented = ImageOps.exif_transpose(image)
                pnginfo = PngImagePlugin.PngInfo()
                for key, value in oriented.info.items():
                    if key != "parameters" and isinstance(key, str) and isinstance(value, str):
                        pnginfo.add_text(key, value)
                if infotext:
                    pnginfo.add_text("parameters", infotext)
                oriented.save(temporary, format="PNG", pnginfo=pnginfo, exif=oriented.info.get("exif"))
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
            variant["image"] = str(destination)
            variant["infotext"] = infotext
            committed = False
            try:
                self._write(project)
                committed = True
            finally:
                if not committed:
                    destination.unlink(missing_ok=True)
            if previous and previous.parent == directory and previous.name.startswith(prefix):
                try:
                    previous.unlink(missing_ok=True)
                except OSError as error:
                    raise OSError(f"Image was assigned, but the previous image could not be removed: {previous}") from error
            return destination

    def publish(self, project_id):
        with _lock:
            project = self.load(project_id)
            if not project["scenes"]:
                raise ValueError("Add scenes before publishing")
            images = []
            for scene in project["scenes"]:
                if not scene["selected_variant"]:
                    raise ValueError(f"Scene '{scene['name']}' has no selected variant")
                variant = self._find(scene["variants"], scene["selected_variant"], f"Selected variant for {scene['name']}")
                if not variant["image"] or not Path(variant["image"]).is_file():
                    raise ValueError(f"Scene '{scene['name']}' has no image for its selected variant")
                image_path = Path(variant["image"])
                with Image.open(image_path) as source:
                    source.verify()
                images.append(image_path)
            directory = Path(project["image_directory"]).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"published-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{new_id()[:8]}"
            destination.mkdir()
            for index, image_path in enumerate(images, start=1):
                with Image.open(image_path) as source:
                    image = ImageOps.exif_transpose(source)
                    if "transparency" in image.info:
                        image = image.convert("RGBA")
                    image.info.clear()
                    image.save(destination / f"{index}.png", format="PNG")
            return destination
