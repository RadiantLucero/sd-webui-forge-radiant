import json
import os
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
    reference_map,
    resolve_references,
)

_lock = threading.RLock()
_version = 1


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
            scene = {"id": new_id(), "name": required_text(name, "Scene name"), "draft": draft(self.library.section_names()), "variants": [], "selected_variant": None}
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
            self._write(project)

    def add_asset(self, project_id, kind, name, text):
        if kind not in ("Character", "Location", "Object"):
            raise ValueError("Invalid asset type")
        with _lock:
            project = self.load(project_id)
            asset = {"id": new_id(), "kind": kind, "name": required_text(name, "Asset name"), "text": required_text(text, "Asset text"), "outfits": []}
            project["assets"].append(asset)
            self._write(project)
            return asset

    def add_outfit(self, project_id, asset_id, name, text):
        with _lock:
            project = self.load(project_id)
            asset = self._find(project["assets"], asset_id, "Character")
            if asset["kind"] != "Character":
                raise ValueError("Only characters can have outfits")
            outfit = {"id": new_id(), "name": required_text(name, "Outfit name"), "text": required_text(text, "Outfit text")}
            asset["outfits"].append(outfit)
            self._write(project)
            return outfit

    def update_asset(self, project_id, asset_id, name, text):
        with _lock:
            project = self.load(project_id)
            asset = self._find(project["assets"], asset_id, "Asset")
            asset["name"] = required_text(name, "Asset name")
            asset["text"] = required_text(text, "Asset text")
            self._write(project)

    def update_outfit(self, project_id, asset_id, outfit_id, name, text):
        with _lock:
            project = self.load(project_id)
            asset = self._find(project["assets"], asset_id, "Character")
            outfit = self._find(asset["outfits"], outfit_id, "Outfit")
            outfit["name"] = required_text(name, "Outfit name")
            outfit["text"] = required_text(text, "Outfit text")
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
            selections[:] = [item for item in selections if item["asset_id"] != asset_id]
            selections.append({"asset_id": asset_id, "outfit_id": outfit_id or None})
            self._write(project)

    def remove_asset(self, project_id, scene_id, asset_id):
        with _lock:
            project = self.load(project_id)
            scene = self._find(project["scenes"], scene_id, "Scene")
            scene["draft"]["selections"] = [item for item in scene["draft"]["selections"] if item["asset_id"] != asset_id]
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
        assets = context["assets"] if context is not None else project["assets"]
        for selected in scene_draft["selections"]:
            asset = self._find(assets, selected["asset_id"], "Asset")
            parts = [asset["text"]]
            if selected["outfit_id"]:
                parts.append(self._find(asset["outfits"], selected["outfit_id"], "Outfit")["text"])
            section_name = "Subject" if asset["kind"] in ("Character", "Object") else "Environment"
            fallback_index = 0 if section_name == "Subject" else min(2, len(sections) - 1)
            target = next((section for section in sections if section["name"].casefold() == section_name.casefold()), sections[fallback_index])
            text = ", ".join(parts)
            if asset["kind"] == "Character":
                text = f"{asset['name']}: {text}"
            target["text"] = "; ".join(part for part in (target["text"].strip(), text) if part)
        references = context["references"] if context is not None else reference_map(self.library.load())
        positive = compose_prompt(scene_draft["template"], sections, references)
        negative = resolve_references(scene_draft["negative"], references)
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
