import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import gradio as gr
from PIL import Image

EXTENSION_DIR = Path(__file__).resolve().parents[1] / "extensions-builtin" / "sd_forge_prompt_workspace"
sys.path.insert(0, str(EXTENSION_DIR))

from prompt_workspace_data import MAX_SECTIONS, PromptWorkspaceStore
from storyboard_data import StoryboardStore
from storyboard_ui import (
    add_selection,
    cast_values,
    delete_asset,
    delete_project,
    delete_scene,
    delete_variant,
    gallery_callback,
    load_variant,
    step_variant,
    take_image,
)


class StoryboardGalleryCallbackTests(unittest.TestCase):
    def test_gallery_hook_does_not_create_panels_for_runtime_copies(self):
        for context, render in ((None, True), (object(), False)):
            for tab in ("txt2img", "img2img"):
                with (
                    self.subTest(context=context, render=render, tab=tab),
                    patch("storyboard_ui.get_blocks_context", return_value=context),
                    patch("storyboard_ui.gr") as ui,
                ):
                    factory = Mock(return_value=SimpleNamespace(path=Path("library.json")))
                    callback = gallery_callback(factory)
                    callback(SimpleNamespace(elem_id=f"{tab}_gallery"), render=render)
                    factory.assert_not_called()
                    self.assertEqual([], ui.mock_calls)

    def test_gallery_hook_binds_original_components_and_ignores_runtime_info_updates(self):
        for tab in ("txt2img", "img2img"):
            with (
                self.subTest(tab=tab),
                patch("storyboard_ui.get_blocks_context", return_value=object()) as context,
                patch("storyboard_ui.gr", MagicMock()) as ui,
                patch("storyboard_ui.StoryboardStore"),
                patch("storyboard_ui.project_choices", return_value=[]),
            ):
                project, scene, variant = Mock(), Mock(), Mock()
                refresh, take = Mock(), Mock()
                ui.Dropdown.side_effect = [project, scene, variant]
                ui.Button.side_effect = [refresh, take]
                factory = Mock(return_value=SimpleNamespace(path=Path("library.json")))
                callback = gallery_callback(factory)
                gallery = SimpleNamespace(elem_id=f"{tab}_gallery")
                info = SimpleNamespace(elem_id=f"generation_info_{tab}")
                callback(gallery)
                callback(info)
                factory.assert_called_once_with()
                project.input.assert_called_once()
                scene.input.assert_called_once()
                take.click.assert_called_once()
                inputs = take.click.call_args.kwargs["inputs"]
                self.assertIs(gallery, inputs[3])
                self.assertIs(info, inputs[4])

                ui.reset_mock()
                factory.reset_mock()
                take.click.reset_mock()
                callback(SimpleNamespace(elem_id=info.elem_id), render=False)
                context.return_value = None
                callback(SimpleNamespace(elem_id=info.elem_id))
                factory.assert_not_called()
                self.assertEqual([], ui.mock_calls)
                take.click.assert_not_called()


class StoryboardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.library = PromptWorkspaceStore(self.root / "library.json")
        self.store = StoryboardStore(self.root / "storyboards", self.library)
        self.project = self.store.create("Story", str(self.root / "images"))
        self.pid = self.project["id"]
        self.scene = self.store.add_scene(self.pid, "Arrival")
        self.sid = self.scene["id"]

    def snippet(self, entry_id, text):
        return self.library.save_entry(
            entry_id, kind="snippet", name="Character", category="People",
            tags=[], prompt=text, negative_prompt="", reference="character",
        )

    def test_scenes_reuse_characters_outfits_and_freeze_variants(self):
        snippet = self.snippet(None, "red hair")
        character, outfit = self.store.save_asset(self.pid, None, "Character", "Mara", "<character>", "Coat", "blue coat")
        self.store.select_asset(self.pid, self.sid, character["id"], outfit["id"])
        self.library.save_entry(
            None, kind="snippet", name="Unrelated", category="Other", tags=[],
            prompt="unrelated text", negative_prompt="", reference="unrelated",
        )
        self.store.update_draft(self.pid, self.sid, "section", "walking", 1)
        variant = self.store.save_variant(self.pid, self.sid, "First")
        self.assertEqual(["character"], list(variant["draft"]["context"]["references"]))
        self.assertIn("Mara: red hair, blue coat", variant["prompt"])
        self.assertIn("walking", variant["prompt"])

        self.snippet(snippet.id, "dark hair")
        self.store.save_asset(self.pid, character["id"], "Character", "Mara", "new <character>")
        project = self.store.load(self.pid)
        scene = project["scenes"][0]
        self.assertEqual(variant["prompt"], scene["variants"][0]["prompt"])
        self.assertIn("new dark hair", self.store.resolve(project, scene["draft"])[0])
        self.store.update_draft(self.pid, self.sid, "section", "running", 1)
        restored = self.store.select_variant(self.pid, self.sid, variant["id"])
        self.assertEqual("walking", restored["draft"]["sections"][1]["text"])
        self.assertIn("Mara: red hair, blue coat", self.store.resolve(self.store.load(self.pid), restored["draft"])[0])
        self.store.refresh_draft_context(self.pid, self.sid)
        project = self.store.load(self.pid)
        self.assertIn("new dark hair", self.store.resolve(project, project["scenes"][0]["draft"])[0])
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.save_variant(self.pid, self.sid, "first")

    def test_each_variant_has_its_own_image_and_publication_is_clean(self):
        self.store.update_draft(self.pid, self.sid, "section", "station", 0)
        first = self.store.save_variant(self.pid, self.sid, "A")
        second = self.store.save_variant(self.pid, self.sid, "B")
        red = Image.new("RGB", (2, 2), "red")
        blue = Image.new("RGB", (2, 2), "blue")
        blue.info["Comment"] = "private comment"
        blue.info["icc_profile"] = b"private profile"
        exif = Image.Exif()
        exif[0x010F] = "private camera"
        blue.info["exif"] = exif.tobytes()
        with self.assertRaisesRegex(ValueError, "Select this variant"):
            self.store.assign_image(self.pid, self.sid, first["id"], red, "")
        self.store.select_variant(self.pid, self.sid, first["id"])
        first_path = self.store.assign_image(self.pid, self.sid, first["id"], red, "secret metadata")
        self.store.select_variant(self.pid, self.sid, second["id"])
        second_path = self.store.assign_image(self.pid, self.sid, second["id"], blue, "other metadata")
        self.assertNotEqual(first_path, second_path)
        self.store.select_variant(self.pid, self.sid, first["id"])
        with patch.object(self.store, "_write", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.store.assign_image(self.pid, self.sid, first["id"], blue, "replacement")
        self.assertTrue(first_path.is_file())
        old_path = first_path
        first_path = self.store.assign_image(self.pid, self.sid, first["id"], blue, "replacement")
        self.assertFalse(old_path.exists())
        with Image.open(first_path) as image:
            self.assertEqual((0, 0, 255), image.getpixel((0, 0)))
            self.assertEqual("replacement", image.info["parameters"])
            self.assertEqual("private comment", image.info["Comment"])
        self.store.select_variant(self.pid, self.sid, second["id"])

        next_scene = self.store.add_scene(self.pid, "Departure")
        self.store.update_draft(self.pid, next_scene["id"], "section", "train", 0)
        third = self.store.save_variant(self.pid, next_scene["id"], "C")
        with self.assertRaisesRegex(ValueError, "has no image"):
            self.store.publish(self.pid)
        self.store.assign_image(self.pid, next_scene["id"], third["id"], red, "private")
        output = self.store.publish(self.pid)
        self.assertEqual(["1.png", "2.png"], sorted(path.name for path in output.iterdir()))
        with Image.open(output / "1.png") as image:
            self.assertEqual({}, image.info)
            self.assertEqual((0, 0, 255), image.getpixel((0, 0)))
        with Image.open(output / "2.png") as image:
            self.assertEqual({}, image.info)
            self.assertEqual((255, 0, 0), image.getpixel((0, 0)))
        self.assertNotEqual(output, self.store.publish(self.pid))
        self.store.move_scene(self.pid, next_scene["id"], -1)
        reordered = self.store.publish(self.pid)
        with Image.open(reordered / "1.png") as image:
            self.assertEqual((255, 0, 0), image.getpixel((0, 0)))
        self.assertTrue(first_path.is_file())
        self.assertTrue(second_path.is_file())

    def test_gallery_assignment_uses_selected_image_and_infotext(self):
        self.store.update_draft(self.pid, self.sid, "section", "scene", 0)
        variant = self.store.save_variant(self.pid, self.sid, "A")
        gallery = [(Image.new("RGB", (1, 1), color), None) for color in ("red", "blue")]
        info = json.dumps({"infotexts": ["first", "second"]})
        with self.assertRaisesRegex(ValueError, "Generate an image"):
            take_image(self.store, self.pid, self.sid, variant["id"], None, info, -1)
        with self.assertRaisesRegex(ValueError, "Select an image"):
            take_image(self.store, self.pid, self.sid, variant["id"], gallery, info, -1)
        take_image(self.store, self.pid, self.sid, variant["id"], gallery, info, 1)
        image_path = self.store.load(self.pid)["scenes"][0]["variants"][0]["image"]
        with Image.open(image_path) as image:
            self.assertEqual((0, 0, 255), image.getpixel((0, 0)))
            self.assertEqual("second", image.info["parameters"])

    def test_storyboard_actions_report_missing_selections_without_clearing_inputs(self):
        cases = (
            (delete_project(self.store, None), MAX_SECTIONS + 11, "Select a storyboard to delete"),
            (delete_scene(self.store, self.pid, None), MAX_SECTIONS + 9, "Select a storyboard and scene to delete"),
            (load_variant(self.store, self.pid, self.sid, None), MAX_SECTIONS + 8, "Select a storyboard, scene and variant to load"),
            (step_variant(self.store, None, None, 1), MAX_SECTIONS + 8, "Select a storyboard and scene first"),
            (step_variant(self.store, self.pid, self.sid, 1), MAX_SECTIONS + 8, "This scene has no saved variants"),
            (delete_variant(self.store, self.pid, self.sid, None), 5, "Select a storyboard, scene and variant to delete"),
        )
        for result, length, message in cases:
            with self.subTest(message=message):
                self.assertEqual(length, len(result))
                self.assertEqual(message, result[-1])
                self.assertTrue(all(value == {"__type__": "update"} for value in result[:-1]))
        self.assertEqual(self.sid, self.store.load(self.pid)["scenes"][0]["id"])

    def test_variant_navigation_and_deletion_take_effect_immediately(self):
        self.store.update_draft(self.pid, self.sid, "section", "first draft", 0)
        first = self.store.save_variant(self.pid, self.sid, "First")
        self.store.update_draft(self.pid, self.sid, "section", "second draft", 0)
        second = self.store.save_variant(self.pid, self.sid, "Second")
        self.store.update_draft(self.pid, self.sid, "section", "unsaved changes", 0)

        result = step_variant(self.store, self.pid, self.sid, -1)
        scene = self.store.load(self.pid)["scenes"][0]
        self.assertEqual(first["id"], scene["selected_variant"])
        self.assertEqual("first draft", scene["draft"]["sections"][0]["text"])
        self.assertEqual((MAX_SECTIONS + 8, ""), (len(result), result[-1]))

        result = load_variant(self.store, self.pid, self.sid, second["id"])
        self.assertEqual(second["id"], self.store.load(self.pid)["scenes"][0]["selected_variant"])
        self.assertEqual((MAX_SECTIONS + 8, ""), (len(result), result[-1]))
        result = delete_variant(self.store, self.pid, self.sid, second["id"])
        self.assertEqual("Variant deleted", result[-1])
        self.assertEqual([first["id"]], [variant["id"] for variant in self.store.load(self.pid)["scenes"][0]["variants"]])

    def test_saving_assets_by_name_updates_them_with_outfits(self):
        first, _ = self.store.save_asset(self.pid, None, "Character", "Mara", "red hair")
        again, outfit = self.store.save_asset(self.pid, None, "Character", "mara", "dark hair", "Coat", "blue coat")
        self.assertEqual(first["id"], again["id"])
        same, _ = self.store.save_asset(self.pid, first["id"], "Character", "Mara", "dark hair", "coat", "red coat")
        assets = self.store.load(self.pid)["assets"]
        self.assertEqual(1, len(assets))
        self.assertEqual("dark hair", assets[0]["text"])
        self.assertEqual([(outfit["id"], "coat", "red coat")], [(item["id"], item["name"], item["text"]) for item in assets[0]["outfits"]])
        with self.assertRaisesRegex(ValueError, "already exists as Character"):
            self.store.save_asset(self.pid, None, "Object", "Mara", "a statue")
        with self.assertRaisesRegex(ValueError, "Only characters"):
            self.store.save_asset(self.pid, None, "Object", "Lamp", "a lamp", "Coat", "blue coat")
        self.assertEqual(1, len(self.store.load(self.pid)["assets"]))

    def test_scene_asset_actions_show_selection_and_deletion_updates_drafts(self):
        character, outfit = self.store.save_asset(self.pid, None, "Character", "Mara", "red hair", "Coat", "blue coat")
        selection, message = add_selection(self.store, self.pid, self.sid, character["id"], outfit["id"])
        self.assertEqual(([("Mara / Coat", character["id"])], character["id"]), (selection["choices"], selection["value"]))
        self.assertEqual("Added to scene", message)
        self.assertEqual((gr.skip(), "Select a scene and a project asset first"), add_selection(self.store, self.pid, None, character["id"], None))

        self.store.select_variant(self.pid, self.sid, self.store.save_variant(self.pid, self.sid, "Frozen")["id"])
        other = self.store.add_scene(self.pid, "Other")
        self.store.select_asset(self.pid, other["id"], character["id"], None)
        result = delete_asset(self.store, self.pid, other["id"], character["id"])
        self.assertEqual(([], [], "Asset deleted"), (result[0]["choices"], result[2]["choices"], result[3]))
        project = self.store.load(self.pid)
        frozen, other_scene = project["scenes"]
        self.assertEqual([], other_scene["draft"]["selections"])
        self.assertIn("Mara: red hair, blue coat", self.store.resolve(project, frozen["draft"])[0])
        lamp, _ = self.store.save_asset(self.pid, None, "Object", "Lamp", "a lamp")
        assets, outfits = cast_values(self.store, self.pid, character["id"], outfit["id"])
        self.assertEqual(([("Object: Lamp", lamp["id"])], None, None), (assets["choices"], assets["value"], outfits["value"]))
        self.assertEqual(lamp["id"], cast_values(self.store, self.pid, lamp["id"], None)[0]["value"])

    def test_new_scenes_use_configured_section_names(self):
        self.library.save_section_names(["Character", "Place"])
        scene = self.store.add_scene(self.pid, "Custom")
        self.assertEqual(["Character", "Place"], [section["name"] for section in scene["draft"]["sections"]])
        location, _ = self.store.save_asset(self.pid, None, "Location", "Station", "a station")
        self.store.select_asset(self.pid, scene["id"], location["id"], None)
        project = self.store.load(self.pid)
        selected = next(item for item in project["scenes"] if item["id"] == scene["id"])
        self.assertEqual("a station", self.store.resolve(project, selected["draft"])[0])

    def test_publication_preserves_exif_orientation_without_metadata(self):
        self.store.update_draft(self.pid, self.sid, "section", "scene", 0)
        variant = self.store.save_variant(self.pid, self.sid, "A")
        image = Image.new("RGB", (2, 1))
        image.putpixel((0, 0), (255, 0, 0))
        image.putpixel((1, 0), (0, 0, 255))
        exif = Image.Exif()
        exif[274] = 6
        image.info["exif"] = exif.tobytes()
        self.store.assign_image(self.pid, self.sid, variant["id"], image, "metadata")
        output = self.store.publish(self.pid)
        with Image.open(output / "1.png") as published:
            self.assertEqual((1, 2), published.size)
            self.assertEqual((255, 0, 0), published.getpixel((0, 0)))
            self.assertEqual((0, 0, 255), published.getpixel((0, 1)))
            self.assertEqual({}, published.info)

    def test_publication_preserves_palette_transparency(self):
        self.store.update_draft(self.pid, self.sid, "section", "scene", 0)
        variant = self.store.save_variant(self.pid, self.sid, "A")
        image = Image.new("P", (2, 1))
        image.putpalette([255, 0, 0, 0, 0, 255] + [0] * 762)
        image.putpixel((1, 0), 1)
        image.info["transparency"] = 0
        self.store.assign_image(self.pid, self.sid, variant["id"], image, "")
        output = self.store.publish(self.pid)
        with Image.open(output / "1.png") as published:
            self.assertEqual((255, 0, 0, 0), published.getpixel((0, 0)))
            self.assertEqual((0, 0, 255, 255), published.getpixel((1, 0)))
            self.assertEqual({}, published.info)


if __name__ == "__main__":
    unittest.main()
