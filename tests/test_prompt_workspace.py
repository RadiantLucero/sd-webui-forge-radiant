import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extensions-builtin" / "sd_forge_prompt_workspace"))

from prompt_workspace_data import PromptWorkspaceStore, library_path


class PromptWorkspaceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "prompt-workspace" / "library.json"
        self.store = PromptWorkspaceStore(self.path)

    def save_prompt(self, name="Rainy street", variant_name="Base", prompt="rain at night"):
        return self.store.save_entry(
            None,
            kind="prompt",
            name=name,
            variant_name=variant_name,
            category="Weather",
            tags=["night", "rain"],
            prompt=prompt,
            negative_prompt="",
        )

    def test_library_path_defaults_to_extension_or_uses_configured_directory(self):
        extension_dir = Path(self.temp_dir.name) / "extension"
        custom_dir = Path(self.temp_dir.name) / "my library"

        self.assertEqual(extension_dir / "data" / "library.json", library_path(extension_dir, ""))
        self.assertEqual(extension_dir / "data" / "library.json", library_path(extension_dir, "  "))
        self.assertEqual(custom_dir / "library.json", library_path(extension_dir, str(custom_dir)))

    def test_switching_library_directory_does_not_move_or_replace_libraries(self):
        extension_dir = Path(self.temp_dir.name) / "extension"
        default_path = library_path(extension_dir, "")
        custom_dir = Path(self.temp_dir.name) / "custom"
        custom_path = library_path(extension_dir, str(custom_dir))
        default_store = PromptWorkspaceStore(default_path)
        custom_store = PromptWorkspaceStore(custom_path)

        for store, name in ((default_store, "Default"), (custom_store, "Custom")):
            store.save_entry(
                None,
                kind="snippet",
                name=name,
                category="General",
                tags=[],
                prompt=name,
                negative_prompt="",
            )

        self.assertEqual(["Default"], [entry.name for entry in default_store.load()])
        self.assertEqual(["Custom"], [entry.name for entry in custom_store.load()])

    def test_saves_and_reloads_snippets_and_prompts(self):
        snippet = self.store.save_entry(
            None,
            kind="snippet",
            name="Cold light",
            category="Lighting",
            tags=["blue"],
            prompt="cold blue light",
            negative_prompt="",
        )
        prompt = self.save_prompt()

        self.assertEqual([snippet, prompt], self.store.load())
        self.assertTrue(self.path.is_file())

    def test_saves_named_variants_and_steps_through_them(self):
        base = self.save_prompt()
        alternate = self.store.save_variant(
            base.id,
            variant_name="Foggy",
            prompt="rain and fog at night",
            negative_prompt="",
        )

        self.assertEqual(["Base", "Foggy"], [entry.variant_name for entry in self.store.variants(base.id)])
        self.assertEqual(base.variant_group, alternate.variant_group)
        self.assertNotEqual(base.id, alternate.id)

    def test_updating_a_prompt_keeps_its_variants_together(self):
        base = self.save_prompt()
        alternate = self.store.save_variant(
            base.id,
            variant_name="Foggy",
            prompt="rain and fog at night",
            negative_prompt="",
        )

        updated = self.store.save_entry(
            base.id,
            kind="prompt",
            name=base.name,
            variant_name=base.variant_name,
            category=base.category,
            tags=base.tags,
            prompt="rain at night, wider view",
            negative_prompt="",
        )

        self.assertEqual(base.variant_group, updated.variant_group)
        self.assertEqual({base.id, alternate.id}, {entry.id for entry in self.store.variants(updated.id)})

    def test_rejects_duplicate_variant_names_without_overwriting_data(self):
        base = self.save_prompt()
        before = self.path.read_text(encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.save_variant(base.id, variant_name="base", prompt="changed", negative_prompt="")

        self.assertEqual(before, self.path.read_text(encoding="utf-8"))

    def test_updates_and_deletes_selected_entries(self):
        snippet = self.store.save_entry(
            None,
            kind="snippet",
            name="Clouds",
            category="Weather",
            tags=[],
            prompt="overcast sky",
            negative_prompt="",
        )
        updated = self.store.save_entry(
            snippet.id,
            kind="snippet",
            name="Heavy clouds",
            category="Weather",
            tags=[],
            prompt="dark overcast sky",
            negative_prompt="",
        )

        self.assertEqual(snippet.id, updated.id)
        self.assertEqual([updated], self.store.load())
        self.assertTrue(self.store.delete_entry(updated.id))
        self.assertFalse(self.store.delete_entry(updated.id))
        self.assertEqual([], self.store.load())

    def test_reports_invalid_json_instead_of_replacing_it(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Invalid prompt library JSON"):
            self.store.load()

        self.assertEqual("{", self.path.read_text(encoding="utf-8"))

    def test_rejects_empty_prompt_content(self):
        with self.assertRaisesRegex(ValueError, "Enter a prompt"):
            self.store.save_entry(
                None,
                kind="snippet",
                name="Empty",
                category="General",
                tags=[],
                prompt=" ",
                negative_prompt=" ",
            )

    def save_snippet(self, name, prompt, reference="", entry_id=None):
        return self.store.save_entry(
            entry_id,
            kind="snippet",
            name=name,
            category="Characters",
            tags=[],
            prompt=prompt,
            negative_prompt="",
            reference=reference,
        )

    def test_template_places_sections_and_free_text(self):
        entry = self.store.save_entry(
            None,
            kind="prompt",
            name="Scene A",
            category="Scenes",
            tags=[],
            prompt="<<Subject>>\n\nBehind him, <<Lighting>> <<Style>>",
            negative_prompt="",
            sections=[
                {"name": "Subject", "text": "A traveler\n  with a gray coat."},
                {"name": "Lighting", "text": "soft morning light."},
                {"name": "Style", "text": " "},
                {"name": "Unused", "text": "not placed"},
            ],
        )

        self.assertEqual("A traveler with a gray coat. Behind him, soft morning light.", entry.prompt)
        self.assertEqual("<<Subject>>\n\nBehind him, <<Lighting>> <<Style>>", entry.template)
        self.assertEqual([entry], self.store.load())

    def test_rejects_placeholders_for_unknown_sections(self):
        with self.assertRaisesRegex(ValueError, "Unknown section <<Lightning>>"):
            self.store.save_entry(
                None, kind="prompt", name="X", category="C", tags=[], prompt="<<Lightning>>", negative_prompt="", sections=[{"name": "Lighting", "text": "x"}]
            )

    def test_resolves_nested_references_case_insensitively_and_keeps_lora_syntax(self):
        self.save_snippet("Character A body", "tall, <character_a_body_global>", "character_A_Body")
        self.save_snippet("Character A global", "dark hair", "character_a_body_global")
        self.save_snippet("Character A", "Character A, <character_A_Body>", "character_A")

        entry = self.store.save_entry(
            None, kind="prompt", name="Scene", category="C", tags=[], prompt="<character_a> walks <lora:style:0.8>", negative_prompt=""
        )

        self.assertEqual("Character A, tall, dark hair walks <lora:style:0.8>", entry.prompt)
        self.assertEqual("<character_a> walks <lora:style:0.8>", entry.template)

    def test_rejects_unknown_and_circular_references_without_writing(self):
        first = self.save_snippet("A", "a", "ref_a")
        self.save_snippet("B", "<ref_a>", "ref_b")
        before = self.path.read_text(encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Unknown reference <missing>"):
            self.store.save_entry(None, kind="prompt", name="P", category="C", tags=[], prompt="<missing>", negative_prompt="")
        with self.assertRaisesRegex(ValueError, "Circular reference"):
            self.save_snippet("A", "<ref_b>", "ref_a", entry_id=first.id)
        self.assertEqual(before, self.path.read_text(encoding="utf-8"))

    def test_snippets_may_reference_parts_defined_later(self):
        self.save_snippet("Character A", "Character A, <character_a_body>", "character_a")
        with self.assertRaisesRegex(ValueError, "Unknown reference <character_a_body>"):
            self.store.preview("snippet", "<character_a>")

        self.save_snippet("Character A body", "tall", "character_a_body")
        self.assertEqual("Character A, tall", self.store.preview("snippet", "<character_a>"))

    def test_protects_references_that_are_in_use(self):
        body = self.save_snippet("Body", "tall", "body")
        self.save_snippet("Character A", "<body>", "character_a")

        with self.assertRaisesRegex(ValueError, "still used by: Character A"):
            self.store.delete_entry(body.id)
        with self.assertRaisesRegex(ValueError, "still used by: Character A"):
            self.save_snippet("Body", "tall", "renamed", entry_id=body.id)
        with self.assertRaisesRegex(ValueError, "already used"):
            self.save_snippet("Other", "x", "BODY")

    def test_free_text_is_normalized_to_single_spaces(self):
        entry = self.store.save_entry(
            None,
            kind="snippet",
            name="Sad light",
            category="Lighting",
            tags=[],
            prompt="  Muted light,\n\nheavy  shadows. ",
            negative_prompt="bright\ncolors",
            section="Lighting",
        )

        self.assertEqual("Muted light, heavy shadows.", entry.prompt)
        self.assertEqual("bright colors", entry.negative_prompt)

    def test_default_section_names_are_stored_with_the_library(self):
        self.assertEqual(["Subject", "Action", "Environment", "Framing", "Lighting", "Details", "Style"], self.store.section_names())
        self.save_prompt()
        self.store.save_section_names(["Character", " Mood ", ""])

        self.assertEqual(["Character", "Mood"], self.store.section_names())
        self.assertEqual(1, len(self.store.load()))
        with self.assertRaisesRegex(ValueError, "unique"):
            self.store.save_section_names(["Mood", "mood"])
        with self.assertRaisesRegex(ValueError, "At most 10"):
            self.store.save_section_names([f"S{index}" for index in range(11)])

    def test_loads_entries_saved_before_templates_existed(self):
        self.path.parent.mkdir(parents=True)
        legacy = {"id": "a", "kind": "prompt", "name": "Old", "category": "C", "tags": [], "prompt": "x", "negative_prompt": "", "variant_group": "g", "variant_name": "Base"}
        self.path.write_text(json.dumps({"format_version": 1, "entries": [legacy]}), encoding="utf-8")

        entry = self.store.load()[0]
        self.assertIsNone(entry.template)
        self.assertEqual("x", entry.source_text())

    def test_rejects_unsupported_library_format(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({"format_version": 2, "entries": []}), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Unsupported prompt library format"):
            self.store.load()


if __name__ == "__main__":
    unittest.main()
