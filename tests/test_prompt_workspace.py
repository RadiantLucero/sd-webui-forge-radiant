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

    def test_rejects_unsupported_library_format(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({"format_version": 2, "entries": []}), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Unsupported prompt library format"):
            self.store.load()


if __name__ == "__main__":
    unittest.main()
