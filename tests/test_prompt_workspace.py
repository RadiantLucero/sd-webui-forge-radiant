import importlib.util
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

EXTENSION_DIR = Path(__file__).resolve().parents[1] / "extensions-builtin" / "sd_forge_prompt_workspace"
sys.path.insert(0, str(EXTENSION_DIR))

from prompt_workspace_data import (
    PromptWorkspaceStore,
    library_path,
    normalize_prompt_text,
)
from prompt_workspace_generation import install_prompt_resolver


class FakeProcessing:
    def __init__(self, prompt, negative_prompt="", hr_prompt=None, hr_negative_prompt=None):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.hr_prompt = hr_prompt
        self.hr_negative_prompt = hr_negative_prompt
        self.enable_hr = hr_prompt is not None or hr_negative_prompt is not None
        self.setup_calls = 0

    def setup_prompts(self):
        self.setup_calls += 1
        self.all_prompts = list(self.prompt) if isinstance(self.prompt, list) else [self.prompt]
        self.all_negative_prompts = (
            list(self.negative_prompt)
            if isinstance(self.negative_prompt, list)
            else [self.negative_prompt] * len(self.all_prompts)
        )
        self.all_hr_prompts = [self.hr_prompt] if self.hr_prompt is not None else None
        self.all_hr_negative_prompts = [self.hr_negative_prompt] if self.hr_negative_prompt is not None else None
        self.main_prompt = self.all_prompts[0]
        self.main_negative_prompt = self.all_negative_prompts[0]


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

    def test_prompt_normalization_removes_spaces_before_punctuation(self):
        self.assertEqual("one, two. three; four: five", normalize_prompt_text("one  ,  two . three ; four : five"))

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

    def test_upserts_entries_by_category_and_name(self):
        first = self.store.upsert_entry(
            kind="snippet", name=" Fog ", category=" Weather ", tags=[],
            prompt="light fog", negative_prompt="",
        )
        updated = self.store.upsert_entry(
            kind="snippet", name="Fog", category="Weather", tags=["mood"],
            prompt="thick fog", negative_prompt="",
        )
        other_category = self.store.upsert_entry(
            kind="snippet", name="Fog", category="Lighting", tags=[],
            prompt="backlit fog", negative_prompt="",
        )
        other_case = self.store.upsert_entry(
            kind="snippet", name="fog", category="Weather", tags=[],
            prompt="low fog", negative_prompt="",
        )

        self.assertEqual(first.id, updated.id)
        self.assertEqual("thick fog", updated.prompt)
        self.assertEqual(3, len({updated.id, other_category.id, other_case.id}))
        with self.assertRaisesRegex(ValueError, "another entry type"):
            self.store.upsert_entry(
                kind="prompt", name="Fog", category="Weather", tags=[],
                prompt="a foggy scene", negative_prompt="",
            )
        with self.assertRaisesRegex(ValueError, "already used"):
            self.store.save_entry(
                None, kind="snippet", name="Fog", category="Weather", tags=[],
                prompt="duplicate", negative_prompt="",
            )
        self.assertEqual(3, len(self.store.load()))

    def test_moving_a_selected_prompt_keeps_its_variants_and_is_atomic(self):
        base = self.store.upsert_entry(
            kind="prompt", name="Station", category="Scenes", variant_name="Base",
            tags=[], prompt="a station", negative_prompt="",
        )
        fog = self.store.upsert_entry(
            kind="prompt", name="Station", category="Scenes", variant_name="Fog",
            tags=[], prompt="a foggy station", negative_prompt="",
        )
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Unknown reference"):
            self.store.upsert_entry(
                selected_entry_id=base.id, kind="prompt", name="Station", category="Stories",
                variant_name="Base", tags=[], prompt="<missing>", negative_prompt="",
            )
        self.assertEqual(before, self.path.read_bytes())

        updated = self.store.upsert_entry(
            selected_entry_id=base.id, kind="prompt", name="Station", category="Stories",
            variant_name="Base", tags=["updated"], prompt="a new station", negative_prompt="",
        )
        variants = {entry.id: entry for entry in self.store.variants(base.id)}
        self.assertEqual({base.id, fog.id}, set(variants))
        self.assertEqual({"Stories"}, {entry.category for entry in variants.values()})
        self.assertEqual(base.variant_group, updated.variant_group)
        self.assertEqual("a new station", variants[base.id].prompt)
        self.assertEqual("a foggy station", variants[fog.id].prompt)

        dawn = self.store.upsert_entry(
            selected_entry_id=base.id, kind="prompt", name="Station", category="Places",
            variant_name="Dawn", tags=[], prompt="a station at dawn", negative_prompt="",
        )
        variants = self.store.variants(base.id)
        self.assertEqual({base.id, fog.id, dawn.id}, {entry.id for entry in variants})
        self.assertEqual({"Places"}, {entry.category for entry in variants})
        self.assertEqual(base.variant_group, dawn.variant_group)

    def test_moving_to_an_existing_name_does_not_overwrite_either_entry(self):
        for source_kind, target_kind in (("snippet", "snippet"), ("prompt", "prompt"), ("snippet", "prompt")):
            with self.subTest(source_kind=source_kind, target_kind=target_kind):
                store = PromptWorkspaceStore(Path(self.temp_dir.name) / f"{source_kind}-{target_kind}" / "library.json")
                source = store.upsert_entry(
                    kind=source_kind, name="Shared", category="Original", tags=[],
                    prompt="original", negative_prompt="",
                )
                store.upsert_entry(
                    kind=target_kind, name="Shared", category="Target", tags=[],
                    prompt="target", negative_prompt="",
                )
                before = store.path.read_bytes()
                with self.assertRaisesRegex(ValueError, "already used"):
                    store.upsert_entry(
                        selected_entry_id=source.id, kind=source_kind, name="Shared", category="Target",
                        tags=[], prompt="changed", negative_prompt="",
                    )
                self.assertEqual(before, store.path.read_bytes())

    def test_save_as_new_variant_does_not_move_the_selected_prompt(self):
        source = self.store.upsert_entry(
            kind="prompt", name="Station", category="Scenes", variant_name="Base",
            tags=[], prompt="a station", negative_prompt="",
        )
        target = self.store.upsert_entry(
            kind="prompt", name="Station", category="Places", variant_name="Base",
            tags=[], prompt="another station", negative_prompt="",
        )
        night = self.store.upsert_entry(
            selected_entry_id=source.id, kind="prompt", name="Station", category="Places",
            variant_name="Night", tags=[], prompt="a station at night", negative_prompt="", new_variant=True,
        )
        self.assertEqual("Scenes", next(entry for entry in self.store.load() if entry.id == source.id).category)
        self.assertEqual(target.variant_group, night.variant_group)

    def test_overlapping_saves_from_separate_stores_preserve_both_entries(self):
        second_store = PromptWorkspaceStore(self.path)
        first_at_write = Event()
        release_first = Event()
        second_started = Event()
        second_read = Event()
        original_write = self.store._write
        original_read = second_store._read

        def delayed_write(entries, sections):
            first_at_write.set()
            if not release_first.wait(5):
                raise TimeoutError("First save was not released")
            original_write(entries, sections)

        def observed_read():
            result = original_read()
            second_read.set()
            return result

        def save_second():
            second_started.set()
            return second_store.upsert_entry(
                kind="snippet", name="Second", category="Weather", tags=[],
                prompt="rain", negative_prompt="",
            )

        with patch.object(self.store, "_write", side_effect=delayed_write), patch.object(second_store, "_read", side_effect=observed_read):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(
                    self.store.upsert_entry, kind="snippet", name="First", category="Weather",
                    tags=[], prompt="fog", negative_prompt="",
                )
                try:
                    self.assertTrue(first_at_write.wait(5))
                    second = pool.submit(save_second)
                    self.assertTrue(second_started.wait(5))
                    second_read.wait(0.1)
                finally:
                    release_first.set()
                first.result(timeout=5)
                second.result(timeout=5)

        self.assertEqual({"First", "Second"}, {entry.name for entry in self.store.load()})

    def test_upserts_prompt_variants_without_losing_their_group(self):
        base = self.store.upsert_entry(
            kind="prompt", name="Station", category="Scenes", variant_name="Base",
            tags=["travel"], prompt="a station", negative_prompt="",
        )
        fog = self.store.upsert_entry(
            kind="prompt", name="Station", category="Scenes", variant_name="Fog",
            tags=["night"], prompt="a foggy station", negative_prompt="",
        )
        updated = self.store.upsert_entry(
            kind="prompt", name="Station", category="Scenes", variant_name="fog",
            tags=["dawn"], prompt="fog at dawn", negative_prompt="",
        )
        self.assertEqual(base.variant_group, fog.variant_group)
        self.assertEqual(fog.id, updated.id)
        self.assertEqual(["dawn"], updated.tags)
        self.assertEqual({base.id, fog.id}, {entry.id for entry in self.store.variants(base.id)})
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.upsert_entry(
                kind="prompt", name="Station", category="Scenes", variant_name="FOG",
                tags=[], prompt="duplicate", negative_prompt="", new_variant=True,
            )
        with self.assertRaisesRegex(ValueError, "Save the prompt"):
            self.store.upsert_entry(
                kind="prompt", name="New", category="Scenes", variant_name="First",
                tags=[], prompt="new scene", negative_prompt="", new_variant=True,
            )
        self.assertEqual(2, len(self.store.load()))

    def test_legacy_name_and_variant_group_conflicts_do_not_overwrite_data(self):
        snippet = self.store.save_entry(
            None, kind="snippet", name="Fog", category="Weather", tags=[],
            prompt="first", negative_prompt="",
        )
        duplicate = replace(snippet, id="legacy-duplicate", prompt="second")
        self.store._write([snippet, duplicate], self.store.section_names())
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Multiple entries"):
            self.store.upsert_entry(
                kind="snippet", name="Fog", category="Weather", tags=[],
                prompt="replacement", negative_prompt="",
            )
        self.assertEqual(before, self.path.read_bytes())
        self.assertTrue(self.store.delete_entry(duplicate.id))
        self.assertEqual(snippet.id, self.store.upsert_entry(
            kind="snippet", name="Fog", category="Weather", tags=[],
            prompt="replacement", negative_prompt="",
        ).id)

        prompt = self.save_prompt(name="Legacy")
        separate_group = replace(prompt, id="legacy-group", variant_group="other", variant_name="Other")
        self.store._write([prompt, separate_group], self.store.section_names())
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Multiple prompts"):
            self.store.upsert_entry(
                kind="prompt", name="Legacy", category="Weather", variant_name="New",
                tags=[], prompt="replacement", negative_prompt="",
            )
        self.assertEqual(before, self.path.read_bytes())

        split_group = replace(prompt, id="split-group", name="Different", variant_name="Other")
        self.store._write([prompt, split_group], self.store.section_names())
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "different names or categories"):
            self.store.upsert_entry(
                kind="prompt", name="Legacy", category="Weather", variant_name="New",
                tags=[], prompt="replacement", negative_prompt="",
            )
        self.assertEqual(before, self.path.read_bytes())
        with self.assertRaisesRegex(ValueError, "different names or categories"):
            self.store.upsert_entry(
                selected_entry_id=prompt.id, kind="prompt", name="Legacy", category="New category",
                variant_name="Base", tags=[], prompt="replacement", negative_prompt="",
            )
        self.assertEqual(before, self.path.read_bytes())

        duplicate_id = replace(prompt, variant_name="Other")
        self.store._write([prompt, duplicate_id], self.store.section_names())
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "share an ID"):
            self.store.upsert_entry(
                selected_entry_id=prompt.id, kind="prompt", name="Legacy", category="New category",
                variant_name="Base", tags=[], prompt="replacement", negative_prompt="",
            )
        self.assertEqual(before, self.path.read_bytes())

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

    def test_quick_selection_resolves_current_references_for_snippets_and_prompts(self):
        body = self.save_snippet("Body", "tall", "body")
        character = self.save_snippet("Character", "Character, <body>", "character")
        scene = self.store.save_entry(
            None,
            kind="prompt",
            name="Scene",
            category="Scenes",
            tags=[],
            prompt="<<Subject>> walks",
            negative_prompt="blur",
            sections=[{"name": "Subject", "text": "<character>"}],
        )
        self.save_snippet("Body", "short", "body", entry_id=body.id)

        self.assertEqual(("snippet", "Character, short", ""), self.store.resolve_entry(character.id))
        self.assertEqual(("prompt", "Character, short walks", "blur"), self.store.resolve_entry(scene.id))
        self.assertEqual("Character, tall walks", self.store.load()[2].prompt)

    def test_quick_selection_rejects_missing_entry_or_unresolved_reference(self):
        snippet = self.save_snippet("Character", "<not_yet_defined>", "character")
        with self.assertRaisesRegex(ValueError, "Unknown reference"):
            self.store.resolve_entry(snippet.id)
        with self.assertRaisesRegex(ValueError, "no longer exists"):
            self.store.resolve_entry("missing")

    def test_quick_selection_can_insert_reference_or_resolved_text(self):
        self.save_snippet("Body", "tall", "body")
        snippet = self.store.save_entry(
            None,
            kind="snippet",
            name="Character",
            category="Characters",
            tags=[],
            prompt="Character, <body>",
            negative_prompt="avoid <body>",
            reference="character",
        )
        entry, prompt, negative = self.store.select_entry(snippet.id, as_reference=True)
        self.assertEqual(snippet, entry)
        self.assertEqual(("<character>", "avoid <body>"), (prompt, negative))

        _, prompt, negative = self.store.select_entry(snippet.id, as_reference=False)
        self.assertEqual(("Character, tall", "avoid tall"), (prompt, negative))

        unnamed = self.save_snippet("Without reference", "A <body> person")
        _, prompt, _ = self.store.select_entry(unnamed.id, as_reference=True)
        self.assertEqual("A tall person", prompt)

    def test_reference_selection_can_include_parts_defined_later(self):
        snippet = self.save_snippet("Character", "<not_yet_defined>", "character")
        _, prompt, _ = self.store.select_entry(snippet.id, as_reference=True)
        self.assertEqual("<character>", prompt)

    def test_generation_resolves_both_prompts_and_hires_without_changing_input(self):
        self.save_snippet("Body", "tall", "body")
        self.save_snippet("Character", "a <body> person", "character")
        processing = FakeProcessing(
            ["<character> walks <lora:style:0.8>", "A <body> person"],
            ["avoid <body>", ""],
            hr_prompt="<character> close-up",
            hr_negative_prompt="<body>",
        )
        install_prompt_resolver(processing, lambda: self.store)
        processing.setup_prompts()

        self.assertEqual(["a tall person walks <lora:style:0.8>", "A tall person"], processing.all_prompts)
        self.assertEqual(["avoid tall", ""], processing.all_negative_prompts)
        self.assertEqual(["a tall person close-up"], processing.all_hr_prompts)
        self.assertEqual(["tall"], processing.all_hr_negative_prompts)
        self.assertEqual(processing.all_prompts[0], processing.main_prompt)
        self.assertEqual(processing.all_negative_prompts[0], processing.main_negative_prompt)
        self.assertEqual("<character> walks <lora:style:0.8>", processing.prompt[0])

    def test_generation_errors_on_missing_references_without_partial_resolution(self):
        self.save_snippet("Body", "tall", "body")
        processing = FakeProcessing("<body>", "<missing>")
        install_prompt_resolver(processing, lambda: self.store)

        with self.assertRaisesRegex(ValueError, "Unknown reference <missing>"):
            processing.setup_prompts()

        self.assertEqual(["<body>"], processing.all_prompts)
        self.assertEqual(["<missing>"], processing.all_negative_prompts)

    def test_generation_resolver_handles_copied_and_reused_processing_objects(self):
        self.save_snippet("Body", "tall", "body")
        original = FakeProcessing("<body>")
        install_prompt_resolver(original, lambda: self.store)
        copied = copy(original)
        copied.prompt = "Copied <body>"
        install_prompt_resolver(copied, lambda: self.store)
        copied.setup_prompts()

        self.assertEqual(["Copied tall"], copied.all_prompts)
        self.assertEqual(0, original.setup_calls)
        install_prompt_resolver(original, lambda: self.store)
        original.setup_prompts()
        self.assertEqual(["tall"], original.all_prompts)
        self.assertEqual(1, original.setup_calls)

    def test_generation_without_references_does_not_read_library(self):
        processing = FakeProcessing("Plain text <lora:style:0.8>", "")

        def unavailable():
            raise AssertionError("Library should not be read")

        install_prompt_resolver(processing, unavailable)
        processing.setup_prompts()
        self.assertEqual(["Plain text <lora:style:0.8>"], processing.all_prompts)

    def test_generation_ignores_hires_prompts_when_hires_is_disabled(self):
        self.save_snippet("Body", "tall", "body")
        processing = FakeProcessing("<body>", hr_prompt="<missing>")
        processing.enable_hr = False
        install_prompt_resolver(processing, lambda: self.store)
        processing.setup_prompts()

        self.assertEqual(["tall"], processing.all_prompts)
        self.assertEqual(["<missing>"], processing.all_hr_prompts)

    def test_protects_references_that_are_in_use(self):
        body = self.save_snippet("Body", "tall", "body")
        self.save_snippet("Character A", "<body>", "character_a")

        with self.assertRaisesRegex(ValueError, "still used by: Character A"):
            self.store.delete_entry(body.id)
        with self.assertRaisesRegex(ValueError, "still used by: Character A"):
            self.save_snippet("Body", "tall", "renamed", entry_id=body.id)
        with self.assertRaisesRegex(ValueError, "already used"):
            self.save_snippet("Other", "x", "BODY")

    def test_protects_references_used_in_negative_prompts(self):
        body = self.save_snippet("Body", "tall", "body")
        self.store.save_entry(
            None,
            kind="snippet",
            name="Character",
            category="Characters",
            tags=[],
            prompt="a person",
            negative_prompt="avoid <body>",
        )

        with self.assertRaisesRegex(ValueError, "still used by: Character"):
            self.store.delete_entry(body.id)
        with self.assertRaisesRegex(ValueError, "still used by: Character"):
            self.save_snippet("Body", "tall", "renamed", entry_id=body.id)

    def test_renaming_reference_requires_updating_its_own_negative_prompt(self):
        body = self.store.save_entry(
            None,
            kind="snippet",
            name="Body",
            category="Characters",
            tags=[],
            prompt="tall",
            negative_prompt="<body>",
            reference="body",
        )

        with self.assertRaisesRegex(ValueError, "still used by: Body"):
            self.store.save_entry(
                body.id,
                kind="snippet",
                name="Body",
                category="Characters",
                tags=[],
                prompt="tall",
                negative_prompt="<body>",
                reference="renamed",
            )
        updated = self.store.save_entry(
            body.id,
            kind="snippet",
            name="Body",
            category="Characters",
            tags=[],
            prompt="tall",
            negative_prompt="<renamed>",
            reference="renamed",
        )
        self.assertTrue(self.store.delete_entry(updated.id))

    def test_saved_prompt_rejects_unknown_negative_references(self):
        with self.assertRaisesRegex(ValueError, "Unknown reference <missing>"):
            self.store.save_entry(
                None,
                kind="prompt",
                name="Scene",
                category="Scenes",
                tags=[],
                prompt="a person",
                negative_prompt="<missing>",
            )

        snippet = self.store.save_entry(
            None,
            kind="snippet",
            name="Character",
            category="Characters",
            tags=[],
            prompt="a person",
            negative_prompt="<missing>",
        )
        self.assertEqual("<missing>", snippet.negative_prompt)

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


class PromptWorkspaceBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gradio = ModuleType("gradio")
        gradio.update = lambda **kwargs: kwargs
        gradio.skip = lambda: {"__type__": "update"}
        gradio_context = ModuleType("gradio.context")
        gradio_context.get_blocks_context = lambda: None
        modules = ModuleType("modules")
        modules.__path__ = []
        modules.scripts = SimpleNamespace(Script=object, AlwaysVisible=object())
        modules.shared = SimpleNamespace()
        modules.script_callbacks = SimpleNamespace(
            on_ui_settings=lambda callback: None,
            on_ui_tabs=lambda callback: None,
            on_after_component=lambda callback: None,
        )
        ui_common = ModuleType("modules.ui_common")
        ui_common.refresh_symbol = ""
        ui_components = ModuleType("modules.ui_components")
        ui_components.ToolButton = object
        storyboard_ui = ModuleType("storyboard_ui")
        storyboard_ui.gallery_callback = lambda factory: lambda component, **kwargs: None
        storyboard_ui.storyboard_ui = lambda *args: None
        dependencies = {
            "gradio": gradio,
            "gradio.context": gradio_context,
            "modules": modules,
            "modules.ui_common": ui_common,
            "modules.ui_components": ui_components,
            "storyboard_ui": storyboard_ui,
        }
        spec = importlib.util.spec_from_file_location("prompt_workspace_browser_test", EXTENSION_DIR / "scripts" / "prompt_workspace.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load Prompt Workspace script")
        cls.browser = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, dependencies):
            spec.loader.exec_module(cls.browser)

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.store = PromptWorkspaceStore(Path(temp_dir.name) / "library.json")

    def test_generation_can_use_default_library_before_settings_are_registered(self):
        with patch.object(self.browser.shared, "opts", SimpleNamespace(), create=True):
            self.assertEqual(EXTENSION_DIR / "data" / "library.json", self.browser.workspace_store().path)

    def test_quick_browser_hook_ignores_runtime_copies_but_handles_ui_construction(self):
        for context, render, builds_ui in ((None, True, False), (object(), False, False), (object(), True, True)):
            for tab in ("txt2img", "img2img"):
                with (
                    self.subTest(context=context, render=render, tab=tab),
                    patch.object(self.browser, "get_blocks_context", return_value=context),
                    patch.object(self.browser, "workspace_store", return_value=self.store) as factory,
                    patch.object(self.browser, "quick_browser") as build,
                ):
                    self.browser.add_quick_browser(SimpleNamespace(elem_id=f"{tab}_neg_prompt_row"), render=render)
                    if builds_ui:
                        factory.assert_called_once_with()
                        build.assert_called_once_with(self.store, tab)
                    else:
                        factory.assert_not_called()
                        build.assert_not_called()

    def test_template_edits_only_update_changed_preview_components(self):
        names = json.dumps(["Subject", "Lighting"])
        skip = {"__type__": "update"}
        unchanged = self.browser.update_source(self.store, ["Subject"], "a cat", "", "prompt", "<<Subject>>", names, "a cat", "")
        self.assertEqual((skip, skip, skip), unchanged)

        changed = self.browser.update_source(self.store, ["Subject"], "a cat", "", "prompt", "<<Subject>> <<Lighting>>", names, "a cat", "soft light")
        self.assertEqual(({"value": ["Subject", "Lighting"]}, "a cat soft light", skip), changed)

    def test_source_preview_with_added_default_sections(self):
        updates = self.browser.apply_default_sections(self.store, json.dumps(["Subject"]), "a cat")
        texts = [update.get("value", "") for update in updates[2:-2]]
        result = self.browser.update_source(
            self.store, ["Subject"], "", "⚠️ Unknown section <<Lighting>>",
            "prompt", "<<Subject>> <<Lighting>>", updates[-2], *texts,
        )
        self.assertEqual(({"value": ["Subject", "Lighting"]}, "a cat", ""), result)

    def test_saving_a_changed_category_moves_the_selected_referenced_snippet(self):
        original = self.store.save_entry(
            None, kind="snippet", name="Fog", category="Weather", tags=[],
            prompt="fog", negative_prompt="", reference="fog",
        )
        dependent = self.store.save_entry(
            None, kind="snippet", name="Scene", category="Weather", tags=[],
            prompt="<fog> clouds", negative_prompt="",
        )
        result = self.browser.save_entry(
            self.store, original.id, "Weather", "", "snippet", "Fog", "", "Lighting",
            "mood", "fog", "", "thick fog", "", "[]",
        )
        moved = next(entry for entry in self.store.load() if entry.id == original.id)
        self.assertEqual(("Lighting", "fog", "thick fog"), (moved.category, moved.reference, moved.prompt))
        self.assertEqual(2, len(self.store.load()))
        self.assertEqual("thick fog clouds", self.store.select_entry(dependent.id)[1])
        self.assertEqual({"choices": [("Fog <fog> — Lighting", original.id)], "value": original.id}, result[0])
        self.assertEqual({"choices": ["All categories", "Lighting", "Weather"], "value": "Lighting"}, result[1])
        self.assertIn("Saved 'Fog <fog> — Lighting'", result[-1])

        before = self.store.path.read_bytes()
        failed = self.browser.save_entry(
            self.store, original.id, "Lighting", "", "snippet", "Fog", "", "Atmosphere",
            "mood", "mist", "", "thick fog", "", "[]",
        )
        self.assertIn("<fog> is still used by: Scene", failed[-1])
        self.assertEqual(before, self.store.path.read_bytes())

    def test_saving_a_changed_category_without_reference_does_not_create_a_copy(self):
        original = self.store.save_entry(
            None, kind="snippet", name="Fog", category="Weather", tags=[],
            prompt="fog", negative_prompt="",
        )
        result = self.browser.save_entry(
            self.store, original.id, "All categories", "", "snippet", "Fog", "", "Lighting",
            "", "", "", "fog", "", "[]",
        )
        self.assertEqual([(original.id, "Lighting")], [(entry.id, entry.category) for entry in self.store.load()])
        self.assertEqual("All categories", result[1]["value"])
        self.assertEqual(original.id, result[0]["value"])

    def test_saving_a_missing_selection_does_not_create_an_entry(self):
        result = self.browser.save_entry(
            self.store, "deleted", "All categories", "", "snippet", "Fog", "", "Lighting",
            "", "", "", "fog", "", "[]",
        )
        self.assertIn("selected prompt entry no longer exists", result[-1])
        self.assertFalse(self.store.path.exists())

    def test_deleting_without_a_selection_reports_the_problem(self):
        result = self.browser.delete_entry(self.store, None, "All categories", "")
        self.assertEqual("Select a library entry to delete", result[-1])
        self.assertTrue(all(value == {"__type__": "update"} for value in result[:-1]))

    def test_saving_a_copy_keeps_the_original_selected_for_deletion(self):
        original = self.store.save_entry(
            None, kind="snippet", name="Original", category="Weather", tags=[],
            prompt="old text", negative_prompt="",
        )
        result = self.browser.save_entry(
            self.store, original.id, "Lighting", "", "snippet", "Copy", "", "Lighting",
            "", "", "", "new text", "", "[]",
        )
        self.assertEqual(original.id, result[0]["value"])
        self.assertIn(("Original — Weather", original.id), result[0]["choices"])
        self.assertIn("Selected for deletion: 'Original — Weather'", result[-1])
        copied = next(entry for entry in self.store.load() if entry.name == "Copy")
        self.assertEqual("new text", copied.prompt)

        deleted = self.browser.delete_entry(self.store, original.id, "All categories", "")
        self.assertEqual(4, len(deleted))
        self.assertIn("Original", deleted[-1])
        self.assertEqual([copied], self.store.load())

    def test_reference_conflicts_and_used_reference_deletions_are_reported_inline(self):
        original = self.store.save_entry(
            None, kind="snippet", name="Character", category="People", tags=[],
            prompt="a person", negative_prompt="", reference="character",
        )
        before = self.store.load()
        result = self.browser.save_entry(
            self.store, original.id, "All categories", "", "snippet", "Copy", "", "People",
            "", "character", "", "a person", "", "[]",
        )
        self.assertTrue(all(value == {"__type__": "update"} for value in result[:-1]))
        self.assertIn("reference name <character> is already used", result[-1])
        self.assertEqual(before, self.store.load())

        self.store.save_entry(
            None, kind="snippet", name="Scene", category="People", tags=[],
            prompt="<character> walking", negative_prompt="",
        )
        result = self.browser.delete_entry(self.store, original.id, "All categories", "")
        self.assertTrue(all(value == {"__type__": "update"} for value in result[:-1]))
        self.assertIn("still used by: Scene", result[-1])
        self.assertEqual(2, len(self.store.load()))

    def test_saving_a_new_variant_uses_the_editor_values(self):
        original = self.store.upsert_entry(
            kind="prompt", name="Station", category="Scenes", variant_name="Base",
            tags=["travel"], prompt="a station", negative_prompt="",
        )
        result = self.browser.save_entry(
            self.store, original.id, "All categories", "", "prompt", "Station", "Night", "Scenes",
            "night, rain", "", "", "a station at night", "bright", "[]", new_variant=True,
        )
        self.assertEqual(original.id, result[0]["value"])
        variants = self.store.variants(original.id)
        night = next(entry for entry in variants if entry.variant_name == "Night")
        self.assertEqual(["night", "rain"], night.tags)
        self.assertEqual("a station at night", night.prompt)
        self.assertEqual("bright", night.negative_prompt)
        self.assertIn("Selected for deletion: 'Station · Base — Scenes'", result[-1])

    def test_unused_snippet_reference_can_be_deleted(self):
        snippet = self.store.save_entry(
            None, kind="snippet", name="Unused", category="People", tags=[],
            prompt="a person", negative_prompt="", reference="unused",
        )
        result = self.browser.delete_entry(self.store, snippet.id, "All categories", "")
        self.assertIn("Deleted 'Unused <unused> — People'", result[-1])
        self.assertEqual([], self.store.load())

    def test_grid_filters_and_clamps_pages(self):
        for index in range(23):
            self.store.save_entry(
                None,
                kind="snippet",
                name=f"Entry {index:02}",
                category="Weather" if index < 21 else "Characters",
                tags=["moon"] if index == 20 else [],
                prompt="clouds",
                negative_prompt="avoid fog" if index == 20 else "",
            )

        first = self.browser.refresh_quick_entries(self.store, "All categories", "", 0)
        second = self.browser.refresh_quick_entries(self.store, "All categories", "", 1)
        self.assertEqual((0, 20, {"interactive": False}, {"interactive": True}),
                         (first[1], len(json.loads(first[2])), first[4], first[5]))
        self.assertEqual((1, 3, {"interactive": True}, {"interactive": False}),
                         (second[1], len(json.loads(second[2])), second[4], second[5]))
        self.assertTrue(set(json.loads(first[2])).isdisjoint(json.loads(second[2])))

        filtered = self.browser.refresh_quick_entries(self.store, "Weather", "MOON", 99)
        self.assertEqual((0, 1, "Page 1 / 1 · 1 entries"),
                         (filtered[1], len(json.loads(filtered[2])), filtered[3]))
        self.assertEqual({"visible": False}, filtered[7])
        self.assertEqual(1, self.browser.quick_page(self.store, "Weather", "FOG", 0)[-1])

        categories, selected, choices, page, last_page, total = self.browser.quick_page(
            self.store, "Removed category", "", -1
        )
        self.assertIn("Weather", categories)
        self.assertEqual(("All categories", 0, 1, 23), (selected, page, last_page, total))
        self.assertEqual(20, len(choices))

    def test_selecting_grid_entries_and_installing_generation_resolver(self):
        snippet = self.store.save_entry(
            None,
            kind="snippet",
            name="Clouds",
            category="Weather",
            tags=[],
            reference="clouds",
            prompt="dark clouds",
            negative_prompt="",
        )
        selected = self.browser.select_quick_entry(self.store, 0, json.dumps([snippet.id]), True)
        self.assertEqual(("snippet", "<clouds>", "", "dark clouds", snippet.id, {"interactive": False}), selected)
        with self.assertRaisesRegex(ValueError, "page has changed"):
            self.browser.select_quick_entry(self.store, 1, json.dumps([snippet.id]), True)

        processing = FakeProcessing("<clouds>", "<missing>")
        with patch.object(self.browser, "workspace_store", return_value=self.store):
            self.browser.PromptWorkspaceGeneration().before_process(processing)
            with self.assertRaisesRegex(ValueError, "Unknown reference <missing>"):
                processing.setup_prompts()


if __name__ == "__main__":
    unittest.main()
