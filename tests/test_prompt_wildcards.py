import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

from modules.prompt_wildcards import PromptValues, create_resolver

SETTINGS = ("is_enabled", "is_combinatorial", "unlink_seed_from_prompt", "enable_jinja_templates", "max_generations")


class FakeDynamicPrompts:
    """Stands in for the Dynamic Prompts script; only the parameter names of process() matter."""

    def __init__(self, manager):
        self._wildcard_manager = manager
        self.args_from = 1
        self.args_to = 1 + len(SETTINGS)

    def title(self):
        return "Dynamic Prompts v2.17.1"

    def process(self, p, is_enabled, is_combinatorial, unlink_seed_from_prompt, enable_jinja_templates, max_generations):
        pass


@unittest.skipUnless(find_spec("dynamicprompts"), "dynamicprompts is not installed")
class PromptWildcardTest(unittest.TestCase):
    def setUp(self):
        from dynamicprompts.wildcards import WildcardManager

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        (root / "animals.txt").write_text("dog\nhorse, galloping\n# comment\n__colors__ bird\n")
        (root / "colors.txt").write_text("red\nblue\n")
        self.manager = WildcardManager(root)

    def resolver(self, seed=7, **settings):
        values = dict(is_enabled=True, is_combinatorial=False, unlink_seed_from_prompt=False, enable_jinja_templates=False, max_generations=0) | settings
        script = FakeDynamicPrompts(self.manager)
        p = SimpleNamespace(seed=seed, scripts=SimpleNamespace(alwayson_scripts=[script]), script_args=[None, *(values[name] for name in SETTINGS)])
        return create_resolver(p, SimpleNamespace())

    def test_combinatorial_expands_every_value_and_keeps_commas_together(self):
        values = self.resolver(is_combinatorial=True).resolve_values(["cat", "__animals__"])

        self.assertEqual(values, ["cat", "blue bird", "red bird", "dog", "horse, galloping"])
        self.assertEqual(values.search, "cat")

    def test_wildcard_search_text_is_replaced_by_its_values(self):
        values = self.resolver(is_combinatorial=True).resolve_values(["__colors__", "green"])

        self.assertEqual(values, ["blue", "red", "green"])
        self.assertEqual(values.search, "__colors__")

    def test_random_draw_is_repeatable_and_advances_per_value(self):
        first = self.resolver().resolve_values(["cat", "__animals__", "__animals__", "__animals__"])

        self.assertEqual(first, self.resolver().resolve_values(["cat", "__animals__", "__animals__", "__animals__"]))
        self.assertEqual(len(first), 4)
        self.assertGreater(len(set(first[1:])), 1)

    def test_values_without_syntax_stay_unchanged(self):
        values = ["cat", "<lora:style:0.8>", "[a|b]"]

        self.assertIs(self.resolver(is_combinatorial=True).resolve_values(values), values)

    def test_disabled_or_jinja_dynamic_prompts_leaves_values_literal(self):
        self.assertIsNone(self.resolver(is_enabled=False))
        self.assertIsNone(self.resolver(enable_jinja_templates=True))
        self.assertIsNone(create_resolver(SimpleNamespace(seed=1, scripts=None, script_args=[]), SimpleNamespace()))

    def test_missing_wildcard_fails_instead_of_looping(self):
        for combinatorial in (True, False):
            with self.subTest(combinatorial=combinatorial), self.assertRaisesRegex(RuntimeError, "__missing__"):
                self.resolver(is_combinatorial=combinatorial).resolve_values(["cat", "__missing__"])

    def test_max_generations_limits_combinations_per_value(self):
        values = self.resolver(is_combinatorial=True, max_generations=2).resolve_values(["cat", "__animals__"])

        self.assertIsInstance(values, PromptValues)
        self.assertEqual(values, ["cat", "blue bird", "red bird"])


if __name__ == "__main__":
    unittest.main()
