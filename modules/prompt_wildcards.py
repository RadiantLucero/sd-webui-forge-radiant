"""Dynamic Prompts syntax in X/Y/Z plot Prompt S/R values.

The Dynamic Prompts extension resolves wildcards only while each grid cell is processed. A wildcard in a Prompt S/R
value would therefore show as a placeholder label, and with combinatorial generation every cell would render all
combinations while the grid keeps only its first image. Resolving the values before the grid is built avoids both.
"""

import inspect
import re

# Combinatorial generation repeats unresolvable wildcards endlessly, so it always needs a limit.
MAX_COMBINATIONS = 1000


class PromptValues(list):
    """Resolved Prompt S/R values with the original search text, which may itself be a wildcard."""

    def __init__(self, values, search):
        super().__init__(values)
        self.search = search


def find_dynamic_prompts(p):
    """Return the Dynamic Prompts script of this generation and its settings by parameter name, or None."""
    for script in getattr(getattr(p, "scripts", None), "alwayson_scripts", None) or []:
        if getattr(script, "_wildcard_manager", None) is not None and script.title().startswith("Dynamic Prompts"):
            names = list(inspect.signature(script.process).parameters)[1:]
            return script, dict(zip(names, p.script_args[script.args_from : script.args_to]))
    return None


def create_resolver(p, opts):
    """Return a resolver that follows the Dynamic Prompts settings of this generation, or None while it is off."""
    found = find_dynamic_prompts(p)
    if found is None:
        return None
    script, settings = found
    # Jinja templates use their own syntax for wildcards, so values stay literal like before.
    if not settings.get("is_enabled") or settings.get("enable_jinja_templates"):
        return None
    seed = None if p.seed in (None, "", -1) else int(p.seed)
    return WildcardResolver(script._wildcard_manager, settings, opts, seed)


class WildcardResolver:
    def __init__(self, manager, settings, opts, seed):
        from dynamicprompts.generators import (
            CombinatorialPromptGenerator,
            RandomPromptGenerator,
        )
        from dynamicprompts.parser.parse import ParserConfig

        # Same wildcard and parser settings that Dynamic Prompts applies to the prompt itself.
        manager.dedup_wildcards = not getattr(opts, "dp_wildcard_manager_no_dedupe", False)
        manager.sort_wildcards = not getattr(opts, "dp_wildcard_manager_no_sort", False)
        manager.shuffle_wildcards = getattr(opts, "dp_wildcard_manager_shuffle", False)
        if getattr(opts, "dp_auto_purge_cache", False):
            manager.clear_cache()
        config = ParserConfig(
            variant_start=getattr(opts, "dp_parser_variant_start", "{"),
            variant_end=getattr(opts, "dp_parser_variant_end", "}"),
            wildcard_wrap=getattr(opts, "dp_parser_wildcard_wrap", "__"),
        )
        ignore_whitespace = getattr(opts, "dp_ignore_whitespace", False)
        unlink_seed = bool(settings.get("unlink_seed_from_prompt"))

        self.folder = manager.path
        self.markers = (config.wildcard_wrap, config.variant_start)
        self.unresolved = re.compile(f"{re.escape(config.wildcard_wrap)}(.+?){re.escape(config.wildcard_wrap)}")
        self.combinatorial = bool(settings.get("is_combinatorial"))
        self.limit = int(settings.get("max_generations") or 0) or MAX_COMBINATIONS
        self.combinations = CombinatorialPromptGenerator(manager, ignore_whitespace=ignore_whitespace, parser_config=config)
        self.random = RandomPromptGenerator(manager, seed=seed, unlink_seed_from_prompt=unlink_seed, ignore_whitespace=ignore_whitespace, parser_config=config)
        # Like Dynamic Prompts for a batch, every value draws with the next seed unless prompts are unlinked from seeds.
        self.seed = None if unlink_seed else seed
        self.draws = 0

    def resolve(self, value):
        """Return every combination of a value, or one random draw; values without syntax stay unchanged."""
        if not any(marker in value for marker in self.markers):
            return [value]
        if self.combinatorial:
            results = self.combinations.generate(value, self.limit)
        else:
            results = self.random.generate(value, 1, seeds=None if self.seed is None else [self.seed + self.draws])
            self.draws += 1
        if not results:
            raise RuntimeError(f'Prompt S/R: "{value}" produced no prompts (wildcard folder: {self.folder})')
        for result in results:
            if match := self.unresolved.search(result):
                raise RuntimeError(f'Prompt S/R could not resolve wildcard "{match.group(0)}" in "{value}" (wildcard folder: {self.folder})')
        return results

    def resolve_values(self, values):
        """Resolve all Prompt S/R values; the first value stays the search text, and its own column shows its resolved values."""
        resolved = [result for value in values for result in self.resolve(value)]
        return values if resolved == values else PromptValues(resolved, values[0])
