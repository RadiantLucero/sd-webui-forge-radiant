import weakref
from collections.abc import Callable

from prompt_workspace_data import (
    REFERENCE_PATTERN,
    PromptWorkspaceStore,
    reference_map,
    resolve_references,
)


def resolve_generation_prompts(processing, store_factory: Callable[[], PromptWorkspaceStore]) -> None:
    fields = ("all_prompts", "all_negative_prompts")
    if getattr(processing, "enable_hr", False):
        fields += ("all_hr_prompts", "all_hr_negative_prompts")
    prompts = {field: getattr(processing, field, None) for field in fields}
    prompts = {field: values for field, values in prompts.items() if values is not None}
    if not any(REFERENCE_PATTERN.search(text) for values in prompts.values() for text in values):
        return

    references = reference_map(store_factory().load())
    resolved = {field: [resolve_references(text, references) for text in values] for field, values in prompts.items()}
    for field, values in resolved.items():
        setattr(processing, field, values)
    processing.main_prompt = processing.all_prompts[0]
    processing.main_negative_prompt = processing.all_negative_prompts[0]


class PromptSetupResolver:
    def __init__(self, processing, original, store_factory: Callable[[], PromptWorkspaceStore]):
        self.processing = weakref.ref(processing)
        self.store_factory = store_factory
        if isinstance(original, PromptSetupResolver):
            self.original = original.original
            self.bind_processing = original.bind_processing
        elif getattr(original, "__self__", None) is processing:
            self.original = original.__func__
            self.bind_processing = True
        else:
            self.original = original
            self.bind_processing = False

    def __call__(self):
        processing = self.processing()
        if processing is None:
            raise RuntimeError("Prompt Workspace processing object no longer exists")
        if self.bind_processing:
            result = self.original(processing)
        else:
            result = self.original()
        resolve_generation_prompts(processing, self.store_factory)
        return result


def install_prompt_resolver(processing, store_factory: Callable[[], PromptWorkspaceStore]) -> None:
    # Resolve after styles but before generation scripts, and let validation errors abort processing.
    processing.setup_prompts = PromptSetupResolver(processing, processing.setup_prompts, store_factory)
