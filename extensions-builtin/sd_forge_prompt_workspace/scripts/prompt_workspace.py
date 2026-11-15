import json
from functools import partial
from pathlib import Path

import gradio as gr
from prompt_workspace_data import PromptEntry, PromptWorkspaceStore, library_path

from modules import script_callbacks, shared


def on_ui_settings():
    shared.opts.add_option(
        "prompt_workspace_directory",
        shared.OptionInfo(
            "",
            "Prompt Workspace library directory",
            section=("prompt_workspace", "Prompt Workspace"),
            category_id="ui",
        ).info("Leave empty to use the extension's data directory. Changing this does not move existing data.").needs_reload_ui(),
    )


def entry_label(entry: PromptEntry) -> str:
    label = entry.name
    if entry.kind == "prompt":
        label += f" · {entry.variant_name}"
    return f"{label} — {entry.category}"


def entry_choices(store, category="All categories", search=""):
    query = search.strip().casefold()
    entries = store.load()
    filtered = [
        entry
        for entry in entries
        if (category == "All categories" or entry.category == category)
        and (
            not query
            or query in entry.name.casefold()
            or query in entry.category.casefold()
            or query in entry.prompt.casefold()
            or query in entry.negative_prompt.casefold()
            or any(query in tag.casefold() for tag in entry.tags)
        )
    ]
    filtered.sort(key=lambda entry: (entry.category.casefold(), entry.name.casefold(), entry.variant_name.casefold()))
    return [(entry_label(entry), entry.id) for entry in filtered]


def category_choices(store):
    categories = sorted({entry.category for entry in store.load()}, key=str.casefold)
    return ["All categories", *categories]


def load_entry(store, entry_id):
    if not entry_id:
        return "snippet", "", gr.update(value="Default", visible=False), "General", "", "", "", gr.update(visible=False)

    entry = next((entry for entry in store.load() if entry.id == entry_id), None)
    if entry is None:
        raise ValueError("The selected prompt entry no longer exists")
    is_prompt = entry.kind == "prompt"
    return (
        entry.kind,
        entry.name,
        gr.update(value=entry.variant_name, visible=is_prompt),
        entry.category,
        ", ".join(entry.tags),
        entry.prompt,
        entry.negative_prompt,
        gr.update(visible=is_prompt),
    )


def refresh_entries(store, category, search, entry_id):
    choices = entry_choices(store, category, search)
    if entry_id and any(value == entry_id for _, value in choices):
        return gr.update(choices=choices, value=entry_id), *load_entry(store, entry_id), gr.skip()
    return gr.update(choices=choices, value=None), *load_entry(store, None), ""


def save_entry(store, entry_id, kind, name, variant_name, category, tags, prompt, negative_prompt):
    entry = store.save_entry(
        entry_id,
        kind=kind,
        name=name,
        variant_name=variant_name,
        category=category,
        tags=[tag.strip() for tag in tags.split(",") if tag.strip()],
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    return (
        gr.update(choices=entry_choices(store, entry.category, ""), value=entry.id),
        gr.update(choices=category_choices(store), value=entry.category),
        gr.update(value=""),
        f"Saved '{entry_label(entry)}'",
    )


def save_variant(store, entry_id, variant_name, prompt, negative_prompt):
    if not entry_id:
        raise ValueError("Select a saved prompt before creating a variant")
    entry = store.save_variant(
        entry_id,
        variant_name=variant_name,
        prompt=prompt,
        negative_prompt=negative_prompt,
    )
    return (
        gr.update(choices=entry_choices(store, entry.category, ""), value=entry.id),
        gr.update(value=entry.category),
        gr.update(value=""),
        f"Saved variant '{entry.variant_name}'",
    )


def delete_entry(store, entry_id, category_filter, search):
    if not entry_id:
        return (
            gr.skip(),
            gr.skip(),
            *([gr.skip()] * 8),
            "Deletion cancelled",
        )
    if not store.delete_entry(entry_id):
        raise ValueError("The selected prompt entry no longer exists")
    categories = category_choices(store)
    selected_category = category_filter if category_filter in categories else "All categories"
    return (
        gr.update(choices=entry_choices(store, category_filter, search), value=None),
        gr.update(choices=categories, value=selected_category),
        *load_entry(store, None),
        "Entry deleted",
    )


def select_variant(store, entry_id, direction):
    variants = store.variants(entry_id)
    if not variants:
        return gr.update()
    current_index = next(index for index, entry in enumerate(variants) if entry.id == entry_id)
    next_index = (current_index + direction) % len(variants)
    return gr.update(value=variants[next_index].id)


def apply_prompt_js(tabname):
    positive_id = json.dumps(f"{tabname}_prompt")
    negative_id = json.dumps(f"{tabname}_neg_prompt")
    switch_tab = "switch_to_txt2img" if tabname == "txt2img" else "switch_to_img2img"
    return f"""function(kind, prompt, negativePrompt) {{
        const app = gradioApp();
        const positive = app.getElementById({positive_id})?.querySelector("textarea");
        const negative = app.getElementById({negative_id})?.querySelector("textarea");
        if (!positive || !negative) throw new Error("Could not find the {tabname} prompt fields");

        function apply(target, text, append) {{
            const value = (text || "").trim();
            if (append && !value) return;
            const current = target.value.trim();
            target.value = append ? (current && value ? current + ", " + value : current || value) : value;
            updateInput(target);
        }}

        apply(positive, prompt, kind === "snippet");
        apply(negative, negativePrompt, kind === "snippet");
        {switch_tab}();
        return [];
    }}"""


def prompt_workspace_ui():
    extension_dir = Path(__file__).resolve().parent.parent
    store = PromptWorkspaceStore(library_path(extension_dir, shared.opts.prompt_workspace_directory))

    with gr.Blocks() as ui:
        gr.Markdown("Browse reusable text snippets and save named prompt versions.")
        gr.Textbox(
            label="Active library directory",
            value=str(store.path.parent),
            interactive=False,
            info="Change this in Settings > Prompt Workspace and reload the UI. Existing data is not moved.",
        )
        with gr.Row():
            category_filter = gr.Dropdown(
                label="Category",
                choices=category_choices(store),
                value="All categories",
                elem_id="prompt_workspace_category_filter",
            )
            search = gr.Textbox(label="Search", placeholder="Search names, tags, or prompt text", elem_id="prompt_workspace_search")
        with gr.Row():
            entries = gr.Dropdown(
                label="Library",
                choices=entry_choices(store),
                value=None,
                elem_id="prompt_workspace_entries",
                scale=5,
            )
            previous_variant = gr.Button("Previous variant", elem_id="prompt_workspace_previous_variant")
            next_variant = gr.Button("Next variant", elem_id="prompt_workspace_next_variant")
            new_entry = gr.Button("New")

        with gr.Row():
            kind = gr.Radio(choices=["snippet", "prompt"], value="snippet", label="Entry type")
            name = gr.Textbox(label="Name")
            variant_name = gr.Textbox(label="Variant name", value="Default", visible=False)
            category = gr.Textbox(label="Category", value="General")
            tags = gr.Textbox(label="Tags", placeholder="Comma-separated")

        prompt = gr.Textbox(label="Prompt", lines=5, elem_classes=["prompt"])
        negative_prompt = gr.Textbox(label="Negative prompt", lines=3, elem_classes=["prompt"])

        with gr.Row():
            save = gr.Button("Save", variant="primary")
            save_as_variant = gr.Button("Save as new variant", visible=False)
            delete = gr.Button("Delete")
            apply_txt2img = gr.Button("Apply to txt2img")
            apply_img2img = gr.Button("Apply to img2img")
        status = gr.Markdown()

        fields = [kind, name, variant_name, category, tags, prompt, negative_prompt]
        editor_outputs = [*fields, save_as_variant]

        entries.change(fn=partial(load_entry, store), inputs=entries, outputs=editor_outputs, show_progress=False)
        filter_outputs = [entries, *editor_outputs, status]
        filter_inputs = [category_filter, search, entries]
        category_filter.change(fn=partial(refresh_entries, store), inputs=filter_inputs, outputs=filter_outputs, show_progress=False)
        search.change(fn=partial(refresh_entries, store), inputs=filter_inputs, outputs=filter_outputs, show_progress=False)

        kind.change(
            fn=lambda value: (gr.update(visible=value == "prompt"), gr.update(visible=value == "prompt")),
            inputs=kind,
            outputs=[variant_name, save_as_variant],
            show_progress=False,
        )

        new_entry.click(
            fn=lambda: (
                gr.update(value=None),
                "snippet",
                "",
                gr.update(value="Default", visible=False),
                "General",
                "",
                "",
                "",
                gr.update(visible=False),
                "",
            ),
            outputs=[entries, *fields, save_as_variant, status],
            show_progress=False,
        )

        save.click(
            fn=partial(save_entry, store),
            inputs=[entries, kind, name, variant_name, category, tags, prompt, negative_prompt],
            outputs=[entries, category_filter, search, status],
            show_progress=False,
        )
        save_as_variant.click(
            fn=partial(save_variant, store),
            inputs=[entries, variant_name, prompt, negative_prompt],
            outputs=[entries, category_filter, search, status],
            show_progress=False,
        )
        delete.click(
            fn=partial(delete_entry, store),
            _js='function(id, category, search){return confirm("Delete this library entry?") ? [id, category, search] : [null, category, search];}',
            inputs=[entries, category_filter, search],
            outputs=[entries, category_filter, *editor_outputs, status],
            show_progress=False,
        )

        previous_variant.click(
            fn=lambda entry_id: select_variant(store, entry_id, -1),
            inputs=entries,
            outputs=entries,
            show_progress=False,
        ).then(fn=partial(load_entry, store), inputs=entries, outputs=editor_outputs, show_progress=False)
        next_variant.click(
            fn=lambda entry_id: select_variant(store, entry_id, 1),
            inputs=entries,
            outputs=entries,
            show_progress=False,
        ).then(fn=partial(load_entry, store), inputs=entries, outputs=editor_outputs, show_progress=False)

        apply_inputs = [kind, prompt, negative_prompt]
        apply_txt2img.click(fn=None, _js=apply_prompt_js("txt2img"), inputs=apply_inputs, show_progress=False)
        apply_img2img.click(fn=None, _js=apply_prompt_js("img2img"), inputs=apply_inputs, show_progress=False)

    return ui


def prompt_workspace_tab():
    return [(prompt_workspace_ui(), "Prompt Workspace", "prompt-workspace")]


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(prompt_workspace_tab)
