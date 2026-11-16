import json
from functools import partial
from pathlib import Path

import gradio as gr
from prompt_workspace_data import (
    MAX_SECTIONS,
    PromptEntry,
    PromptWorkspaceStore,
    library_path,
    used_section_names,
    validate_section_names,
)
from prompt_workspace_generation import install_prompt_resolver
from storyboard_ui import gallery_callback, storyboard_ui

from modules import script_callbacks, scripts, shared
from modules.ui_common import refresh_symbol
from modules.ui_components import ToolButton

QUICK_GRID_COLUMNS = 4
QUICK_GRID_ROWS = 5
QUICK_PAGE_SIZE = QUICK_GRID_COLUMNS * QUICK_GRID_ROWS


def workspace_store():
    extension_dir = Path(__file__).resolve().parent.parent
    directory = getattr(shared.opts, "prompt_workspace_directory", "")
    return PromptWorkspaceStore(library_path(extension_dir, directory))


class PromptWorkspaceGeneration(scripts.Script):
    create_group = False

    def title(self):
        return "Prompt Workspace"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def before_process(self, p):
        install_prompt_resolver(p, workspace_store)


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
    elif entry.reference:
        label += f" <{entry.reference}>"
    return f"{label} — {entry.category}"


def entry_choices(store, category="All categories", search=""):
    query = search.strip().casefold()
    filtered = [
        entry
        for entry in store.load()
        if (category == "All categories" or entry.category == category)
        and (
            not query
            or query in entry.name.casefold()
            or query in entry.category.casefold()
            or query in entry.reference.casefold()
            or query in entry.mentioned_text().casefold()
            or any(query in tag.casefold() for tag in entry.tags)
        )
    ]
    filtered.sort(key=lambda entry: (entry.category.casefold(), entry.name.casefold(), entry.variant_name.casefold()))
    return [(entry_label(entry), entry.id) for entry in filtered]


def category_choices(store):
    categories = sorted({entry.category for entry in store.load()}, key=str.casefold)
    return ["All categories", *categories]


def snippet_choices(store):
    snippets = [entry for entry in store.load() if entry.kind == "snippet"]
    snippets.sort(key=lambda entry: (entry.category.casefold(), entry.name.casefold()))
    return [(entry_label(entry), entry.id) for entry in snippets]


def make_sections(names_json, texts):
    return [{"name": name, "text": text} for name, text in zip(json.loads(names_json), texts)]


def preview_values(store, kind, source, names_json, *texts):
    try:
        return store.preview(kind, source, make_sections(names_json, texts)), ""
    except ValueError as error:
        return "", f"⚠️ {error}"


def section_box_updates(sections):
    return [
        gr.update(label=sections[index]["name"], value=sections[index]["text"], visible=True)
        if index < len(sections)
        else gr.update(label=f"Section {index + 1}", value="", visible=False)
        for index in range(MAX_SECTIONS)
    ]


def editor_values(store, entry: PromptEntry | None):
    defaults = store.section_names()
    kind = entry.kind if entry else "snippet"
    is_prompt = kind == "prompt"
    sections = entry.sections if entry and entry.sections is not None else [{"name": name, "text": ""} for name in defaults]
    names = [section["name"] for section in sections]
    names_json = json.dumps(names)
    source = entry.source_text() if entry else ""
    result, problem = preview_values(store, kind, source, names_json, *(section["text"] for section in sections))
    default_section = entry.section if entry and entry.section else ""
    section_options = ["", *defaults, *([default_section] if default_section and default_section not in defaults else [])]
    return (
        kind,
        entry.name if entry else "",
        gr.update(value=entry.variant_name if is_prompt else "Default", visible=is_prompt),
        entry.category if entry else "General",
        ", ".join(entry.tags) if entry else "",
        gr.update(value=entry.reference if entry else "", visible=not is_prompt),
        gr.update(choices=section_options, value=default_section, visible=not is_prompt),
        gr.update(choices=names, value=used_section_names(source, names), visible=is_prompt),
        gr.update(visible=is_prompt),
        gr.update(choices=names, value=names[0] if names else None),
        *section_box_updates(sections),
        source,
        result,
        problem,
        entry.negative_prompt if entry else "",
        gr.update(visible=is_prompt),
        names_json,
    )


def load_entry(store, entry_id):
    if not entry_id:
        return editor_values(store, None)
    entry = next((entry for entry in store.load() if entry.id == entry_id), None)
    if entry is None:
        raise ValueError("The selected prompt entry no longer exists")
    return editor_values(store, entry)


def refresh_entries(store, category, search, entry_id):
    choices = entry_choices(store, category, search)
    if entry_id and any(value == entry_id for _, value in choices):
        return gr.update(choices=choices, value=entry_id), *load_entry(store, entry_id), gr.skip()
    return gr.update(choices=choices, value=None), *load_entry(store, None), ""


def update_kind(store, kind, source, names_json, *texts):
    is_prompt = kind == "prompt"
    return (
        gr.update(visible=is_prompt),
        gr.update(visible=is_prompt),
        gr.update(visible=not is_prompt),
        gr.update(visible=not is_prompt),
        gr.update(visible=is_prompt),
        gr.update(visible=is_prompt),
        *preview_values(store, kind, source, names_json, *texts),
    )


def update_source(store, kind, source, names_json, *texts):
    return gr.update(value=used_section_names(source, json.loads(names_json))), *preview_values(store, kind, source, names_json, *texts)


def apply_default_sections(store, names_json, *texts):
    current = make_sections(names_json, texts)
    known = {section["name"].casefold() for section in current}
    by_name = {section["name"].casefold(): section for section in current}
    defaults = store.section_names()
    merged = [by_name.get(name.casefold(), {"name": name, "text": ""}) for name in defaults]
    merged += [section for section in current if section["name"].casefold() not in {name.casefold() for name in defaults} and section["text"].strip()]
    validate_section_names([section["name"] for section in merged])
    added = [section["name"] for section in merged if section["name"].casefold() not in known]
    names = [section["name"] for section in merged]
    return (
        gr.update(choices=names),
        gr.update(choices=names, value=names[0]),
        *section_box_updates(merged),
        json.dumps(names),
        f"Added sections: {', '.join(added)}" if added else "No new default sections",
    )


def save_default_sections(store, text):
    names = store.save_section_names(text.splitlines())
    return "\n".join(names), gr.update(choices=["", *names]), f"Saved {len(names)} default sections"


def select_snippet(store, names_json, snippet_id):
    snippet = next((entry for entry in store.load() if entry.id == snippet_id), None)
    names = json.loads(names_json)
    if snippet is None or snippet.section.casefold() not in {name.casefold() for name in names}:
        return gr.skip()
    return gr.update(value=next(name for name in names if name.casefold() == snippet.section.casefold()))


def insert_snippet(store, snippet_id, target, negative_prompt, names_json, *texts):
    snippet = next((entry for entry in store.load() if entry.id == snippet_id), None)
    if snippet is None:
        raise ValueError("Select a text snippet to insert")
    names = json.loads(names_json)
    if target not in names:
        raise ValueError("Select a section for the snippet")
    texts = list(texts)
    index = names.index(target)
    addition = f"<{snippet.reference}>" if snippet.reference else snippet.prompt
    texts[index] = " ".join(part for part in (texts[index].strip(), addition) if part)
    if snippet.negative_prompt:
        negative_prompt = ", ".join(part for part in (negative_prompt.strip(), snippet.negative_prompt) if part)
    return (*texts, negative_prompt)


def save_entry(store, entry_id, category_filter, search, kind, name, variant_name, category, tags, reference, section, source, negative_prompt, names_json, *texts, new_variant=False):
    try:
        entry = store.upsert_entry(
            kind=kind,
            name=name,
            variant_name=variant_name,
            category=category,
            tags=[tag.strip() for tag in tags.split(",") if tag.strip()],
            prompt=source,
            negative_prompt=negative_prompt,
            reference=reference,
            section=section or "",
            sections=make_sections(names_json, texts) if kind == "prompt" else None,
            new_variant=new_variant,
        )
    except (ValueError, OSError) as error:
        return gr.skip(), gr.skip(), gr.skip(), f"Could not save: {error}"

    try:
        selected = next((item for item in store.load() if item.id == entry_id), None)
        choices = entry_choices(store, category_filter, search)
        if selected and all(value != selected.id for _, value in choices):
            choices.append((entry_label(selected), selected.id))
        categories = category_choices(store)
        snippets = snippet_choices(store)
    except (ValueError, OSError) as error:
        return gr.skip(), gr.skip(), gr.skip(), f"Saved '{entry_label(entry)}', but could not refresh the library: {error}"

    status = f"Saved '{entry_label(entry)}'"
    if selected and selected.id != entry.id:
        status += f". Selected for deletion: '{entry_label(selected)}'"
    elif entry_id and not selected:
        status += ". The previous selection no longer exists"
    return (
        gr.update(choices=choices, value=selected.id if selected else None),
        gr.update(choices=categories),
        gr.update(choices=snippets),
        status,
    )


def delete_entry(store, entry_id, category_filter, search):
    if not entry_id:
        return gr.skip(), gr.skip(), gr.skip(), "Select a library entry to delete"
    try:
        entry = next((item for item in store.load() if item.id == entry_id), None)
        if entry is None or not store.delete_entry(entry_id):
            raise ValueError("The selected prompt entry no longer exists")
    except (ValueError, OSError) as error:
        return gr.skip(), gr.skip(), gr.skip(), f"Could not delete: {error}"

    try:
        categories = category_choices(store)
        selected_category = category_filter if category_filter in categories else "All categories"
        choices = entry_choices(store, selected_category, search)
        snippets = snippet_choices(store)
    except (ValueError, OSError) as error:
        return gr.update(value=None), gr.skip(), gr.skip(), f"Deleted '{entry_label(entry)}', but could not refresh the library: {error}"
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=categories, value=selected_category),
        gr.update(choices=snippets, value=None),
        f"Deleted '{entry_label(entry)}'",
    )


def select_variant(store, entry_id, direction):
    variants = store.variants(entry_id)
    if not variants:
        return gr.update()
    current_index = next(index for index, entry in enumerate(variants) if entry.id == entry_id)
    return gr.update(value=variants[(current_index + direction) % len(variants)].id)


def quick_page(store, category, search, page):
    categories = category_choices(store)
    selected_category = category if category in categories else "All categories"
    choices = entry_choices(store, selected_category, search)
    last_page = max(0, (len(choices) - 1) // QUICK_PAGE_SIZE)
    page = max(0, min(int(page or 0), last_page))
    start = page * QUICK_PAGE_SIZE
    return categories, selected_category, choices[start : start + QUICK_PAGE_SIZE], page, last_page, len(choices)


def refresh_quick_entries(store, category, search, page):
    categories, selected_category, choices, page, last_page, total = quick_page(store, category, search, page)
    return (
        gr.update(choices=categories, value=selected_category),
        page,
        json.dumps([entry_id for _, entry_id in choices]),
        f"Page {page + 1} / {last_page + 1} · {total} entries",
        gr.update(interactive=page > 0),
        gr.update(interactive=page < last_page),
        *(gr.update(visible=row * QUICK_GRID_COLUMNS < len(choices)) for row in range(QUICK_GRID_ROWS)),
        *(
            gr.update(value=choices[index][0], visible=True)
            if index < len(choices)
            else gr.update(value="", visible=False)
            for index in range(QUICK_PAGE_SIZE)
        ),
        "",
        "",
        gr.update(interactive=False),
        *("" for _ in range(3)),
    )


def select_quick_entry(store, index, entries_json, as_reference):
    entry_ids = json.loads(entries_json)
    if index >= len(entry_ids):
        raise ValueError("The library page has changed; refresh it and select the entry again")
    entry, prompt, negative = store.select_entry(entry_ids[index], as_reference=as_reference)
    preview = entry.prompt if entry.kind == "snippet" and as_reference and entry.reference else prompt
    if negative:
        preview += f"\n\nNegative prompt: {negative}"
    return entry.kind, prompt, negative, preview, entry.id, gr.update(interactive=entry.kind == "prompt")


def select_quick_prompt(store, entry_id):
    if not entry_id:
        raise ValueError("Select a saved prompt first")
    kind, prompt, negative = store.resolve_entry(entry_id)
    if kind != "prompt":
        raise ValueError("Select a saved prompt first")
    return kind, prompt, negative


def quick_apply_js(tabname, replace):
    tab_id = json.dumps(tabname)
    action = "true" if replace else "false"
    return f"""function(kind, prompt, negativePrompt) {{
        const replace = {action};
        if ((kind === "prompt") !== replace) return [""];
        const app = gradioApp();
        const positive = app.getElementById({tab_id} + "_prompt")?.querySelector("textarea");
        const negative = app.getElementById({tab_id} + "_neg_prompt")?.querySelector("textarea");
        if (!positive || !negative) return ["Could not find the prompt fields"];
        if (replace) {{
            positive.value = prompt;
            negative.value = negativePrompt;
            updateInput(positive);
            updateInput(negative);
        }} else {{
            if (prompt) {{
                const start = positive.selectionStart;
                const end = positive.selectionEnd;
                const before = positive.value.slice(0, start);
                const after = positive.value.slice(end);
                const left = before && !/\\s$/.test(before) ? " " : "";
                const right = after && !/^\\s/.test(after) ? " " : "";
                positive.setRangeText(left + prompt + right, start, end, "end");
                updateInput(positive);
            }}
            if (negativePrompt) {{
                negative.value = negative.value.trimEnd() + (negative.value.trim() ? ", " : "") + negativePrompt;
                updateInput(negative);
            }}
        }}
        return [""];
    }}"""


def quick_browser(store, tabname):
    categories, category_value, choices, page, last_page, total = quick_page(store, "All categories", "", 0)
    with gr.Accordion("Prompt Workspace", open=False, elem_id=f"{tabname}_prompt_workspace_browser", elem_classes=["prompt-workspace-ui"]):
        with gr.Row(equal_height=False):
            category = gr.Dropdown(label="Category", choices=categories, value=category_value)
            search = gr.Textbox(label="Search", placeholder="Search library")
            refresh = ToolButton(value=refresh_symbol, tooltip="Refresh library")
        as_reference = gr.Checkbox(
            label="Insert snippets as references",
            value=True,
            info="Snippets without a reference name are inserted as text.",
        )
        grid_rows = []
        grid_buttons = []
        for row in range(QUICK_GRID_ROWS):
            with gr.Row(visible=row * QUICK_GRID_COLUMNS < len(choices), equal_height=False) as grid_row:
                grid_rows.append(grid_row)
                for column in range(QUICK_GRID_COLUMNS):
                    index = row * QUICK_GRID_COLUMNS + column
                    grid_buttons.append(
                        gr.Button(
                            value=choices[index][0] if index < len(choices) else "",
                            visible=index < len(choices),
                            size="sm",
                            scale=0,
                            min_width=0,
                        )
                    )
        with gr.Row(equal_height=False, elem_classes=["prompt-workspace-actions"]):
            previous_page = gr.Button("Previous", size="sm", scale=0, min_width=0, interactive=False)
            page_status = gr.Markdown(f"Page {page + 1} / {last_page + 1} · {total} entries")
            next_page = gr.Button("Next", size="sm", scale=0, min_width=0, interactive=page < last_page)
        preview = gr.Textbox(label="Selected entry", lines=2, interactive=False)
        quick_status = gr.Markdown()
        apply_saved = gr.Button("Replace with selected prompt", size="sm", scale=0, min_width=0, interactive=False)
        page_index = gr.Number(value=0, precision=0, visible=False)
        grid_ids = gr.Textbox(value=json.dumps([entry_id for _, entry_id in choices]), visible=False)
        selected_id = gr.Textbox(visible=False)
        selected_kind = gr.Textbox(visible=False)
        selected_prompt = gr.Textbox(visible=False)
        selected_negative = gr.Textbox(visible=False)

        selected_values = [selected_kind, selected_prompt, selected_negative]
        refresh_inputs = [category, search]
        refresh_outputs = [
            category,
            page_index,
            grid_ids,
            page_status,
            previous_page,
            next_page,
            *grid_rows,
            *grid_buttons,
            preview,
            selected_id,
            apply_saved,
            *selected_values,
        ]
        for event in (category.input, search.input, refresh.click):
            event(
                fn=lambda category, search: refresh_quick_entries(store, category, search, 0),
                inputs=refresh_inputs,
                outputs=refresh_outputs,
                show_progress=False,
            )
        for button, direction in ((previous_page, -1), (next_page, 1)):
            button.click(
                fn=lambda category, search, page, direction=direction: refresh_quick_entries(store, category, search, page + direction),
                inputs=[*refresh_inputs, page_index],
                outputs=refresh_outputs,
                show_progress=False,
            )

        for index, button in enumerate(grid_buttons):
            button.click(
                fn=partial(select_quick_entry, store, index),
                inputs=[grid_ids, as_reference],
                outputs=[*selected_values, preview, selected_id, apply_saved],
                show_progress=False,
            ).success(fn=None, js=quick_apply_js(tabname, False), inputs=selected_values, outputs=quick_status, show_progress=False)
        apply_saved.click(
            fn=partial(select_quick_prompt, store),
            inputs=selected_id,
            outputs=selected_values,
            show_progress=False,
        ).success(fn=None, js=quick_apply_js(tabname, True), inputs=selected_values, outputs=quick_status, show_progress=False)


def add_quick_browser(component, **_kwargs):
    for tabname in ("txt2img", "img2img"):
        if component.elem_id == f"{tabname}_neg_prompt_row":
            quick_browser(workspace_store(), tabname)
            break


TOGGLE_SECTION_JS = r"""function(selected, template, namesJson) {
    const key = (name) => name.trim().toLowerCase();
    const names = new Set(JSON.parse(namesJson || "[]").map(key));
    const chosen = new Set(selected.map(key));
    const present = new Set([...template.matchAll(/<<([^<>]*)>>/g)].map((match) => key(match[1])));
    const added = selected.filter((name) => !present.has(key(name)));
    if (!added.length) {
        return [template.replace(/[ \t]*<<([^<>]*)>>/g, (match, name) => names.has(key(name)) && !chosen.has(key(name)) ? "" : match)];
    }
    const textarea = gradioApp().querySelector("#prompt_workspace_source textarea");
    const position = Math.min(textarea ? textarea.selectionStart : template.length, template.length);
    const before = template.slice(0, position);
    const after = template.slice(position);
    const placeholders = added.map((name) => "<<" + name + ">>").join(" ");
    return [before + (before && !/\s$/.test(before) ? " " : "") + placeholders + (after && !/^\s/.test(after) ? " " : "") + after];
}"""


def apply_prompt_js(tabname):
    positive_id = json.dumps(f"{tabname}_prompt")
    negative_id = json.dumps(f"{tabname}_neg_prompt")
    switch_tab = "switch_to_txt2img" if tabname == "txt2img" else "switch_to_img2img"
    return f"""function(kind, prompt, negativePrompt) {{
        const app = gradioApp();
        const positive = app.getElementById({positive_id})?.querySelector("textarea");
        const negative = app.getElementById({negative_id})?.querySelector("textarea");
        if (!positive || !negative) return ["Could not find the {tabname} prompt fields"];
        if (!(prompt || "").trim() && !(negativePrompt || "").trim()) {{
            return ["There is no resolved prompt to apply. Check the messages below the result."];
        }}

        function apply(target, text, append, separator) {{
            const value = (text || "").trim().replace(/\\s+/g, " ");
            if (append && !value) return;
            const current = target.value.trim();
            target.value = append ? (current && value ? current + separator + value : current || value) : value;
            updateInput(target);
        }}

        apply(positive, prompt, kind === "snippet", " ");
        apply(negative, negativePrompt, kind === "snippet", ", ");
        {switch_tab}();
        return [""];
    }}"""


def prompt_workspace_ui():
    store = workspace_store()

    with gr.Blocks() as ui, gr.Column(elem_classes=["prompt-workspace-ui"]):
        gr.Markdown("Browse reusable text snippets and save named prompt versions. Use `<reference>` to insert a snippet and `<<Section>>` to place a section in a prompt.")
        gr.Textbox(
            label="Active library directory",
            value=str(store.path.parent),
            interactive=False,
            info="Change this in Settings > Prompt Workspace and reload the UI. Existing data is not moved.",
        )
        with gr.Accordion("Default sections", open=False):
            default_sections = gr.Textbox(
                label=f"Sections for new prompts (one per line, at most {MAX_SECTIONS})",
                value="\n".join(store.section_names()),
                lines=MAX_SECTIONS,
            )
            save_sections = gr.Button("Save default sections", size="sm", scale=0, min_width=0)

        with gr.Row(equal_height=False):
            category_filter = gr.Dropdown(label="Category", choices=category_choices(store), value="All categories", elem_id="prompt_workspace_category_filter")
            search = gr.Textbox(label="Search", placeholder="Search names, references, tags, or prompt text", elem_id="prompt_workspace_search")
        entries = gr.Dropdown(label="Library", choices=entry_choices(store), value=None, elem_id="prompt_workspace_entries")
        with gr.Row(equal_height=False, elem_classes=["prompt-workspace-actions"]):
            previous_variant = gr.Button("Previous variant", size="sm", scale=0, min_width=0, elem_id="prompt_workspace_previous_variant")
            next_variant = gr.Button("Next variant", size="sm", scale=0, min_width=0, elem_id="prompt_workspace_next_variant")
            new_entry = gr.Button("New", size="sm", scale=0, min_width=0)

        kind = gr.Radio(choices=["snippet", "prompt"], value="snippet", label="Entry type")
        with gr.Row(equal_height=False):
            name = gr.Textbox(label="Name")
            variant_name = gr.Textbox(label="Variant name", value="Default", visible=False)
            category = gr.Textbox(label="Category", value="General")
        with gr.Row(equal_height=False):
            tags = gr.Textbox(label="Tags", placeholder="Comma-separated")
            reference = gr.Textbox(label="Reference name", placeholder="e.g. character_name", info="Use it elsewhere as <name>")
            snippet_section = gr.Dropdown(label="Default section", choices=["", *store.section_names()], value="")

        section_select = gr.CheckboxGroup(label="Sections in the template", choices=store.section_names(), value=[], visible=False)
        with gr.Column(visible=False) as sections_column:
            section_boxes = [gr.Textbox(label=f"Section {index + 1}", lines=2, visible=False) for index in range(MAX_SECTIONS)]
            with gr.Row(equal_height=False):
                snippet_picker = gr.Dropdown(label="Insert snippet", choices=snippet_choices(store), value=None, scale=3)
                snippet_target = gr.Dropdown(label="Into section", choices=store.section_names(), value=None, scale=2)
            with gr.Row(equal_height=False, elem_classes=["prompt-workspace-actions"]):
                insert = gr.Button("Insert", size="sm", scale=0, min_width=0)
                use_defaults = gr.Button("Add default sections", size="sm", scale=0, min_width=0)
        section_names = gr.Textbox(value=json.dumps(store.section_names()), visible=False)

        source = gr.Textbox(label="Text / template", lines=5, elem_id="prompt_workspace_source", elem_classes=["prompt"])
        result = gr.Textbox(label="Result", lines=3, interactive=False)
        problem = gr.Markdown()
        negative_prompt = gr.Textbox(label="Negative prompt", lines=3, elem_classes=["prompt"])

        with gr.Row(equal_height=False, elem_classes=["prompt-workspace-actions"]):
            save = gr.Button("Save", variant="primary", size="sm", scale=0, min_width=0)
            save_as_variant = gr.Button("Save as new variant", size="sm", scale=0, min_width=0, visible=False)
            delete = gr.Button("Delete selected entry", size="sm", scale=0, min_width=0)
            apply_txt2img = gr.Button("Apply to txt2img", size="sm", scale=0, min_width=0)
            apply_img2img = gr.Button("Apply to img2img", size="sm", scale=0, min_width=0)
        status = gr.Markdown()

        editor_outputs = [
            kind,
            name,
            variant_name,
            category,
            tags,
            reference,
            snippet_section,
            section_select,
            sections_column,
            snippet_target,
            *section_boxes,
            source,
            result,
            problem,
            negative_prompt,
            save_as_variant,
            section_names,
        ]
        preview_inputs = [kind, source, section_names, *section_boxes]

        entries.input(fn=partial(load_entry, store), inputs=entries, outputs=editor_outputs, show_progress=False)
        filter_outputs = [entries, *editor_outputs, status]
        filter_inputs = [category_filter, search, entries]
        category_filter.input(fn=partial(refresh_entries, store), inputs=filter_inputs, outputs=filter_outputs, show_progress=False)
        search.change(fn=partial(refresh_entries, store), inputs=filter_inputs, outputs=filter_outputs, show_progress=False)

        kind.input(
            fn=partial(update_kind, store),
            inputs=preview_inputs,
            outputs=[variant_name, save_as_variant, reference, snippet_section, section_select, sections_column, result, problem],
            show_progress=False,
        )
        source.change(fn=partial(update_source, store), inputs=preview_inputs, outputs=[section_select, result, problem], show_progress=False)
        for box in section_boxes:
            box.input(fn=partial(preview_values, store), inputs=preview_inputs, outputs=[result, problem], show_progress=False)
        section_select.input(fn=None, _js=TOGGLE_SECTION_JS, inputs=[section_select, source, section_names], outputs=source, show_progress=False)

        snippet_picker.input(fn=partial(select_snippet, store), inputs=[section_names, snippet_picker], outputs=snippet_target, show_progress=False)
        insert.click(
            fn=partial(insert_snippet, store),
            inputs=[snippet_picker, snippet_target, negative_prompt, section_names, *section_boxes],
            outputs=[*section_boxes, negative_prompt],
            show_progress=False,
        ).then(fn=partial(preview_values, store), inputs=preview_inputs, outputs=[result, problem], show_progress=False)
        use_defaults.click(
            fn=partial(apply_default_sections, store),
            inputs=[section_names, *section_boxes],
            outputs=[section_select, snippet_target, *section_boxes, section_names, status],
            show_progress=False,
        ).then(fn=partial(update_source, store), inputs=preview_inputs, outputs=[section_select, result, problem], show_progress=False)
        save_sections.click(
            fn=partial(save_default_sections, store),
            inputs=default_sections,
            outputs=[default_sections, snippet_section, status],
            show_progress=False,
        )

        new_entry.click(
            fn=lambda: (gr.update(value=None), *editor_values(store, None), ""),
            outputs=[entries, *editor_outputs, status],
            show_progress=False,
        )

        save_inputs = [entries, category_filter, search, kind, name, variant_name, category, tags, reference, snippet_section, source, negative_prompt, section_names, *section_boxes]
        save_outputs = [entries, category_filter, snippet_picker, status]
        save.click(
            fn=partial(save_entry, store),
            inputs=save_inputs,
            outputs=save_outputs,
            show_progress=False,
        )
        save_as_variant.click(
            fn=partial(save_entry, store, new_variant=True),
            inputs=save_inputs,
            outputs=save_outputs,
            show_progress=False,
        )
        delete.click(
            fn=partial(delete_entry, store),
            inputs=[entries, category_filter, search],
            outputs=[entries, category_filter, snippet_picker, status],
            show_progress=False,
        )

        for button, direction in ((previous_variant, -1), (next_variant, 1)):
            button.click(
                fn=partial(select_variant, store, direction=direction),
                inputs=entries,
                outputs=entries,
                show_progress=False,
            ).then(fn=partial(load_entry, store), inputs=entries, outputs=editor_outputs, show_progress=False)

        apply_inputs = [kind, result, negative_prompt]
        apply_txt2img.click(fn=None, _js=apply_prompt_js("txt2img"), inputs=apply_inputs, outputs=status, show_progress=False)
        apply_img2img.click(fn=None, _js=apply_prompt_js("img2img"), inputs=apply_inputs, outputs=status, show_progress=False)

    return ui


def prompt_workspace_tab():
    return [
        (prompt_workspace_ui(), "Prompt Workspace", "prompt-workspace"),
        (storyboard_ui(workspace_store(), apply_prompt_js), "Storyboard", "storyboard"),
    ]


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(prompt_workspace_tab)
script_callbacks.on_after_component(add_quick_browser)
script_callbacks.on_after_component(gallery_callback(workspace_store))
