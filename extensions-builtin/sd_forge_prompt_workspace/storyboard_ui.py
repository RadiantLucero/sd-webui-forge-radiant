import json
from functools import partial
from html import escape

import gradio as gr
from gradio.context import get_blocks_context
from prompt_workspace_data import MAX_SECTIONS
from storyboard_data import StoryboardStore, asset_reference


def project_choices(store):
    return [(project["name"], project["id"]) for project in store.list_projects()]


def scene_choices(project):
    return [(scene["name"], scene["id"]) for scene in project["scenes"]]


def asset_choices(project):
    return [(f"{asset['kind']}: {asset['name']}", asset["id"]) for asset in project["assets"]]


def variant_choices(scene):
    return [(variant["name"], variant["id"]) for variant in scene["variants"]]


def snippet_choices(library):
    return [(f"{entry.name} — {entry.category}", entry.id) for entry in library.load() if entry.kind == "snippet"]


def insert_snippet(library, entry_id, target):
    _, text, negative = library.select_entry(entry_id, as_reference=True)
    warning = "This snippet also has a negative prompt; copy it from the library if needed." if negative else ""
    return " ".join(part for part in (target.strip(), text) if part), warning


def empty_scene_values():
    return (
        gr.update(value="", interactive=False),
        *(gr.update(value="", visible=False, interactive=False) for _ in range(MAX_SECTIONS)),
        gr.update(value="", interactive=False),
        gr.update(choices=[], value=None), gr.update(choices=[], value=None),
        "", "", None, "",
    )


def get_scene(store, project_id, scene_id):
    project = store.load(project_id)
    return project, store._find(project["scenes"], scene_id, "Scene")


def selection_choices(store, project, scene):
    selections = []
    context = scene["draft"]["context"]
    assets = context["assets"] if context is not None else project["assets"]
    for selected in scene["draft"]["selections"]:
        asset = store._find(assets, selected["asset_id"], "Asset")
        name = asset["name"]
        if selected["outfit_id"]:
            name += " / " + store._find(asset["outfits"], selected["outfit_id"], "Outfit")["name"]
        selections.append((name, asset["id"]))
    return selections


def asset_palette(store, project, scene):
    names = {asset["id"]: asset["name"] for asset in store._draft_assets(project, scene["draft"])}
    chips = "".join(
        # A span instead of a button, because Firefox does not start drags from buttons.
        f'<span role="button" tabindex="0" draggable="true" class="storyboard-asset-chip" '
        f'data-asset-reference="{escape(asset_reference(names[asset_id]), quote=True)}" title="{escape(asset_reference(names[asset_id]), quote=True)}">'
        f"{escape(label)}</span>"
        for label, asset_id in selection_choices(store, project, scene)
    )
    if not chips:
        return "<p>Add assets to this scene in the Cast step first.</p>"
    return (
        "<p>Drag an asset into a section or the template at the desired text position, "
        "or click in a text field and then click the asset.</p>"
        f'<div class="storyboard-asset-chips">{chips}</div>'
        '<p class="storyboard-asset-feedback" role="status" aria-live="polite"></p>'
    )


def scene_values(store, project_id, scene_id):
    project, scene = get_scene(store, project_id, scene_id)
    scene_draft = scene["draft"]
    selections = selection_choices(store, project, scene)
    variants = variant_choices(scene)
    chosen = scene["selected_variant"]
    variant = store._find(scene["variants"], chosen, "Variant") if chosen else None
    return (
        gr.update(value=scene_draft["template"], interactive=True),
        *(
            gr.update(label=scene_draft["sections"][index]["name"], value=scene_draft["sections"][index]["text"], visible=True, interactive=True)
            if index < len(scene_draft["sections"]) else gr.update(value="", visible=False)
            for index in range(MAX_SECTIONS)
        ),
        gr.update(value=scene_draft["negative"], interactive=True),
        gr.update(choices=selections, value=None),
        gr.update(choices=variants, value=chosen),
        variant["prompt"] if variant else "",
        variant["negative"] if variant else "",
        variant["image"] if variant else None,
        asset_palette(store, project, scene),
    )


def project_values(store, project_id):
    project = store.load(project_id)
    return (
        project["image_directory"],
        gr.update(choices=scene_choices(project), value=None),
        gr.update(choices=asset_choices(project), value=None),
        *empty_scene_values(),
    )


def create_project(store, name, directory):
    project = store.create(name, directory)
    return gr.update(choices=project_choices(store), value=project["id"]), *project_values(store, project["id"])


def rename_project(store, project_id, name):
    store.rename(project_id, name)
    return gr.update(choices=project_choices(store), value=project_id)


def delete_project(store, project_id):
    if not project_id:
        return *((gr.skip(),) * (MAX_SECTIONS + 11)), "Select a storyboard to delete"
    store.delete(project_id)
    return (
        gr.update(choices=project_choices(store), value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        *empty_scene_values(),
        "Storyboard deleted",
    )


def create_scene(store, project_id, name):
    scene = store.add_scene(project_id, name)
    project = store.load(project_id)
    return gr.update(choices=scene_choices(project), value=scene["id"]), *scene_values(store, project_id, scene["id"])


def move_scene(store, project_id, scene_id, direction):
    store.move_scene(project_id, scene_id, direction)
    return gr.update(choices=scene_choices(store.load(project_id)), value=scene_id)


def rename_scene(store, project_id, scene_id, name):
    store.rename_scene(project_id, scene_id, name)
    return gr.update(choices=scene_choices(store.load(project_id)), value=scene_id)


def delete_scene(store, project_id, scene_id):
    if not project_id or not scene_id:
        return *((gr.skip(),) * (MAX_SECTIONS + 9)), "Select a storyboard and scene to delete"
    store.delete_scene(project_id, scene_id)
    return gr.update(choices=scene_choices(store.load(project_id)), value=None), *empty_scene_values(), "Scene deleted"


def update_text(store, field, project_id, scene_id, value, section_index=None):
    store.update_draft(project_id, scene_id, field, value, section_index)
    return "Draft saved"


def outfit_choices(asset):
    return [(outfit["name"], outfit["id"]) for outfit in asset["outfits"]]


def load_asset(store, project_id, asset_id):
    project = store.load(project_id)
    asset = store._find(project["assets"], asset_id, "Asset")
    return (
        asset["kind"], asset["name"], asset["text"],
        gr.update(choices=outfit_choices(asset), value=None),
        "", "",
    )


def save_asset(store, project_id, asset_id, kind, name, text, outfit_name, outfit_text):
    if not project_id:
        return gr.skip(), gr.skip(), "Select a storyboard first"
    asset, outfit = store.save_asset(project_id, asset_id, kind, name, text, outfit_name, outfit_text)
    return (
        gr.update(choices=asset_choices(store.load(project_id)), value=asset["id"]),
        gr.update(choices=outfit_choices(asset), value=outfit["id"] if outfit else None),
        "Asset and outfit saved" if outfit else "Asset saved",
    )


def delete_asset(store, project_id, scene_id, asset_id):
    if not project_id or not asset_id:
        return gr.skip(), gr.skip(), gr.skip(), "Select a project asset to delete"
    store.delete_asset(project_id, asset_id)
    project = store.load(project_id)
    selection = gr.update(choices=selection_choices(store, *get_scene(store, project_id, scene_id)), value=None) if scene_id else gr.skip()
    return gr.update(choices=asset_choices(project), value=None), gr.update(choices=[], value=None), selection, "Asset deleted"


def save_outfit(store, project_id, asset_id, name, text):
    if not project_id or not asset_id:
        return gr.skip(), "Select a character first"
    outfit = store.save_outfit(project_id, asset_id, name, text)
    asset = store._find(store.load(project_id)["assets"], asset_id, "Character")
    return gr.update(choices=outfit_choices(asset), value=outfit["id"]), "Outfit saved"


def load_outfit(store, project_id, asset_id, outfit_id):
    project = store.load(project_id)
    asset = store._find(project["assets"], asset_id, "Character")
    outfit = store._find(asset["outfits"], outfit_id, "Outfit")
    return outfit["name"], outfit["text"]


def cast_values(store, project_id, asset_id, outfit_id):
    if not project_id:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
    project = store.load(project_id)
    asset = next((item for item in project["assets"] if item["id"] == asset_id), None)
    outfits = outfit_choices(asset) if asset else []
    return (
        gr.update(choices=asset_choices(project), value=asset["id"] if asset else None),
        gr.update(choices=outfits, value=outfit_id if outfit_id in (value for _, value in outfits) else None),
    )


def cast_outfits(store, project_id, asset_id):
    asset = store._find(store.load(project_id)["assets"], asset_id, "Asset")
    return gr.update(choices=outfit_choices(asset), value=None)


def add_selection(store, project_id, scene_id, asset_id, outfit_id):
    if not project_id or not scene_id or not asset_id:
        return gr.skip(), "Select a scene and a project asset first"
    store.select_asset(project_id, scene_id, asset_id, outfit_id)
    return gr.update(choices=selection_choices(store, *get_scene(store, project_id, scene_id)), value=asset_id), "Added to scene"


def remove_selection(store, project_id, scene_id, asset_id):
    if not project_id or not scene_id or not asset_id:
        return *((gr.skip(),) * (MAX_SECTIONS + 8)), "Select a scene asset to remove"
    store.remove_asset(project_id, scene_id, asset_id)
    return *scene_values(store, project_id, scene_id), "Removed from scene"


def save_variant(store, project_id, scene_id, name):
    variant = store.save_variant(project_id, scene_id, name)
    _, scene = get_scene(store, project_id, scene_id)
    return gr.update(choices=variant_choices(scene), value=variant["id"]), variant["prompt"], variant["negative"], None


def load_variant(store, project_id, scene_id, variant_id):
    if not project_id or not scene_id or not variant_id:
        return *((gr.skip(),) * (MAX_SECTIONS + 8)), "Select a storyboard, scene and variant to load"
    store.select_variant(project_id, scene_id, variant_id)
    return *scene_values(store, project_id, scene_id), ""


def step_variant(store, project_id, scene_id, direction):
    if not project_id or not scene_id:
        return *((gr.skip(),) * (MAX_SECTIONS + 8)), "Select a storyboard and scene first"
    _, scene = get_scene(store, project_id, scene_id)
    variants = scene["variants"]
    if not variants:
        return *((gr.skip(),) * (MAX_SECTIONS + 8)), "This scene has no saved variants"
    current = next((index for index, item in enumerate(variants) if item["id"] == scene["selected_variant"]), 0)
    store.select_variant(project_id, scene_id, variants[(current + direction) % len(variants)]["id"])
    return *scene_values(store, project_id, scene_id), ""


def delete_variant(store, project_id, scene_id, variant_id):
    if not project_id or not scene_id or not variant_id:
        return *((gr.skip(),) * 4), "Select a storyboard, scene and variant to delete"
    store.delete_variant(project_id, scene_id, variant_id)
    _, scene = get_scene(store, project_id, scene_id)
    return gr.update(choices=variant_choices(scene), value=None), "", "", None, "Variant deleted"


def variant_prompt(store, project_id, scene_id, variant_id):
    _, scene = get_scene(store, project_id, scene_id)
    if scene["selected_variant"] != variant_id:
        raise ValueError("Load and select this variant before applying it")
    variant = store._find(scene["variants"], variant_id, "Variant")
    return "prompt", variant["prompt"], variant["negative"]


def preview_draft(store, project_id, scene_id):
    project, scene = get_scene(store, project_id, scene_id)
    return store.resolve(project, scene["draft"])


def prompt_step_values(store, project_id, scene_id):
    if not project_id or not scene_id:
        return "", "", ""
    project, scene = get_scene(store, project_id, scene_id)
    return asset_palette(store, project, scene), *store.resolve(project, scene["draft"])


def storyboard_ui(library, apply_js):
    store = StoryboardStore(library.path.parent / "storyboards", library)
    with gr.Blocks(analytics_enabled=False) as ui, gr.Column(elem_classes=["prompt-workspace-ui"]):
        project = gr.Dropdown(label="Storyboard", choices=project_choices(store))
        with gr.Tabs():
            with gr.Tab("1 · Storyboard"):
                gr.Markdown("Create a storyboard, then define its assets and scenes in the next steps.")
                project_name = gr.Textbox(label="Storyboard name")
                with gr.Row(equal_height=False, elem_classes=["prompt-workspace-actions"]):
                    create = gr.Button("Create storyboard", size="sm", scale=0, min_width=0)
                    rename = gr.Button("Rename storyboard", size="sm", scale=0, min_width=0)
                    delete = gr.Button("Delete storyboard", size="sm", scale=0, min_width=0)
                directory = gr.Textbox(label="Storyboard image directory", value=str(library.path.parent / "storyboard-images"))
                with gr.Row(equal_height=False, elem_classes=["prompt-workspace-actions"]):
                    set_directory = gr.Button("Save image directory", size="sm", scale=0, min_width=0)
                    publish = gr.Button("Publish storyboard", size="sm", scale=0, min_width=0)
                project_status = gr.Markdown()
            with gr.Tab("2 · Assets"):
                with gr.Row(equal_height=False):
                    asset = gr.Dropdown(label="Project asset")
                    asset_kind = gr.Dropdown(label="Type", choices=["Character", "Location", "Object"], value="Character")
                    asset_name = gr.Textbox(label="Name")
                asset_text = gr.Textbox(label="Base description (library references are supported)", lines=2)
                with gr.Row(equal_height=False):
                    outfit = gr.Dropdown(label="Outfit")
                    outfit_name = gr.Textbox(label="Outfit name")
                outfit_text = gr.Textbox(label="Outfit description")
                with gr.Row(equal_height=False, elem_classes=["prompt-workspace-actions"]):
                    save_asset_button = gr.Button("Save asset", size="sm", scale=0, min_width=0)
                    save_outfit_button = gr.Button("Save outfit", size="sm", scale=0, min_width=0)
                    new_outfit = gr.Button("New outfit", size="sm", scale=0, min_width=0)
                    delete_asset_button = gr.Button("Delete asset", size="sm", scale=0, min_width=0)
                asset_status = gr.Markdown()
                with gr.Accordion("Insert library snippet", open=False):
                    snippet = gr.Dropdown(label="Library snippet", choices=snippet_choices(library))
                    with gr.Row(equal_height=False, elem_classes=["prompt-workspace-actions"]):
                        refresh_snippets = gr.Button("Refresh library", size="sm", scale=0, min_width=0)
                        snippet_to_asset = gr.Button("Insert into base description", size="sm", scale=0, min_width=0)
                        snippet_to_outfit = gr.Button("Insert into outfit", size="sm", scale=0, min_width=0)
            with gr.Tab("3 · Scenes") as scenes_tab:
                scene = gr.Dropdown(label="Scene")
                with gr.Accordion("Add, rename, reorder or delete scenes", open=False):
                    scene_name = gr.Textbox(label="Scene name")
                    with gr.Row(equal_height=False, elem_classes=["prompt-workspace-actions"]):
                        create_scene_button = gr.Button("Add scene", size="sm", scale=0, min_width=0)
                        rename_scene_button = gr.Button("Rename scene", size="sm", scale=0, min_width=0)
                        move_up = gr.Button("Move earlier", size="sm", scale=0, min_width=0)
                        move_down = gr.Button("Move later", size="sm", scale=0, min_width=0)
                        delete_scene_button = gr.Button("Delete scene", size="sm", scale=0, min_width=0)
                    scene_status = gr.Markdown()
                with gr.Tabs():
                    with gr.Tab("a · Cast") as cast_tab:
                        with gr.Row(equal_height=False):
                            cast_asset = gr.Dropdown(label="Project asset")
                            cast_outfit = gr.Dropdown(label="Outfit")
                        add_selected_asset = gr.Button("Add to scene", size="sm", scale=0, min_width=0)
                        selection = gr.Dropdown(label="Assets in scene")
                        remove_selected_asset = gr.Button("Remove from scene", size="sm", scale=0, min_width=0)
                        cast_status = gr.Markdown()
                    with gr.Tab("b · Prompt & image") as prompt_tab:
                        palette = gr.HTML(elem_id="storyboard_asset_palette")
                        template = gr.Textbox(label="Scene template", lines=3, interactive=False, info="Place sections using <<Section>> and write transitions freely.", elem_classes=["storyboard-prompt-source"])
                        with gr.Row(equal_height=False):
                            sections = [gr.Textbox(label=f"Section {index + 1}", lines=2, visible=False, interactive=False, elem_classes=["storyboard-prompt-source"]) for index in range(5)]
                        with gr.Row(equal_height=False):
                            sections += [gr.Textbox(label=f"Section {index + 1}", lines=2, visible=False, interactive=False, elem_classes=["storyboard-prompt-source"]) for index in range(5, MAX_SECTIONS)]
                        negative = gr.Textbox(label="Negative prompt", lines=2, interactive=False, elem_classes=["prompt-workspace-negative"])
                        draft_status = gr.Markdown()
                        with gr.Row(equal_height=False):
                            variant = gr.Dropdown(label="Scene variant")
                            with gr.Row(equal_height=False, elem_classes=["prompt-workspace-actions"]):
                                previous_variant = gr.Button("Previous", size="sm", scale=0, min_width=0)
                                next_variant = gr.Button("Next", size="sm", scale=0, min_width=0)
                                load = gr.Button("Load and select variant", size="sm", scale=0, min_width=0)
                                delete_variant_button = gr.Button("Delete variant", size="sm", scale=0, min_width=0)
                        with gr.Row(equal_height=False):
                            variant_name = gr.Textbox(label="New variant name")
                            with gr.Row(equal_height=False, elem_classes=["prompt-workspace-actions"]):
                                save = gr.Button("Save draft as new variant", size="sm", scale=0, min_width=0)
                                preview = gr.Button("Preview draft", size="sm", scale=0, min_width=0)
                                refresh_context = gr.Button("Use current library and project assets", size="sm", scale=0, min_width=0)
                        with gr.Row(equal_height=False):
                            with gr.Column():
                                prompt = gr.Textbox(label="Resolved prompt", lines=4, interactive=False)
                                negative_result = gr.Textbox(label="Resolved negative prompt", lines=2, interactive=False, elem_classes=["prompt-workspace-negative"])
                                with gr.Row(equal_height=False, elem_classes=["prompt-workspace-actions"]):
                                    apply_txt = gr.Button("Apply selected variant to txt2img", size="sm", scale=0, min_width=0)
                                    apply_img = gr.Button("Apply selected variant to img2img", size="sm", scale=0, min_width=0)
                                    refresh_scene = gr.Button("Refresh scene", size="sm", scale=0, min_width=0)
                                status = gr.Markdown()
                            with gr.Column():
                                image = gr.Image(label="Selected variant image", interactive=False, type="filepath", height=320)
        kind = gr.Textbox(value="prompt", visible=False)
        selected_prompt = gr.Textbox(visible=False)
        selected_negative = gr.Textbox(visible=False)

        scene_outputs = [template, *sections, negative, selection, variant, prompt, negative_result, image, palette]
        common = {"concurrency_id": "storyboard", "show_progress": "hidden"}
        create.click(partial(create_project, store), [project_name, directory], [project, directory, scene, asset, *scene_outputs], **common)
        project.input(partial(project_values, store), project, [directory, scene, asset, *scene_outputs], **common)
        project.input(lambda project_id: store.load(project_id)["name"], project, project_name, **common)
        project.input(partial(cast_values, store), [project, cast_asset, cast_outfit], [cast_asset, cast_outfit], **common)
        for tab in (scenes_tab, cast_tab):
            tab.select(partial(cast_values, store), [project, cast_asset, cast_outfit], [cast_asset, cast_outfit], **common)
        prompt_tab.select(partial(prompt_step_values, store), [project, scene], [palette, prompt, negative_result], **common)
        rename.click(partial(rename_project, store), [project, project_name], project, **common)
        delete.click(
            partial(delete_project, store), project, [project, scene, asset, *scene_outputs, project_status],
            **common,
        ).then(partial(cast_values, store), [project, cast_asset, cast_outfit], [cast_asset, cast_outfit], **common)
        set_directory.click(lambda project_id, path: (store.set_image_directory(project_id, path), "Image directory saved")[1], [project, directory], project_status, **common)
        create_scene_button.click(partial(create_scene, store), [project, scene_name], [scene, *scene_outputs], **common)
        scene.input(partial(scene_values, store), [project, scene], scene_outputs, **common)
        refresh_scene.click(partial(scene_values, store), [project, scene], scene_outputs, **common)
        scene.input(lambda project_id, scene_id: get_scene(store, project_id, scene_id)[1]["name"], [project, scene], scene_name, **common)
        rename_scene_button.click(partial(rename_scene, store), [project, scene, scene_name], scene, **common)
        move_up.click(partial(move_scene, store, direction=-1), [project, scene], scene, **common)
        move_down.click(partial(move_scene, store, direction=1), [project, scene], scene, **common)
        delete_scene_button.click(
            partial(delete_scene, store), [project, scene], [scene, *scene_outputs, scene_status],
            **common,
        )
        asset.input(partial(load_asset, store), [project, asset], [asset_kind, asset_name, asset_text, outfit, outfit_name, outfit_text], **common)
        save_asset_button.click(
            partial(save_asset, store), [project, asset, asset_kind, asset_name, asset_text, outfit_name, outfit_text],
            [asset, outfit, asset_status], **common,
        )
        delete_asset_button.click(
            partial(delete_asset, store), [project, scene, asset], [asset, outfit, selection, asset_status], **common,
        ).then(
            lambda project_id, scene_id: scene_values(store, project_id, scene_id) if project_id and scene_id else (gr.skip(),) * len(scene_outputs),
            [project, scene], scene_outputs, **common,
        ).then(partial(cast_values, store), [project, cast_asset, cast_outfit], [cast_asset, cast_outfit], **common)
        refresh_snippets.click(lambda: gr.update(choices=snippet_choices(library)), outputs=snippet, **common)
        snippet_to_asset.click(partial(insert_snippet, library), [snippet, asset_text], [asset_text, asset_status], **common)
        snippet_to_outfit.click(partial(insert_snippet, library), [snippet, outfit_text], [outfit_text, asset_status], **common)
        outfit.input(partial(load_outfit, store), [project, asset, outfit], [outfit_name, outfit_text], **common)
        save_outfit_button.click(partial(save_outfit, store), [project, asset, outfit_name, outfit_text], [outfit, asset_status], **common)
        new_outfit.click(lambda: (gr.update(value=None), "", ""), outputs=[outfit, outfit_name, outfit_text], **common)
        cast_asset.input(partial(cast_outfits, store), [project, cast_asset], cast_outfit, **common)
        add_selected_asset.click(partial(add_selection, store), [project, scene, cast_asset, cast_outfit], [selection, cast_status], **common)
        remove_selected_asset.click(partial(remove_selection, store), [project, scene, selection], [*scene_outputs, cast_status], **common)
        for field, component in (("template", template), ("negative", negative)):
            component.input(partial(update_text, store, field), [project, scene, component], draft_status, trigger_mode="always_last", **common)
        for index, component in enumerate(sections):
            component.input(partial(update_text, store, "section", section_index=index), [project, scene, component], draft_status, trigger_mode="always_last", **common)
        preview.click(partial(preview_draft, store), [project, scene], [prompt, negative_result], **common)
        refresh_context.click(
            lambda project_id, scene_id: (store.refresh_draft_context(project_id, scene_id), "Draft now uses current definitions")[1],
            [project, scene], draft_status, **common,
        )
        save.click(partial(save_variant, store), [project, scene, variant_name], [variant, prompt, negative_result, image], **common)
        load.click(
            partial(load_variant, store), [project, scene, variant], [*scene_outputs, draft_status],
            **common,
        )
        for button, direction in ((previous_variant, -1), (next_variant, 1)):
            button.click(
                partial(step_variant, store, direction=direction), [project, scene], [*scene_outputs, draft_status],
                **common,
            )
        delete_variant_button.click(
            partial(delete_variant, store),
            [project, scene, variant], [variant, prompt, negative_result, image, draft_status],
            **common,
        )
        for button, tab in ((apply_txt, "txt2img"), (apply_img, "img2img")):
            button.click(
                partial(variant_prompt, store), [project, scene, variant],
                [kind, selected_prompt, selected_negative], **common,
            ).success(
                fn=None,
                js=apply_js(tab),
                inputs=[kind, selected_prompt, selected_negative], outputs=status, show_progress="hidden",
            )
        publish.click(lambda project_id: f"Published to: {store.publish(project_id)}", project, project_status, **common)
    return ui


def gallery_project(store, project_id):
    project = store.load(project_id)
    return gr.update(choices=scene_choices(project), value=None), gr.update(choices=[], value=None)


def gallery_scene(store, project_id, scene_id):
    _, scene = get_scene(store, project_id, scene_id)
    return gr.update(choices=variant_choices(scene), value=scene["selected_variant"])


def refresh_gallery(store, project_id, scene_id):
    projects = store.list_projects()
    project = next((item for item in projects if item["id"] == project_id), None)
    if project is None:
        return gr.update(choices=project_choices(store), value=None), gr.update(choices=[], value=None), gr.update(choices=[], value=None)
    scene = next((item for item in project["scenes"] if item["id"] == scene_id), None)
    return (
        gr.update(choices=project_choices(store), value=project_id),
        gr.update(choices=scene_choices(project), value=scene_id if scene else None),
        gr.update(choices=variant_choices(scene), value=scene["selected_variant"]) if scene else gr.update(choices=[], value=None),
    )


def take_image(store, project_id, scene_id, variant_id, gallery, generation_info, index):
    if not gallery:
        raise ValueError("Generate an image before assigning it to a storyboard")
    if index is None or index < 0:
        if len(gallery) != 1:
            raise ValueError("Select an image in the gallery")
        index = 0
    if index >= len(gallery):
        raise ValueError("The selected gallery image no longer exists")
    if not generation_info:
        raise ValueError("Generation parameters for the selected image are unavailable")
    try:
        info = json.loads(generation_info)
    except json.JSONDecodeError as error:
        raise ValueError("Invalid generation information for the selected image") from error
    infotexts = info.get("infotexts", [])
    if index >= len(infotexts):
        raise ValueError("Generation parameters for the selected image are unavailable")
    image = gallery[index][0]
    path = store.assign_image(project_id, scene_id, variant_id, image, infotexts[index])
    return f"Image assigned: {path}"


def gallery_callback(library_factory):
    panels = {}

    def after_component(component, **_kwargs):
        if _kwargs.get("render", True) is False or get_blocks_context() is None:
            return
        for tab in ("txt2img", "img2img"):
            if component.elem_id == f"{tab}_gallery":
                library = library_factory()
                store = StoryboardStore(library.path.parent / "storyboards", library)
                with gr.Accordion("Assign image to storyboard", open=False, elem_classes=["prompt-workspace-ui"]):
                    with gr.Row(equal_height=False):
                        project = gr.Dropdown(label="Storyboard", choices=project_choices(store))
                        scene = gr.Dropdown(label="Scene")
                        variant = gr.Dropdown(label="Variant")
                    with gr.Row(equal_height=False, elem_classes=["prompt-workspace-actions"]):
                        refresh = gr.Button("Refresh", size="sm", scale=0, min_width=0)
                        take = gr.Button("Use selected image", size="sm", scale=0, min_width=0)
                    status = gr.Markdown()
                    index = gr.Number(visible=False, value=-1)
                    project.input(partial(gallery_project, store), project, [scene, variant], show_progress="hidden")
                    scene.input(partial(gallery_scene, store), [project, scene], variant, show_progress="hidden")
                    refresh.click(partial(refresh_gallery, store), [project, scene], [project, scene, variant], show_progress="hidden")
                panels[tab] = (store, component, project, scene, variant, take, status, index)
            elif component.elem_id == f"generation_info_{tab}" and tab in panels:
                store, gallery, project, scene, variant, take, status, index = panels[tab]
                take.click(
                    partial(take_image, store),
                    inputs=[project, scene, variant, gallery, component, index],
                    outputs=status,
                    _js=f'(...args) => [...args.slice(0, 5), selected_gallery_index_id("{tab}_gallery")]',
                    show_progress="hidden",
                    concurrency_id="storyboard",
                )
    return after_component
