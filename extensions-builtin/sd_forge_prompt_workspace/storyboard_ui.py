import json
from functools import partial

import gradio as gr
from prompt_workspace_data import MAX_SECTIONS
from storyboard_data import StoryboardStore


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
    warning = f"Add this snippet's negative text to the scene manually: {negative}" if negative else ""
    return " ".join(part for part in (target.strip(), text) if part), warning


def empty_scene_values():
    return (
        gr.update(value="", interactive=False),
        *(gr.update(value="", visible=False, interactive=False) for _ in range(MAX_SECTIONS)),
        gr.update(value="", interactive=False),
        gr.update(choices=[], value=None), gr.update(choices=[], value=None),
        "", "", None,
    )


def get_scene(store, project_id, scene_id):
    project = store.load(project_id)
    return project, store._find(project["scenes"], scene_id, "Scene")


def scene_values(store, project_id, scene_id):
    project, scene = get_scene(store, project_id, scene_id)
    scene_draft = scene["draft"]
    selections = []
    context = scene_draft["context"]
    assets = context["assets"] if context is not None else project["assets"]
    for selected in scene_draft["selections"]:
        asset = store._find(assets, selected["asset_id"], "Asset")
        name = asset["name"]
        if selected["outfit_id"]:
            name += " / " + store._find(asset["outfits"], selected["outfit_id"], "Outfit")["name"]
        selections.append((name, asset["id"]))
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
        return (gr.skip(),) * (MAX_SECTIONS + 10)
    store.delete(project_id)
    return (
        gr.update(choices=project_choices(store), value=None),
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        *empty_scene_values(),
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
    if not project_id:
        return (gr.skip(), *([gr.skip()] * (MAX_SECTIONS + 7)))
    store.delete_scene(project_id, scene_id)
    return gr.update(choices=scene_choices(store.load(project_id)), value=None), *empty_scene_values()


def update_text(store, field, project_id, scene_id, value, section_index=None):
    store.update_draft(project_id, scene_id, field, value, section_index)
    return "Draft saved"


def create_asset(store, project_id, kind, name, text):
    asset = store.add_asset(project_id, kind, name, text)
    return (
        gr.update(choices=asset_choices(store.load(project_id)), value=asset["id"]),
        gr.update(choices=[], value=None), "", "", "Asset saved",
    )


def load_asset(store, project_id, asset_id):
    project = store.load(project_id)
    asset = store._find(project["assets"], asset_id, "Asset")
    return (
        asset["kind"], asset["name"], asset["text"],
        gr.update(choices=[(outfit["name"], outfit["id"]) for outfit in asset["outfits"]], value=None),
        "", "",
    )


def save_asset(store, project_id, asset_id, name, text):
    store.update_asset(project_id, asset_id, name, text)
    return gr.update(choices=asset_choices(store.load(project_id)), value=asset_id), "Asset saved"


def save_outfit(store, project_id, asset_id, outfit_id, name, text):
    if outfit_id:
        store.update_outfit(project_id, asset_id, outfit_id, name, text)
    else:
        outfit_id = store.add_outfit(project_id, asset_id, name, text)["id"]
    project = store.load(project_id)
    asset = store._find(project["assets"], asset_id, "Character")
    return gr.update(choices=[(item["name"], item["id"]) for item in asset["outfits"]], value=outfit_id), "Outfit saved"


def load_outfit(store, project_id, asset_id, outfit_id):
    project = store.load(project_id)
    asset = store._find(project["assets"], asset_id, "Character")
    outfit = store._find(asset["outfits"], outfit_id, "Outfit")
    return outfit["name"], outfit["text"]


def add_selection(store, project_id, scene_id, asset_id, outfit_id):
    store.select_asset(project_id, scene_id, asset_id, outfit_id)
    return scene_values(store, project_id, scene_id)[MAX_SECTIONS + 2]


def remove_selection(store, project_id, scene_id, asset_id):
    store.remove_asset(project_id, scene_id, asset_id)
    return scene_values(store, project_id, scene_id)[MAX_SECTIONS + 2]


def save_variant(store, project_id, scene_id, name):
    variant = store.save_variant(project_id, scene_id, name)
    _, scene = get_scene(store, project_id, scene_id)
    return gr.update(choices=variant_choices(scene), value=variant["id"]), variant["prompt"], variant["negative"], None


def load_variant(store, project_id, scene_id, variant_id):
    if not project_id:
        return [gr.skip()] * (MAX_SECTIONS + 7)
    store.select_variant(project_id, scene_id, variant_id)
    return scene_values(store, project_id, scene_id)


def step_variant(store, project_id, scene_id, direction):
    if not project_id:
        return [gr.skip()] * (MAX_SECTIONS + 7)
    _, scene = get_scene(store, project_id, scene_id)
    variants = scene["variants"]
    if not variants:
        raise ValueError("This scene has no saved variants")
    current = next((index for index, item in enumerate(variants) if item["id"] == scene["selected_variant"]), 0)
    store.select_variant(project_id, scene_id, variants[(current + direction) % len(variants)]["id"])
    return scene_values(store, project_id, scene_id)


def delete_variant(store, project_id, scene_id, variant_id):
    if not project_id:
        return (gr.skip(),) * 4
    store.delete_variant(project_id, scene_id, variant_id)
    _, scene = get_scene(store, project_id, scene_id)
    return gr.update(choices=variant_choices(scene), value=None), "", "", None


def variant_prompt(store, project_id, scene_id, variant_id):
    _, scene = get_scene(store, project_id, scene_id)
    if scene["selected_variant"] != variant_id:
        raise ValueError("Load and select this variant before applying it")
    variant = store._find(scene["variants"], variant_id, "Variant")
    return "prompt", variant["prompt"], variant["negative"]


def preview_draft(store, project_id, scene_id):
    project, scene = get_scene(store, project_id, scene_id)
    return store.resolve(project, scene["draft"])


def storyboard_ui(library, apply_js):
    store = StoryboardStore(library.path.parent / "storyboards", library)
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown("Create storyboards, arrange scenes and save named variants. Variants keep their resolved prompt; editing a loaded variant creates a separate draft.")
        with gr.Row():
            project = gr.Dropdown(label="Storyboard", choices=project_choices(store))
            project_name = gr.Textbox(label="New storyboard name")
            directory = gr.Textbox(label="Storyboard image directory", value=str(library.path.parent / "storyboard-images"))
            create = gr.Button("Create storyboard")
            rename = gr.Button("Rename storyboard")
            delete = gr.Button("Delete storyboard")
            set_directory = gr.Button("Save image directory")
        with gr.Row():
            scene = gr.Dropdown(label="Scene")
            scene_name = gr.Textbox(label="New scene name")
            create_scene_button = gr.Button("Add scene")
            refresh_scene = gr.Button("Refresh scene")
            rename_scene_button = gr.Button("Rename scene")
            move_up = gr.Button("Move earlier")
            move_down = gr.Button("Move later")
            delete_scene_button = gr.Button("Delete scene")
        with gr.Accordion("Project characters, locations and objects", open=False):
            with gr.Row():
                asset = gr.Dropdown(label="Project asset")
                asset_kind = gr.Dropdown(label="Type", choices=["Character", "Location", "Object"], value="Character")
                asset_name = gr.Textbox(label="Name")
            asset_text = gr.Textbox(label="Base description (library references are supported)", lines=2)
            with gr.Row():
                snippet = gr.Dropdown(label="Library snippet", choices=snippet_choices(library))
                refresh_snippets = gr.Button("Refresh library")
                snippet_to_asset = gr.Button("Insert into base description")
                snippet_to_outfit = gr.Button("Insert into outfit")
            with gr.Row():
                add_asset_button = gr.Button("Add asset")
                update_asset_button = gr.Button("Update selected asset")
            with gr.Row():
                outfit = gr.Dropdown(label="Outfit")
                outfit_name = gr.Textbox(label="Outfit name")
                outfit_text = gr.Textbox(label="Outfit description")
                add_outfit_button = gr.Button("Save outfit")
                new_outfit = gr.Button("New outfit")
            asset_status = gr.Markdown()
        with gr.Row():
            add_selected_asset = gr.Button("Add selected asset and outfit to scene")
            selection = gr.Dropdown(label="Assets in scene")
            remove_selected_asset = gr.Button("Remove asset from scene")
        template = gr.Textbox(label="Scene template", lines=3, interactive=False, info="Place sections using <<Section>> and write transitions freely.")
        with gr.Row():
            sections = [gr.Textbox(label=f"Section {index + 1}", lines=2, visible=False, interactive=False) for index in range(5)]
        with gr.Row():
            sections += [gr.Textbox(label=f"Section {index + 1}", lines=2, visible=False, interactive=False) for index in range(5, MAX_SECTIONS)]
        negative = gr.Textbox(label="Negative prompt", lines=2, interactive=False)
        draft_status = gr.Markdown()
        with gr.Row():
            preview = gr.Button("Preview draft")
            refresh_context = gr.Button("Use current library and project assets")
            variant_name = gr.Textbox(label="New variant name")
            save = gr.Button("Save draft as new variant")
        with gr.Row():
            variant = gr.Dropdown(label="Scene variant")
            previous_variant = gr.Button("Previous variant")
            next_variant = gr.Button("Next variant")
            load = gr.Button("Load and select variant")
            delete_variant_button = gr.Button("Delete variant")
        prompt = gr.Textbox(label="Resolved prompt", lines=4, interactive=False)
        negative_result = gr.Textbox(label="Resolved negative prompt", lines=2, interactive=False)
        image = gr.Image(label="Selected variant image", interactive=False, type="filepath")
        with gr.Row():
            apply_txt = gr.Button("Apply selected variant to txt2img")
            apply_img = gr.Button("Apply selected variant to img2img")
            publish = gr.Button("Publish storyboard")
        status = gr.Markdown()
        kind = gr.Textbox(value="prompt", visible=False)
        selected_prompt = gr.Textbox(visible=False)
        selected_negative = gr.Textbox(visible=False)

        scene_outputs = [template, *sections, negative, selection, variant, prompt, negative_result, image]
        common = {"concurrency_id": "storyboard", "show_progress": "hidden"}
        create.click(partial(create_project, store), [project_name, directory], [project, directory, scene, asset, *scene_outputs], **common)
        project.input(partial(project_values, store), project, [directory, scene, asset, *scene_outputs], **common)
        project.input(lambda project_id: store.load(project_id)["name"], project, project_name, **common)
        rename.click(partial(rename_project, store), [project, project_name], project, **common)
        delete.click(
            partial(delete_project, store), project, [project, scene, asset, *scene_outputs],
            js='(project) => confirm("Delete this storyboard? Its image files will not be deleted.") ? [project] : [null]',
            **common,
        )
        set_directory.click(lambda project_id, path: (store.set_image_directory(project_id, path), "Image directory saved")[1], [project, directory], status, **common)
        create_scene_button.click(partial(create_scene, store), [project, scene_name], [scene, *scene_outputs], **common)
        scene.input(partial(scene_values, store), [project, scene], scene_outputs, **common)
        refresh_scene.click(partial(scene_values, store), [project, scene], scene_outputs, **common)
        scene.input(lambda project_id, scene_id: get_scene(store, project_id, scene_id)[1]["name"], [project, scene], scene_name, **common)
        rename_scene_button.click(partial(rename_scene, store), [project, scene, scene_name], scene, **common)
        move_up.click(partial(move_scene, store, direction=-1), [project, scene], scene, **common)
        move_down.click(partial(move_scene, store, direction=1), [project, scene], scene, **common)
        delete_scene_button.click(
            partial(delete_scene, store), [project, scene], [scene, *scene_outputs],
            js='(project, scene) => confirm("Delete this scene? Its image files will not be deleted.") ? [project, scene] : [null, scene]',
            **common,
        )
        asset.input(partial(load_asset, store), [project, asset], [asset_kind, asset_name, asset_text, outfit, outfit_name, outfit_text], **common)
        add_asset_button.click(partial(create_asset, store), [project, asset_kind, asset_name, asset_text], [asset, outfit, outfit_name, outfit_text, asset_status], **common)
        update_asset_button.click(partial(save_asset, store), [project, asset, asset_name, asset_text], [asset, asset_status], **common)
        refresh_snippets.click(lambda: gr.update(choices=snippet_choices(library)), outputs=snippet, **common)
        snippet_to_asset.click(partial(insert_snippet, library), [snippet, asset_text], [asset_text, asset_status], **common)
        snippet_to_outfit.click(partial(insert_snippet, library), [snippet, outfit_text], [outfit_text, asset_status], **common)
        outfit.input(partial(load_outfit, store), [project, asset, outfit], [outfit_name, outfit_text], **common)
        add_outfit_button.click(partial(save_outfit, store), [project, asset, outfit, outfit_name, outfit_text], [outfit, asset_status], **common)
        new_outfit.click(lambda: (gr.update(value=None), "", ""), outputs=[outfit, outfit_name, outfit_text], **common)
        add_selected_asset.click(partial(add_selection, store), [project, scene, asset, outfit], selection, **common)
        remove_selected_asset.click(partial(remove_selection, store), [project, scene, selection], selection, **common)
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
            partial(load_variant, store), [project, scene, variant], scene_outputs,
            js='(project, scene, variant) => confirm("Replace the current draft with this saved variant?") ? [project, scene, variant] : [null, scene, variant]',
            **common,
        )
        for button, direction in ((previous_variant, -1), (next_variant, 1)):
            button.click(
                partial(step_variant, store, direction=direction), [project, scene], scene_outputs,
                js='(project, scene) => confirm("Replace the current draft with another saved variant?") ? [project, scene] : [null, scene]',
                **common,
            )
        delete_variant_button.click(
            partial(delete_variant, store),
            [project, scene, variant], [variant, prompt, negative_result, image],
            js='(project, scene, variant) => confirm("Delete this variant? Its image file will not be deleted.") ? [project, scene, variant] : [null, scene, variant]',
            **common,
        )
        for button, tab in ((apply_txt, "txt2img"), (apply_img, "img2img")):
            button.click(
                partial(variant_prompt, store), [project, scene, variant],
                [kind, selected_prompt, selected_negative], **common,
            ).success(
                fn=None,
                js=f'function(kind, prompt, negative) {{ if (!confirm("Replace the current prompts?")) return []; return ({apply_js(tab)})(kind, prompt, negative); }}',
                inputs=[kind, selected_prompt, selected_negative], show_progress="hidden",
            )
        publish.click(lambda project_id: f"Published to: {store.publish(project_id)}", project, status, **common)
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
        for tab in ("txt2img", "img2img"):
            if component.elem_id == f"{tab}_gallery":
                library = library_factory()
                store = StoryboardStore(library.path.parent / "storyboards", library)
                with gr.Accordion("Assign image to storyboard", open=False):
                    with gr.Row():
                        project = gr.Dropdown(label="Storyboard", choices=project_choices(store))
                        scene = gr.Dropdown(label="Scene")
                        variant = gr.Dropdown(label="Variant")
                        refresh = gr.Button("Refresh")
                        take = gr.Button("Use selected image")
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
