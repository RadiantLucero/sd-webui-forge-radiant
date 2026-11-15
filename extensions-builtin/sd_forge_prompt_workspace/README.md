# Prompt Workspace

Prompt Workspace is a built-in Forge Neo extension for keeping reusable prompt
snippets and named prompt versions in a browsable library.

- **Snippet** entries are appended to the selected txt2img or img2img prompt.
- **Prompt** entries replace the selected positive and negative prompts.
- Prompt versions can be saved beside one another and browsed with the variant
  navigation buttons.
- Categories, tags, and prompt text can be searched and filtered.
- Line breaks and repeated spaces in saved and applied prompts are reduced to
  single spaces, because Forge passes them to the text encoder unchanged.

## Using the library while generating

Open **Prompt Workspace** under the prompt fields in txt2img or img2img.
Filter by category or search, then click a snippet to insert it at the
positive prompt's cursor position. Use **Previous** and **Next** to browse
more results. With **Insert snippets as references** checked, snippets with a
reference name insert `<name>`; otherwise their resolved text is inserted.
Snippets without a reference name always insert their text. Any negative text
from the snippet is appended to the negative prompt.

Selecting a saved prompt shows its preview; **Replace with selected prompt**
replaces both prompt fields after confirmation. Use the refresh icon after
creating or changing entries in the workspace.

## Prompt templates and sections

Prompt entries have a **Text / template** field and section text fields. Use
`<<Section name>>` to insert a section wherever it belongs in the template.
The read-only **Result** shows the resolved prompt, with free text between
sections preserved. Select sections to insert their placeholders into the
template; when a section is deselected, its placeholder is removed. New
prompts use the default section names below:

1. Subject
2. Action
3. Environment
4. Framing
5. Lighting
6. Details
7. Style

Change the defaults in **Default sections** in the tab. Enter one name per
line; up to 10 names are supported. Each prompt stores its own section list,
so changing the defaults does not alter existing prompts. **Add default
sections** adds any current defaults missing from the selected prompt.

## Reusable snippets

Give a snippet an optional unique **Reference name**, such as `character_A`.
Use it in prompt templates or other snippets as `<character_A>`. References
can be nested and are resolved case-insensitively. A prompt cannot be saved
with an unknown reference or a circular reference. Snippets may refer to
snippets that have not been created yet, allowing nested definitions to be
built incrementally. A referenced snippet cannot be deleted or renamed until
its references are removed.

References can also be typed directly into the positive or negative
txt2img/img2img prompts. During generation, the extension resolves them using
the current library, including nested references. The input fields remain
unchanged; the generated image's prompt records the resolved text. Unknown or
circular references abort generation with an error. Resolution runs before
generation scripts such as Dynamic Prompts, so references introduced later by
a wildcard or another script are not expanded.

LoRA syntax such as `<lora:style:0.8>` is left unchanged. Snippets can also
have a default section; **Insert snippet** adds them to the selected section.
If the snippet has a reference name, its reference is inserted instead of
copying its text. A snippet's negative prompt is appended to the negative
prompt field.

## Storyboards

The **Storyboard** tab manages projects, ordered scenes, project characters,
locations and objects. Create a storyboard with a directory for its images,
then add scenes. The scene editor uses the library's current default section
names when a scene is created. Its template can include section placeholders
and free text between them. Changes to a scene's draft are saved as you edit.

Add a project asset with a base description and, for characters, named outfits.
Descriptions may use library references; the library selector can insert
snippets into them. Select an asset and outfit and add them to a scene.
Character and object descriptions are placed in the Subject section, and
locations in Environment. Character names precede their descriptions and
outfits, and selected assets are separated with semicolons. If those section
names are not present, the first and third
configured sections are used instead (or the last section if there are fewer).
Snippet negative text is not automatically included when inserting into an
asset; the interface shows it so it can be added to the scene's negative prompt.

Save a draft as a named variant to preserve its structure, asset definitions,
library reference texts and resolved positive and negative prompts. Selecting
another variant replaces the current draft after confirmation. A loaded variant
keeps its saved definitions while editing; **Use current library and project
assets** explicitly refreshes them. Applying a selected variant to txt2img or
img2img replaces the prompt fields after confirmation.

After generating, open **Assign image to storyboard** below the output gallery,
select the storyboard and scene, and use the selected gallery image. The
variant must be selected in the Storyboard tab first. Each variant keeps one
image; assigning another replaces it. Use **Refresh scene** in the Storyboard
tab to see an image assigned from a generation tab.

**Publish storyboard** requires an image for the selected variant of every
scene. It creates a new subdirectory in the storyboard's image directory,
writes `1.png`, `2.png`, etc. in scene order, and removes image metadata from
the copies. The stored variant images are not changed. If publishing fails
while writing, the new directory may contain incomplete files.

Storyboard project files are stored separately under `storyboards/` in the
active Prompt Workspace library directory. Each project records its image
directory separately. Changing either directory does not move existing data.
Deleting a storyboard, scene or variant does not delete its image files.

## Storage location

By default, the library is stored at `data/library.json` inside the extension.
This directory is ignored by Git, but deleting the extension can also delete
the library. Back it up or choose a separate directory to keep it independent.

To select a directory on the machine running Forge, enter its path in
**Settings > Prompt Workspace > Prompt Workspace library directory**, apply
the settings, and reload the UI. The active directory is shown in the Prompt
Workspace tab. Leave the setting empty to use the extension's default directory.
Relative paths are resolved from Forge's working directory.

Changing the setting uses the library in the selected directory; it does not
copy, move, or overwrite an existing library. To transfer entries, stop Forge
and copy `library.json` to the new directory, without overwriting a library you
want to keep. Earlier versions used `<data-dir>/prompt-workspace/library.json`;
that file is not moved automatically.
