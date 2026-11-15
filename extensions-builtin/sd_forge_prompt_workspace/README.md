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
