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

Workspace and storyboard controls use compact, text-width buttons and separate
field rows from action bars. The library browser and gallery assignment panel
use the same layout; whole buttons wrap onto another row on narrower screens
instead of being squeezed. These styles apply only to this extension.
Field rows align the bottoms of inputs even when labels or hints differ in
height.

## Saving library entries

**Save** normally uses the category and name to find an entry. For prompts,
the variant name also identifies the version. An existing match is updated
immediately; otherwise a new entry or variant is created, except for category
moves described below. The editor contents and current library selection stay
in place; **New** clears both.
Names and categories are matched exactly, ignoring surrounding whitespace;
variant names are matched case-insensitively. A snippet and a prompt cannot
share a name in the same category.

When a selected entry keeps its name and type but its category changes,
**Save** moves it to the new category. For prompts, all variants move together;
the editor's variant is saved as usual. References remain unchanged unless
edited. If the destination already contains the same name, the move is
rejected without overwriting either entry. To create a separate entry with the
same name in another category, use **New** first; this clears the editor and
selection.

**Save as new variant** requires an existing prompt with the entered category
and name and refuses to replace an existing variant. Older libraries may
contain ambiguous duplicate names or prompt groups. Such conflicts are reported
without changing the library and must be resolved before saving that name.

**Delete selected entry** deletes the entry selected in **Library**, not
necessarily the one shown in the editor after a save. Saving under a different
name or entry type leaves the original selected for deletion. Reference names
remain globally unique, so copying a snippet with its existing reference is not
possible. Deleting a referenced snippet is blocked if another library entry
uses it; references in generation fields and storyboard drafts are not checked.

## Using the library while generating

Open **Prompt Workspace** under the prompt fields in txt2img or img2img.
Filter by category or search, then click a snippet to insert it at the
positive prompt's cursor position. Use **Previous** and **Next** to browse
more results. With **Insert snippets as references** checked, snippets with a
reference name insert `<name>`; otherwise their resolved text is inserted.
Snippets without a reference name always insert their text. Any negative text
from the snippet is appended to the negative prompt.

Selecting a saved prompt shows its preview; **Replace with selected prompt**
immediately replaces both prompt fields. Use the refresh icon after creating
or changing entries in the workspace. Deleting a library entry also takes
effect immediately.

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
locations and objects. Select the storyboard at the top, then work through the
steps: **1 · Storyboard** creates it and sets its image directory, **2 · Assets**
defines its characters, locations and objects, and **3 · Scenes** adds scenes.
Within a scene, **a · Cast** chooses its assets and **b · Prompt & image** edits
the draft, variants and image. The scene editor uses the library's current default section
names when a scene is created. Its template can include section placeholders
and free text between them. Changes to a scene's draft are saved as you edit.

Add a project asset with a base description and, for characters, named outfits.
Asset names are unique within a storyboard, and outfit names within a
character (ignoring case). **Save asset** updates the asset with the entered
name or creates it; if an outfit name and description are filled in for a
character, that outfit is saved as well. **Save outfit** saves only the outfit.
Deleting an asset removes it from scene drafts; saved variants keep their copy.
Descriptions may use library references; the library selector can insert
snippets into them. In **a · Cast**, select an asset and outfit and add them to
the scene.
Character and object descriptions are placed in the Subject section, and
locations in Environment. Character names precede their descriptions and
outfits, and selected assets are separated with semicolons. If those section
names are not present, the first and third
configured sections are used instead (or the last section if there are fewer).
Snippet negative text is not automatically included when inserting into an
asset; the interface shows it so it can be added to the scene's negative prompt.

Save a draft as a named variant to preserve its structure, asset definitions,
library reference texts and resolved positive and negative prompts. Selecting
another variant immediately replaces the current draft, so save it first if you
want to keep it. A loaded variant keeps its saved definitions while editing;
**Use current library and project assets** explicitly refreshes them. Applying
a selected variant to txt2img or img2img immediately replaces the prompt fields.

After generating, open **Assign image to storyboard** below the output gallery,
select the storyboard and scene, and use the selected gallery image. The
variant must be selected in the Storyboard tab first. Each variant keeps one
image; assigning another replaces it. Use **Refresh scene** in the Storyboard
tab to see an image assigned from a generation tab. Gallery updates, including
img2img batch results, reuse the existing assignment panel.

**Publish storyboard** requires an image for the selected variant of every
scene. It creates a new subdirectory in the storyboard's image directory,
writes `1.png`, `2.png`, etc. in scene order, and removes image metadata from
the copies. The stored variant images are not changed. If publishing fails
while writing, the new directory may contain incomplete files.

Storyboard project files are stored separately under `storyboards/` in the
active Prompt Workspace library directory. Each project records its image
directory separately. Changing either directory does not move existing data.
Deleting a storyboard, scene or variant takes effect immediately and does not
delete its image files.

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
