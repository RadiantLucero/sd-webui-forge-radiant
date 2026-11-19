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

Duplicate separators at the boundaries of expanded sections, references and
storyboard assets are avoided; punctuation inside the original text is kept.

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
the active variant and its image. The first scene starts empty with the library's
current default section names and an active variant named **Start**.
Each additional scene copies the current contents of the last scene in scene
order, regardless of which scene is selected, into its own **Start** variant.
This includes its template, sections, negative prompt, asset/outfit selections
and any frozen definitions, but not other variants or images. The copy can be
edited independently. Its template can include section placeholders and free
text between them. Template, section and negative-prompt edits, asset/outfit
assignments and definition refreshes are automatically saved into the active
variant. Other variants are not changed.

Add a project asset with a base description and, for characters, named outfits.
Asset names are unique within a storyboard, and outfit names within a
character (ignoring case). **Save asset** updates the asset with the entered
name or creates it; if an outfit name and description are filled in for a
character, that outfit is saved as well. **Save outfit** saves only the outfit.
Deleting a project asset removes it from scenes using current definitions;
variants with frozen definitions keep their copy.
Descriptions may use library references; the library selector can insert
snippets into them. In **a · Cast**, select an asset and outfit and add them to
the scene. In **b · Prompt & image**, its assets appear as draggable buttons.
Drag one to the desired text position in any section or directly in the
template. Alternatively, click a text position and then the asset button.
This inserts a reference such as `<asset:Old Mill>`, not a copy of the
description; references can also be typed and ignore case and surrounding
spaces. Asset names cannot contain `<` or `>`. The reference expands to the
asset description and selected outfit when resolving the active variant.
If an asset is replaced under a new name, update its references manually. Characters include their name before the
description. The same asset can be referenced at several positions.

New scenes only include assets where you place their references. Older scenes
retain automatic Subject/Environment placement until you manually place each
asset; it is then no longer added automatically, even if you remove its token.
Removing an asset from a scene also removes its references from the active
variant. Other variants retain their own text and definitions.
Snippet negative text is not automatically included when inserting into an
asset; copy it from the library if needed.

In **b · Prompt & image**, open **Prompt Workspace** to browse the library by
category or search. Click in the scene template or a section, then click a
snippet to insert it at that position. Select a saved prompt and use **Insert
selected prompt** to insert it instead of replacing the scene. Both are
inserted as resolved positive text, not references; negative text is not
inserted. Refresh the browser after editing the library.

The **Active scene variant** dropdown immediately activates and loads its selection;
**Previous** and **Next** cycle through variants. No separate load or save step
is needed. Enter a new name and choose **Create variant copy** to make an
independent copy of the current contents and activate it. The copy starts
without an image; edits to an existing variant keep its assigned image until
you explicitly replace it. Deleting the active variant activates a neighbor;
the last remaining variant cannot be deleted.

Variants preserve their asset definitions and library reference texts when
you leave them. A loaded variant keeps those definitions while editing;
**Use current library and project assets** refreshes and saves the active
variant's definitions. **Send active variant to txt2img/img2img** resolves and
saves its current contents, then immediately replaces the target prompt fields.
Unfinished or invalid references are still saved, but their resolution error is
displayed and the resolved prompt fields are cleared rather than showing old
results. Sending or publishing is blocked until the error is fixed.

Existing storyboards are upgraded automatically on first access. Their latest
editor contents are adopted into the previously active variant without
replacing its image or changing other variants. A scene without an active
variant receives **Start** (or **Start 2**, etc. if that name is already taken).

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

## Negative prompt visibility

Negative prompt fields are hidden by default. Enable **Settings > Prompt
Workspace > Show negative prompt fields** to show them in the library,
storyboard, txt2img, img2img, Hires. fix and style editors. This setting only
changes visibility: existing negative prompts are preserved and still applied.
It does not change CFG or model behavior.

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
