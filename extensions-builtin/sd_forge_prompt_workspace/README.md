# Prompt Workspace

Prompt Workspace is a built-in Forge Neo extension for keeping reusable prompt
snippets and named prompt versions in a browsable library.

- **Snippet** entries are appended to the selected txt2img or img2img prompt.
- **Prompt** entries replace the selected positive and negative prompts.
- Prompt versions can be saved beside one another and browsed with the variant
  navigation buttons.
- Categories, tags, and prompt text can be searched and filtered.

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
