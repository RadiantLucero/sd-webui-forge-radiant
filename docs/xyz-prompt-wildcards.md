# Wildcards in X/Y/Z Prompt S/R

When the Dynamic Prompts extension is installed and enabled, X/Y/Z plot
**Prompt S/R** and **Prompt S/R (LoRA)** values may contain its wildcard
(`__name__`) and variant (`{a|b}`) syntax. The values are resolved once before
the grid is built, using the wildcard folder and the parser and wildcard
settings of Dynamic Prompts. Values without that syntax are left unchanged.

- Without **Combinatorial generation**, each value is replaced by one random
  draw. With a fixed seed, the draws are repeatable; repeated wildcards draw
  with consecutive seeds, so `__animals__, __animals__` can produce two
  different columns. **Unlink seed from prompt** makes the draws random.
- With **Combinatorial generation**, each value expands to all of its
  combinations, and every combination becomes its own column. **Max
  generations** limits the combinations per value; 0 means up to 1000.

Resolved values may contain commas because they are no longer split as CSV.

The first value is still the text searched for in the prompt. When it is a
wildcard, for example a prompt containing `__animals__` and the values
`__animals__, cat`, the prompt placeholder is replaced, and the first column is
replaced by the wildcard's resolved values instead of the placeholder.

The **X Values**, **Y Values**, and **Z Values** infotext fields keep the
original text, so pasting the infotext restores the wildcards. The resolved
columns are recorded as **X Resolved Values**, **Y Resolved Values**, or **Z
Resolved Values**.

A wildcard without values stops the plot with an error that names the wildcard
and the wildcard folder. Values stay literal while **Dynamic Prompts enabled**
is off or **Enable Jinja2 templates** is on.

Wildcards in the main prompt are still resolved by Dynamic Prompts for every
cell. In combinatorial mode, such a cell produces several images. With **Show
Grids in gallery** enabled, the cell shows a small grid of all of them;
otherwise, it shows only the first. The plot's size check does not account for
these cell grids. To get one image per cell and labeled values, put that
wildcard on its own Prompt S/R axis as well.

Unit tests for the resolution are in
[`tests/test_prompt_wildcards.py`](../tests/test_prompt_wildcards.py); they
require the `dynamicprompts` package and are skipped without it.
