(function () {
    function updateVisibility() {
        if (!Object.hasOwn(opts, "prompt_workspace_show_negative_prompts")) return;
        const fields = gradioApp().querySelectorAll(
            ".prompt-workspace-negative, #txt2img_neg_prompt, #img2img_neg_prompt, #hires_neg_prompt, " +
            "#txt2img_edit_style_neg_prompt, #img2img_edit_style_neg_prompt"
        );
        for (const field of fields) {
            field.classList.toggle("prompt-workspace-negative-hidden", !opts.prompt_workspace_show_negative_prompts);
        }
    }

    onOptionsAvailable(updateVisibility);
    onOptionsChanged(updateVisibility);
    onAfterUiUpdate(updateVisibility);
})();
