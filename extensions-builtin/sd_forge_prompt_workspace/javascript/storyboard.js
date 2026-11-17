(function () {
    let activeField = null;

    function promptField(target) {
        return target instanceof HTMLTextAreaElement && target.closest(".storyboard-prompt-source") && !target.disabled && !target.readOnly;
    }

    function assetChip(event) {
        const target = event.composedPath()[0];
        return typeof target?.closest === "function" ? target.closest(".storyboard-asset-chip") : null;
    }

    function feedback(message) {
        const status = gradioApp().querySelector("#storyboard_asset_palette .storyboard-asset-feedback");
        if (status) status.textContent = message;
    }

    document.addEventListener("focusin", (event) => {
        const target = event.composedPath()[0];
        if (promptField(target)) activeField = target;
    });

    document.addEventListener("dragstart", (event) => {
        const chip = assetChip(event);
        if (!chip || !event.dataTransfer) return;
        // Let the browser insert plain text at the exact drop position in a textarea.
        event.dataTransfer.setData("text/plain", chip.dataset.assetReference);
        event.dataTransfer.effectAllowed = "copy";
    });

    function insertChip(chip) {
        if (!activeField?.isConnected || !promptField(activeField) || !uiElementIsVisible(activeField)) {
            feedback("Click the desired position in a section or the template first.");
            return;
        }
        activeField.focus();
        activeField.setRangeText(chip.dataset.assetReference, activeField.selectionStart, activeField.selectionEnd, "end");
        updateInput(activeField);
        feedback("Asset reference inserted.");
    }

    document.addEventListener("click", (event) => {
        const chip = assetChip(event);
        if (chip) insertChip(chip);
    });

    document.addEventListener("keydown", (event) => {
        const chip = assetChip(event);
        if (!chip || (event.key !== "Enter" && event.key !== " ")) return;
        event.preventDefault();
        insertChip(chip);
    });
})();
