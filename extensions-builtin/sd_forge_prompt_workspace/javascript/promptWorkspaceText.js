(function () {
    function punctuationBoundary(left, right) {
        const leftEnd = left.trimEnd().length;
        const rightStart = right.length - right.trimStart().length;
        if (!leftEnd || rightStart === right.length) return [left, right];

        const previous = left[leftEnd - 1];
        const following = right[rightStart];
        if (",;".includes(following) && ".,;:!?".includes(previous)) {
            right = right.slice(0, rightStart) + right.slice(rightStart + 1);
        } else if (following === "." && previous === ".") {
            const leftDots = left.slice(0, leftEnd).match(/\.+$/)[0].length;
            const rightDots = right.slice(rightStart).match(/^\.+/)[0].length;
            right = right.slice(0, rightStart) + right.slice(rightStart + Math.min(leftDots, rightDots));
        } else if (following === "." && ",;".includes(previous)) {
            left = left.slice(0, leftEnd - 1) + "." + left.slice(leftEnd);
            right = right.slice(0, rightStart) + right.slice(rightStart + 1);
        } else if (following === "." && "!?".includes(previous) && right[rightStart + 1] !== ".") {
            right = right.slice(0, rightStart) + right.slice(rightStart + 1);
        }
        return [left, right];
    }

    function join(left, right, separator) {
        left = left.trimEnd();
        right = right.trimStart();
        if (!left || !right) return left || right;
        [left, right] = punctuationBoundary(left, right);
        if (!right) return left;
        if (/^\s/.test(right) || ",.;:!?".includes(right[0])) return left + right;
        return left + (",.;:!?".includes(left[left.length - 1]) ? " " : separator) + right;
    }

    function needsSpace(left, right) {
        return left && right && !/\s$/.test(left) && !/^\s/.test(right)
            && !/[([{]$/.test(left) && !/^[,.;:!?)}\]]/.test(right);
    }

    function insert(textarea, text) {
        if (!text) return;
        let before = textarea.value.slice(0, textarea.selectionStart);
        let after = textarea.value.slice(textarea.selectionEnd);
        let inserted = text;
        [before, inserted] = punctuationBoundary(before, inserted);
        if (needsSpace(before, inserted)) inserted = " " + inserted;
        [inserted, after] = punctuationBoundary(inserted, after);
        if (needsSpace(inserted, after)) inserted += " ";

        const position = before.length + inserted.length;
        textarea.focus();
        textarea.setRangeText(before + inserted + after, 0, textarea.value.length, "end");
        textarea.setSelectionRange(position, position);
        updateInput(textarea);
    }

    window.promptWorkspaceText = {join, insert};
})();
