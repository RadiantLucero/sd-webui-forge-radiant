// Show a checkerboard behind transparent gallery and lightbox images; the dark UI background otherwise looks like black pixels.
(function () {
    const cell = 8;
    const checkered = new Set();
    const resizeObserver = new ResizeObserver((entries) => entries.forEach((entry) => paintAlphaChecker(entry.target)));

    function hasAlpha(img) {
        if (/\.jpe?g(\?|$)/i.test(img.currentSrc || img.src)) return false;
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const context = canvas.getContext("2d", {willReadFrequently: true});
        context.drawImage(img, 0, 0);
        let data;
        try {
            data = context.getImageData(0, 0, canvas.width, canvas.height).data;
        } catch {
            return false; // cross-origin images cannot be inspected
        }
        for (let i = 3; i < data.length; i += 4) {
            if (data[i] < 255) return true;
        }
        return false;
    }

    function paintAlphaChecker(img) {
        const style = getComputedStyle(img);
        const boxWidth = img.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
        const boxHeight = img.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
        if (!checkered.has(img) || boxWidth <= 0 || boxHeight <= 0 || !img.naturalWidth) return;

        // Match the area that object-fit actually draws, so letterboxing keeps the normal background.
        let scaleX = boxWidth / img.naturalWidth;
        let scaleY = boxHeight / img.naturalHeight;
        if (style.objectFit === "none") {
            scaleX = scaleY = 1;
        } else if (style.objectFit !== "fill") {
            const scale = style.objectFit === "cover" ? Math.max(scaleX, scaleY) : Math.min(scaleX, scaleY);
            scaleX = scaleY = style.objectFit === "scale-down" ? Math.min(scale, 1) : scale;
        }
        const width = img.naturalWidth * scaleX;
        const height = img.naturalHeight * scaleY;
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">` +
            `<pattern id="c" width="${2 * cell}" height="${2 * cell}" patternUnits="userSpaceOnUse">` +
            `<rect width="${2 * cell}" height="${2 * cell}" fill="#d9d9d9"/>` +
            `<path d="M0 0h${cell}v${cell}H0zM${cell} ${cell}h${cell}v${cell}H${cell}z" fill="#a6a6a6"/></pattern>` +
            `<rect width="100%" height="100%" fill="url(#c)"/></svg>`;
        img.style.backgroundImage = `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
        img.style.backgroundOrigin = "content-box";
        img.style.backgroundPosition = style.objectPosition;
        img.style.backgroundRepeat = "no-repeat";
    }

    function clearAlphaChecker(img) {
        checkered.delete(img);
        resizeObserver.unobserve(img);
        for (const property of ["background-image", "background-origin", "background-position", "background-repeat"]) {
            img.style.removeProperty(property);
        }
    }

    document.addEventListener("load", (event) => {
        const img = event.target;
        if (!(img instanceof HTMLImageElement) || !(img.id === "modalImage" || img.closest(".gradio-gallery"))) return;
        if (!hasAlpha(img)) {
            if (checkered.has(img)) clearAlphaChecker(img);
            return;
        }
        checkered.add(img);
        resizeObserver.observe(img);
        paintAlphaChecker(img);
    }, true);

    onAfterUiUpdate(() => {
        for (const img of checkered) {
            if (!img.isConnected) clearAlphaChecker(img);
        }
    });
})();
