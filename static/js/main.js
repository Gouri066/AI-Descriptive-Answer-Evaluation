document.addEventListener("DOMContentLoaded", function () {
    // 1. Format hint under the "Correct answer" box for each question type
    const typeSelect = document.getElementById("question_type");
    const hint = document.getElementById("answerHint");

    if (typeSelect && hint) {
        const hints = {
            fill_blank: "Type the correct word. For several accepted answers, separate with | (example: colour|color).",
            numerical: "Type a number, for example 12.",
            shape: "Type one of: circle, square, triangle, rectangle, oval, star, heart, diamond.",
            matching: "Type pairs separated by commas, for example 1-C,2-A,3-B."
        };

        function updateHint() {
            hint.textContent = hints[typeSelect.value] || "";
        }

        typeSelect.addEventListener("change", updateHint);
        updateHint();
    }

    // 2. Preview of the chosen answer sheet image on the upload page
    const sheetInput = document.getElementById("sheet");
    const preview = document.getElementById("previewImg");

    if (sheetInput && preview) {
        sheetInput.addEventListener("change", function () {
            const file = sheetInput.files[0];
            if (file && file.type.startsWith("image/")) {
                preview.src = URL.createObjectURL(file);
                preview.hidden = false;
            } else {
                preview.hidden = true;
            }
        });
    }
});