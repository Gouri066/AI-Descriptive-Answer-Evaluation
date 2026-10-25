"""
Rewrites the OCR extraction section and edit_answers route in assessments.py.
Uses Gemini Vision AI for accurate question extraction.
Also fixes UNIQUE constraint error by deleting existing questions before inserting new ones.
"""

REPLACEMENT = """        # After storing the paper image, extract questions using Gemini Vision AI
        try:
            # Delete any existing questions for this assessment first (re-upload support)
            db.execute("DELETE FROM questions WHERE assessment_id = ?", (assessment_id,))

            extracted_questions = _extract_questions_from_image(file_path)
            created = 0
            if extracted_questions:
                for idx, qtext in enumerate(extracted_questions, start=1):
                    db.execute(
                        "INSERT INTO questions "
                        "(assessment_id, question_no, question_text, question_type, max_marks, correct_answer)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        (assessment_id, idx, qtext, "fill_blank", 1, ""),
                    )
                    created += 1
            else:
                db.execute(
                    "INSERT INTO questions "
                    "(assessment_id, question_no, question_text, question_type, max_marks, correct_answer)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (assessment_id, 1,
                     "[Placeholder] Please edit this question after upload",
                     "fill_blank", 1, ""),
                )
                created = 1
            db.commit()
            flash(
                "Extracted " + str(created) + " question(s) from the uploaded paper."
                " Please provide correct answers.",
                "success"
            )
            return redirect(url_for("assessments.edit_answers", assessment_id=assessment_id))
        except Exception as inner_e:
            print("[OCR PROCESSING ERROR]", inner_e)
            import traceback as _tb
            _tb.print_exc()
            flash("Error processing uploaded paper image.", "error")
            return redirect(url_for("assessments.detail", assessment_id=assessment_id))

    except Exception as e:
        print("[UPLOAD_PAPER ERROR]", e)
        import traceback as _tb
        _tb.print_exc()
        if hasattr(e, "args") and any("paper_image" in str(arg).lower() for arg in e.args):
            try:
                db = get_db()
                db.execute("ALTER TABLE assessments ADD COLUMN paper_image TEXT")
                db.commit()
                db.execute(
                    "UPDATE assessments SET paper_image = ? WHERE id = ?",
                    (file_path, assessment_id)
                )
                db.commit()
                flash("Question paper uploaded successfully.", "success")
                return redirect(url_for("assessments.detail", assessment_id=assessment_id))
            except Exception as inner_e:
                print("[UPLOAD_PAPER MIGRATION ERROR]", inner_e)
                flash("An error occurred while fixing the database schema.", "error")
                return redirect(url_for("assessments.detail", assessment_id=assessment_id))
        flash("Error uploading paper: " + str(e), "error")
        return redirect(url_for("assessments.detail", assessment_id=assessment_id))


# ---------------------------------------------------------------------------
# Helper: extract questions from image (Gemini Vision AI with Tesseract fallback)
# ---------------------------------------------------------------------------
def _extract_questions_from_image(image_path):
    """Try Gemini Vision AI first, fall back to Tesseract if no API key."""
    import os as _os
    api_key = _os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        try:
            return _gemini_extract(image_path, api_key)
        except Exception as e:
            print("[GEMINI ERROR]", e)
    return _tesseract_extract(image_path)


def _gemini_extract(image_path, api_key):
    """Use Gemini Vision to intelligently extract questions from a paper image."""
    import google.generativeai as genai
    from PIL import Image as _Image

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    img = _Image.open(image_path)

    prompt = (
        "You are a school teacher's assistant. Look at this question paper image carefully.\\n\\n"
        "Extract ONLY the actual questions (not the header, school name, class, subject, "
        "marks, date, or section headings like 'Fill in the blanks').\\n\\n"
        "Return ONLY a numbered list, one question per line, like this:\\n"
        "1. <question text>\\n"
        "2. <question text>\\n\\n"
        "Rules:\\n"
        "- For fill-in-the-blank questions, keep blanks as ___ (three underscores)\\n"
        "- Fix obvious OCR spelling errors\\n"
        "- Do NOT include school name, class, subject, time, max marks, student name, "
        "or section headings\\n"
        "- Only output the numbered list, nothing else"
    )

    response = model.generate_content([prompt, img])
    raw = response.text.strip()

    questions = []
    for line in raw.split("\\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\\d+[.)\\s]+(.+)", line)
        if m:
            q = m.group(1).strip()
            if len(q) > 3:
                questions.append(q)
    print("[GEMINI] Extracted", len(questions), "questions")
    return questions


def _tesseract_extract(image_path):
    """Fallback Tesseract OCR question extraction."""
    import pytesseract as _tess
    import cv2 as _cv2

    _tess.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    image = _cv2.imread(image_path)
    if image is None:
        return []
    gray = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = _cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = _cv2.adaptiveThreshold(gray, 255, _cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     _cv2.THRESH_BINARY, 11, 2)
    data = _tess.image_to_data(thresh, config="--psm 6 --oem 3",
                                output_type=_tess.Output.DICT)
    lines_dict = {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if word == "" or int(data["conf"][i]) < 30:
            continue
        key = (data["block_num"][i], data["line_num"][i])
        lines_dict.setdefault(key, []).append(word)
    lines = [" ".join(lines_dict[k]) for k in sorted(lines_dict.keys())]

    skip_kw = ["school", "sample paper", "class:", "time :", "subject:",
                "name of student", "max. marks", "max marks", "dhamtari",
                "delhi public", "oflue", "date", "nthe blanks", "fill in"]
    questions = []
    for txt in lines:
        txt = txt.strip()
        if len(txt) < 4:
            continue
        low = txt.lower()
        if any(kw in low for kw in skip_kw):
            continue
        if re.match(r"^\\s*[Qq]\\d+[.):]", txt):
            continue
        if re.match(r"^\\s*(i{1,3}v?|vi{0,3}|ix|x{1,3}|\\d+)[.)\\s]", txt, re.IGNORECASE):
            questions.append(txt)
            continue
        if "_" in txt or txt.endswith("?"):
            questions.append(txt)
            continue
        words = txt.split()
        alpha = sum(c.isalpha() for c in txt)
        if len(words) >= 4 and alpha / max(len(txt), 1) > 0.4:
            questions.append(txt)
    return questions


# ---------------------------------------------------------------------------
# Edit answers route  --  at module level so Flask registers it
# ---------------------------------------------------------------------------
@assessments_bp.route("/<int:assessment_id>/edit_answers", methods=("GET", "POST"))
@login_required
def edit_answers(assessment_id):
    \\"\\"\\"GET: show form to enter correct answers.  POST: save them.\\"\\"\\"
    assessment = get_owned_assessment(assessment_id)
    db = get_db()
    if request.method == "POST":
        for key, value in request.form.items():
            if key.startswith("answer_"):
                try:
                    q_id = int(key.split("_")[1])
                except ValueError:
                    continue
                db.execute(
                    "UPDATE questions SET correct_answer = ? WHERE id = ? AND assessment_id = ?",
                    (value.strip(), q_id, assessment_id),
                )
        db.commit()
        flash("Correct answers saved.", "success")
        return redirect(url_for("assessments.detail", assessment_id=assessment_id))
    questions = db.execute(
        "SELECT * FROM questions WHERE assessment_id = ? ORDER BY question_no",
        (assessment_id,),
    ).fetchall()
    return render_template(
        "edit_answers.html",
        assessment=assessment,
        questions=questions,
        question_types=QUESTION_TYPES,
    )
"""

with open('assessments.py', 'r', encoding='utf-8') as f:
    content = f.read()

start_marker = '        # After storing the paper image, attempt to extract questions'
start_idx = content.find(start_marker)
if start_idx == -1:
    raise RuntimeError("Start marker not found!")

prefix = content[:start_idx]
new_content = prefix + REPLACEMENT + "\n"

with open('assessments.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Written. Lines:", new_content.count("\n"))
