import math
import re

from flask import (
    Blueprint, abort, flash, g, redirect, render_template, request, url_for
)

from auth import login_required
from submissions import remove_submission_files
from database import get_db
import os
import cv2
import traceback
import time
import pytesseract
from pytesseract import Output  # added for image_to_data output type

# ... (rest of imports unchanged)

# Clean and filter question lines (updated version)
def clean_question(text):
    txt = text.strip()
    # Discard obvious meta‑lines
    discard_keywords = [
        'no.', 'question', 'type', 'marks', 'correct answer',
        'edit', 'delete', 'school', 'subject', 'time', 'name',
        'max marks', 'sample', 'paper', 'total', 'fill in the blank',
        'multiple choice', 'true/false', 'fill', 'blank'
    ]
    low = txt.lower()
    if any(kw in low for kw in discard_keywords):
        return None
    # Try to extract a leading number (e.g., "1.", "2)")
    m = re.match(r'^\s*\d+[.)]?\s+(.+)', txt)
    if m:
        candidate = m.group(1).strip()
    else:
        # Fallback: accept lines that look like a fill‑in‑blank (contain underscore or end with '?')
        if '_' in txt or txt.endswith('?'):
            candidate = txt
        else:
            return None
    # Basic sanity – enough alphabetic content
    alpha = sum(c.isalpha() for c in candidate)
    if len(candidate) < 5 or alpha / max(len(candidate), 1) < 0.5:
        return None
    return candidate
from modules.ocr import _TEXT_CFG
from werkzeug.utils import secure_filename
assessments_bp = Blueprint("assessments", __name__, url_prefix="/assessments")

QUESTION_TYPES = {
    "fill_blank": "Fill in the blank",
    "numerical": "Numerical answer",
    "shape": "Shape",
    "matching": "Match the following",
    "addition": "Addition (vertical format)",
    "subtraction": "Subtraction (vertical format)",
    "multiplication": "Multiplication (vertical format)",
    "division": "Division (vertical format)",
    "pattern": "Pattern continuation",
    "clock": "Clock time reading",
    "true_false": "True / False",
    "number_name": "Number to word",
    "table_fill": "Table fill‑in",
    "shape_name": "Shape name identification",
}

SHAPES = ["circle", "square", "triangle", "rectangle", "oval",
          "star", "heart", "diamond"]

SUBJECTS = ["English", "Mathematics", "EVS / Science", "Hindi", "Kannada",
            "Other"]


def get_owned_assessment(assessment_id):
    """Fetch an assessment ONLY if it belongs to the logged-in teacher.
    Otherwise show a 404 page, so teachers cannot open each other's data."""
    assessment = get_db().execute(
        "SELECT * FROM assessments WHERE id = ? AND teacher_id = ?",
        (assessment_id, g.teacher["id"]),
    ).fetchone()
    if assessment is None:
        abort(404)
    return assessment


def validate_question(form):
    """Check the question form and tidy the answer into a standard format.
    Returns (error_message, cleaned_data). One of them is always None."""
    text = form.get("question_text", "").strip()
    qtype = form.get("question_type", "")
    raw_answer = form.get("correct_answer", "").strip()

    try:
        marks = float(form.get("max_marks", ""))
    except ValueError:
        return "Maximum marks must be a number.", None

    if not text:
        return "Please enter the question.", None
    if qtype not in QUESTION_TYPES:
        return "Please choose a valid question type.", None
    if not math.isfinite(marks) or marks <= 0 or marks > 100:
        return "Maximum marks must be more than 0 and at most 100.", None
    if not raw_answer:
        return "Please enter the correct answer.", None

    if qtype == "numerical":
        try:
            value = float(raw_answer)
        except ValueError:
            return "Numerical answer must be a number, for example 12.", None
        if not math.isfinite(value):
            return "Numerical answer must be a normal number.", None
        answer = str(int(value)) if value.is_integer() else str(value)

    elif qtype == "fill_blank":
        options = [a.strip() for a in raw_answer.split("|") if a.strip()]
        if not options:
            return "Please enter at least one correct word.", None
        answer = "|".join(options)

    elif qtype == "shape":
        answer = raw_answer.lower()
        if answer not in SHAPES:
            return "Shape must be one of: " + ", ".join(SHAPES) + ".", None

    elif qtype in ("addition", "subtraction", "multiplication", "division"):
        # Expect the correct result (numeric) as answer
        try:
            val = float(raw_answer)
        except ValueError:
            return f"{qtype.capitalize()} answer must be a number.", None
        answer = str(int(val)) if val.is_integer() else str(val)

    elif qtype == "pattern":
        # Expect comma separated items, blanks as _
        items = [p.strip() for p in raw_answer.split(",") if p.strip()]
        if not items:
            return "Pattern answer cannot be empty.", None
        answer = ",".join(items)

    elif qtype == "clock":
        # Expect time in HH:MM format (24h)
        if not re.fullmatch(r"\d{1,2}:\d{2}", raw_answer):
            return "Clock answer must be in HH:MM format.", None
        answer = raw_answer.strip()

    elif qtype == "true_false":
        if raw_answer.lower() not in ("true", "false"):
            return "True/False answer must be 'True' or 'False'.", None
        answer = raw_answer.lower()

    elif qtype == "number_name":
        # Any non‑empty string; could add numeric‑to‑word validation later
        if not raw_answer:
            return "Number‑name answer cannot be empty.", None
        answer = raw_answer.strip().lower()

    elif qtype == "table_fill":
        # Store raw representation; validation can be enhanced later
        if not raw_answer:
            return "Table fill answer cannot be empty.", None
        answer = raw_answer.strip()

    elif qtype == "shape_name":
        answer = raw_answer.lower()
        if answer not in SHAPES:
            return "Shape name must be one of: " + ", ".join(SHAPES) + ".", None

    else:  # matching
        pairs = [p.strip() for p in raw_answer.split(",") if p.strip()]
        cleaned, lefts = [], set()
        for pair in pairs:
            m = re.fullmatch(r"([A-Za-z0-9]+)\s*-\s*([A-Za-z0-9]+)", pair)
            if not m:
                return "Write pairs like 1-C,2-A,3-B (comma between pairs).", None
            left, right = m.group(1).upper(), m.group(2).upper()
            if left in lefts:
                return f"Item {left} appears more than once.", None
            lefts.add(left)
            cleaned.append(f"{left}-{right}")
        if len(cleaned) < 2:
            return "A matching question needs at least 2 pairs.", None
        answer = ",".join(cleaned)

    return None, {
        "question_text": text,
        "question_type": qtype,
        "max_marks": marks,
        "correct_answer": answer,
    }


def render_detail(assessment, form=None):
    """Show the assessment page with its questions and the add-question form."""
    questions = get_db().execute(
        "SELECT * FROM questions WHERE assessment_id = ? ORDER BY question_no",
        (assessment["id"],),
    ).fetchall()
    total_marks = sum(q["max_marks"] for q in questions)
    return render_template(
        "assessment_detail.html",
        assessment=assessment,
        questions=questions,
        total_marks=total_marks,
        question_types=QUESTION_TYPES,
        form=form or {},
    )


@assessments_bp.route("/")
@login_required
def list_assessments():
    rows = get_db().execute(
        """
        SELECT a.*,
          (SELECT COUNT(*) FROM questions q
             WHERE q.assessment_id = a.id) AS question_count,
          (SELECT COALESCE(SUM(q.max_marks), 0) FROM questions q
             WHERE q.assessment_id = a.id) AS total_marks,
          (SELECT COUNT(*) FROM submissions s
             WHERE s.assessment_id = a.id) AS sheet_count
        FROM assessments a
        WHERE a.teacher_id = ?
        ORDER BY a.id DESC
        """,
        (g.teacher["id"],),
    ).fetchall()
    return render_template("assessments.html", assessments=rows)


@assessments_bp.route("/new", methods=("GET", "POST"))
@login_required
def new_assessment():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        subject = request.form.get("subject", "").strip()
        try:
            class_level = int(request.form.get("class_level", ""))
        except ValueError:
            class_level = 0

        error = None
        if not title:
            error = "Please enter a title."
        elif class_level not in (1, 2, 3, 4, 5):
            error = "Class must be between 1 and 5."
        elif subject not in SUBJECTS:
            error = "Please choose a subject."

        if error is None:
            db = get_db()
            cursor = db.execute(
                "INSERT INTO assessments (teacher_id, title, class_level, subject) "
                "VALUES (?, ?, ?, ?)",
                (g.teacher["id"], title, class_level, subject),
            )
            db.commit()
            flash("Assessment created. Now add its questions.", "success")
            return redirect(url_for("assessments.detail",
                                    assessment_id=cursor.lastrowid))
        flash(error, "error")

    return render_template("assessment_new.html", subjects=SUBJECTS)


@assessments_bp.route("/<int:assessment_id>")
@login_required
def detail(assessment_id):
    assessment = get_owned_assessment(assessment_id)
    return render_detail(assessment)


@assessments_bp.route("/<int:assessment_id>/questions/add", methods=("POST",))
@login_required
def add_question(assessment_id):
    assessment = get_owned_assessment(assessment_id)
    error, data = validate_question(request.form)
    if error:
        flash(error, "error")
        return render_detail(assessment, form=request.form)

    db = get_db()
    next_no = db.execute(
        "SELECT COALESCE(MAX(question_no), 0) + 1 AS n FROM questions "
        "WHERE assessment_id = ?",
        (assessment_id,),
    ).fetchone()["n"]
    db.execute(
        "INSERT INTO questions (assessment_id, question_no, question_text, "
        "question_type, max_marks, correct_answer) VALUES (?, ?, ?, ?, ?, ?)",
        (assessment_id, next_no, data["question_text"], data["question_type"],
         data["max_marks"], data["correct_answer"]),
    )
    db.commit()
    flash(f"Question {next_no} added.", "success")
    return redirect(url_for("assessments.detail", assessment_id=assessment_id))


@assessments_bp.route(
    "/<int:assessment_id>/questions/<int:question_id>/edit",
    methods=("GET", "POST"),
)
@login_required
def edit_question(assessment_id, question_id):
    assessment = get_owned_assessment(assessment_id)
    db = get_db()
    question = db.execute(
        "SELECT * FROM questions WHERE id = ? AND assessment_id = ?",
        (question_id, assessment_id),
    ).fetchone()
    if question is None:
        abort(404)

    form = dict(question)
    if request.method == "POST":
        error, data = validate_question(request.form)
        if error:
            flash(error, "error")
            form = request.form
        else:
            db.execute(
                "UPDATE questions SET question_text = ?, question_type = ?, "
                "max_marks = ?, correct_answer = ? WHERE id = ?",
                (data["question_text"], data["question_type"],
                 data["max_marks"], data["correct_answer"], question_id),
            )
            db.commit()
            flash("Question updated.", "success")
            return redirect(url_for("assessments.detail",
                                    assessment_id=assessment_id))

    return render_template(
        "question_edit.html",
        assessment=assessment,
        question=question,
        question_types=QUESTION_TYPES,
        form=form,
    )


@assessments_bp.route(
    "/<int:assessment_id>/questions/<int:question_id>/delete",
    methods=("POST",),
)
@login_required
def delete_question(assessment_id, question_id):
    get_owned_assessment(assessment_id)
    db = get_db()
    db.execute(
        "DELETE FROM questions WHERE id = ? AND assessment_id = ?",
        (question_id, assessment_id),
    )
    # Renumber the remaining questions so there are no gaps (1, 2, 3 ...)
    remaining = db.execute(
        "SELECT id FROM questions WHERE assessment_id = ? ORDER BY question_no",
        (assessment_id,),
    ).fetchall()
    for number, row in enumerate(remaining, start=1):
        db.execute("UPDATE questions SET question_no = ? WHERE id = ?",
                   (number, row["id"]))
    db.commit()
    flash("Question deleted.", "success")
    return redirect(url_for("assessments.detail", assessment_id=assessment_id))


@assessments_bp.route("/<int:assessment_id>/delete", methods=("POST",))
@login_required
def delete_assessment(assessment_id):
    get_owned_assessment(assessment_id)
    db = get_db()
    # Remember the image files first, because the rows are about to disappear
    sheets = db.execute(
        "SELECT image_path FROM submissions WHERE assessment_id = ?",
        (assessment_id,),
    ).fetchall()
    # ON DELETE CASCADE also removes its questions and answer sheet rows
    db.execute("DELETE FROM assessments WHERE id = ?", (assessment_id,))
    db.commit()
    for sheet in sheets:
        remove_submission_files(sheet["image_path"])
    flash("Assessment deleted.", "success")
    return redirect(url_for("assessments.list_assessments"))

# ---------------------------------------------------------------------------
# Upload question paper image for an assessment
# ---------------------------------------------------------------------------
# Upload question paper image for an assessment
@assessments_bp.route(
    "/<int:assessment_id>/upload_paper",
    methods=("POST",),
)
@login_required
def upload_paper(assessment_id):
    """Handle uploading a question‑paper image for an assessment.
    The uploaded file is saved under a local ``uploads/`` directory and the
    relative path is stored in the ``paper_image`` column of the ``assessments``
    table.
    """
    try:
        # Verify ownership
        get_owned_assessment(assessment_id)

        # Ensure a file was submitted
        if "paper_image" not in request.files:
            flash("No file part in the request.", "error")
            return redirect(url_for("assessments.detail", assessment_id=assessment_id))
        file = request.files["paper_image"]
        if file.filename == "":
            flash("No selected file.", "error")
            return redirect(url_for("assessments.detail", assessment_id=assessment_id))

        # Basic validation – allow only image types
        allowed_exts = {"png", "jpg", "jpeg", "gif", "bmp"}
        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower().lstrip('.')
        if ext not in allowed_exts:
            flash("Invalid file type. Allowed: png, jpg, jpeg, gif, bmp.", "error")
            return redirect(url_for("assessments.detail", assessment_id=assessment_id))

        # Save the uploaded image to the uploads folder
        upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)

        # Store the path in the DB (relative to the project root)
        db = get_db()
        db.execute(
            "UPDATE assessments SET paper_image = ? WHERE id = ?",
            (file_path, assessment_id),
        )
        # After storing the paper image, extract questions using Gemini Vision AI
        try:
            # Delete old questions first so re-upload works without UNIQUE constraint errors
            db.execute("DELETE FROM questions WHERE assessment_id = ?", (assessment_id,))
            extracted_questions = _extract_questions_with_tesseract(file_path)
            # Try Gemini if API key is set
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if api_key:
                try:
                    extracted_questions = _extract_questions_with_gemini(file_path, api_key)
                except Exception as gem_e:
                    print("[GEMINI ERROR]", gem_e)
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
                "Extracted " + str(created) + " question(s) from the uploaded paper. Please provide correct answers.",
                "success"
            )
            return redirect(url_for("assessments.edit_answers", assessment_id=assessment_id))
        except Exception as inner_e:
            print("[OCR PROCESSING ERROR]", inner_e)
            traceback.print_exc()
            flash("Error processing uploaded paper image.", "error")
            return redirect(url_for("assessments.detail", assessment_id=assessment_id))

    except Exception as e:
        print("[UPLOAD_PAPER ERROR]", e)
        traceback.print_exc()
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
                traceback.print_exc()
                flash("An error occurred while fixing the database schema.", "error")
                return redirect(url_for("assessments.detail", assessment_id=assessment_id))
        flash("Error uploading paper: " + str(e), "error")
        return redirect(url_for("assessments.detail", assessment_id=assessment_id))


# ---------------------------------------------------------------------------
# Helper: extract questions from image using Gemini Vision AI
# ---------------------------------------------------------------------------
def _extract_questions_with_gemini(image_path, api_key):
    """Use Google Gemini Vision to read a question-paper image and return
    a clean list of question strings."""
    import google.generativeai as genai
    from PIL import Image

    # api_key is passed in from the caller

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    img = Image.open(image_path)

    prompt = """You are a school teacher's assistant. Look at this question paper image carefully.

Extract ONLY the actual questions from this paper (not the header, school name, class, subject, marks, date, or instructions).

Return ONLY a numbered list, one question per line, in this exact format:
1. <question text>
2. <question text>
...

Rules:
- Include every fill-in-the-blank, choose-the-answer, pattern, clock, or other question
- For fill-in-the-blank questions, keep the blank as ___ (three underscores)
- Fix obvious OCR errors in spelling
- Do NOT include: school name, class, subject, time, max marks, student name, section headings like "Q1. Fill in the blanks"
- Do NOT add explanations or extra text - just the numbered list"""

    response = model.generate_content([prompt, img])
    raw = response.text.strip()

    questions = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove leading number and dot/bracket
        m = re.match(r"^\d+[.)\s]+(.+)", line)
        if m:
            q = m.group(1).strip()
            if len(q) > 3:
                questions.append(q)

    print(f"[GEMINI] Extracted {len(questions)} questions")
    return questions


# ---------------------------------------------------------------------------
# Fallback: Tesseract OCR extraction (used when no Gemini API key)
# ---------------------------------------------------------------------------
def _extract_questions_with_tesseract(image_path):
    """Fallback Tesseract-based question extraction."""
    import pytesseract
    import cv2

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image = cv2.imread(image_path)
    if image is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    data = pytesseract.image_to_data(thresh, config="--psm 6 --oem 3",
                                      output_type=pytesseract.Output.DICT)
    lines_dict = {}
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if word == "" or int(data["conf"][i]) < 30:
            continue
        key = (data["block_num"][i], data["line_num"][i])
        lines_dict.setdefault(key, []).append(word)

    lines = [" ".join(lines_dict[k]) for k in sorted(lines_dict.keys())]

    skip_keywords = [
        "school", "sample paper", "class:", "time :", "subject:",
        "name of student", "max. marks", "max marks", "dhamtari",
        "delhi public", "oflue", "date", "nthe blanks", "fill in"
    ]

    questions = []
    for txt in lines:
        txt = txt.strip()
        if len(txt) < 4:
            continue
        low = txt.lower()
        if any(kw in low for kw in skip_keywords):
            continue
        if re.match(r"^\s*[Qq]\d+[.):]", txt):
            continue
        roman_or_num = re.match(
            r"^\s*(i{1,3}v?|vi{0,3}|ix|x{1,3}|\d+)[.)\s]",
            txt, re.IGNORECASE
        )
        if roman_or_num or "_" in txt or txt.endswith("?"):
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
    """GET: show form to enter correct answers for extracted questions.
    POST: save the answers to the database."""
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