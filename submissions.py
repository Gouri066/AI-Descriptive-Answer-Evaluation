import os
import time
import uuid

from flask import (
    Blueprint, abort, current_app, flash, g, redirect, render_template,
    request, send_from_directory, url_for
)

from auth import login_required
from config import Config
from database import get_db
from modules import preprocess

submissions_bp = Blueprint("submissions", __name__, url_prefix="/submissions")


def allowed_extension(filename):
    """Return the lowercase extension if allowed (png/jpg/jpeg), else None."""
    if "." not in filename:
        return None
    ext = filename.rsplit(".", 1)[1].lower()
    return ext if ext in Config.ALLOWED_EXTENSIONS else None


def upload_path(filename):
    return os.path.join(Config.UPLOAD_FOLDER, filename)


def remove_submission_files(image_filename):
    """Delete an uploaded sheet and the two processed files made from it."""
    if not image_filename:
        return
    binary_name, gray_name = preprocess.processed_names(image_filename)
    for name in (image_filename, binary_name, gray_name):
        try:
            os.remove(upload_path(os.path.basename(name)))
        except FileNotFoundError:
            pass


def get_owned_submission(submission_id):
    """Fetch a submission ONLY if its assessment belongs to this teacher."""
    submission = get_db().execute(
        """
        SELECT s.*, a.title AS assessment_title, a.class_level, a.subject
        FROM submissions s
        JOIN assessments a ON a.id = s.assessment_id
        WHERE s.id = ? AND a.teacher_id = ?
        """,
        (submission_id, g.teacher["id"]),
    ).fetchone()
    if submission is None:
        abort(404)
    return submission


def skew_message(info):
    angle = info["skew_corrected_degrees"]
    if angle == 0:
        return "No tilt correction was needed."
    return f"Tilt corrected by {angle:+.1f} degrees."


@submissions_bp.app_errorhandler(413)
def file_too_large(error):
    flash("That file is too large. The maximum size is 10 MB.", "error")
    return redirect(url_for("submissions.upload"))


@submissions_bp.route("/")
@login_required
def list_submissions():
    rows = get_db().execute(
        """
        SELECT s.id, s.student_name, s.student_roll_no, s.status,
               s.uploaded_at, a.title AS assessment_title
        FROM submissions s
        JOIN assessments a ON a.id = s.assessment_id
        WHERE a.teacher_id = ?
        ORDER BY s.id DESC
        """,
        (g.teacher["id"],),
    ).fetchall()
    return render_template("submissions.html", submissions=rows)


@submissions_bp.route("/upload", methods=("GET", "POST"))
@login_required
def upload():
    db = get_db()
    assessments = db.execute(
        "SELECT id, title, class_level FROM assessments "
        "WHERE teacher_id = ? ORDER BY id DESC",
        (g.teacher["id"],),
    ).fetchall()
    selected_id = request.args.get("assessment_id", type=int)

    if request.method == "POST":
        assessment_id = request.form.get("assessment_id", type=int)
        student_name = request.form.get("student_name", "").strip()
        roll_no = request.form.get("student_roll_no", "").strip()
        file = request.files.get("sheet")
        selected_id = assessment_id

        error, ext, image, data = None, None, None, None

        owned = None
        if assessment_id is not None:
            owned = db.execute(
                "SELECT id FROM assessments WHERE id = ? AND teacher_id = ?",
                (assessment_id, g.teacher["id"]),
            ).fetchone()

        if owned is None:
            error = "Please choose one of your assessments."
        elif not student_name or not roll_no:
            error = "Please enter the student name and roll number."
        elif file is None or file.filename == "":
            error = "Please choose an image file of the answer sheet."
        else:
            # New check: ensure the assessment has at least one question
            qcount = db.execute(
                "SELECT COUNT(*) as cnt FROM questions WHERE assessment_id = ?",
                (assessment_id,)
            ).fetchone()["cnt"]
            if qcount == 0:
                error = "This assessment has no questions. Add questions before uploading answer sheets."
            else:
                ext = allowed_extension(file.filename)
                if ext is None:
                    error = "Only PNG, JPG and JPEG images are allowed."

        if error is None:
            data = file.read()
            image = preprocess.decode_image(data)
            if image is None:
                error = "That file is not a valid image."

        if error is None:
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            # A random file name avoids clashes and unsafe names
            filename = f"{uuid.uuid4().hex}.{ext}"
            with open(upload_path(filename), "wb") as f:
                f.write(data)

            binary_name, gray_name = preprocess.processed_names(filename)
            try:
                info = preprocess.process_image(
                    image, upload_path(binary_name), upload_path(gray_name)
                )
            except ValueError as problem:
                remove_submission_files(filename)
                error = str(problem)
            except Exception:
                current_app.logger.exception("Image processing failed")
                remove_submission_files(filename)
                error = "Could not process this image. Try another photo or scan."
            else:
                cursor = db.execute(
                    "INSERT INTO submissions (assessment_id, student_name, "
                    "student_roll_no, image_path, processed_image_path, status) "
                    "VALUES (?, ?, ?, ?, ?, 'processed')",
                    (assessment_id, student_name, roll_no, filename, binary_name),
                )
                db.commit()
                flash("Answer sheet uploaded and processed. " + skew_message(info),
                      "success")
                return redirect(url_for("submissions.detail",
                                        submission_id=cursor.lastrowid))

        flash(error, "error")

    return render_template("upload.html", assessments=assessments,
                           selected_id=selected_id, form=request.form)


@submissions_bp.route("/<int:submission_id>")
@login_required
def detail(submission_id):
    submission = get_owned_submission(submission_id)
    # "stamp" makes the browser reload images after re-processing
    return render_template("submission_detail.html", submission=submission,
                           stamp=int(time.time()))


@submissions_bp.route("/<int:submission_id>/image/<kind>")
@login_required
def image(submission_id, kind):
    submission = get_owned_submission(submission_id)
    if kind == "original":
        filename = submission["image_path"]
    elif kind == "processed":
        filename = submission["processed_image_path"]
    else:
        abort(404)
    if not filename:
        abort(404)
    return send_from_directory(Config.UPLOAD_FOLDER, filename)


@submissions_bp.route("/<int:submission_id>/reprocess", methods=("POST",))
@login_required
def reprocess(submission_id):
    submission = get_owned_submission(submission_id)
    original = preprocess.load_image(upload_path(submission["image_path"]))
    if original is None:
        flash("The original image file could not be read.", "error")
        return redirect(url_for("submissions.detail", submission_id=submission_id))

    binary_name, gray_name = preprocess.processed_names(submission["image_path"])
    try:
        info = preprocess.process_image(
            original, upload_path(binary_name), upload_path(gray_name)
        )
    except Exception as problem:
        flash(f"Processing failed: {problem}", "error")
    else:
        db = get_db()
        db.execute(
            "UPDATE submissions SET processed_image_path = ?, status = 'processed' "
            "WHERE id = ?",
            (binary_name, submission_id),
        )
        db.commit()
        flash("Preprocessing re-run. " + skew_message(info), "success")
    return redirect(url_for("submissions.detail", submission_id=submission_id))


@submissions_bp.route("/<int:submission_id>/delete", methods=("POST",))
@login_required
def delete(submission_id):
    submission = get_owned_submission(submission_id)
    db = get_db()
    db.execute("DELETE FROM submissions WHERE id = ?", (submission_id,))
    db.commit()
    remove_submission_files(submission["image_path"])
    flash("Answer sheet deleted.", "success")
    return redirect(url_for("submissions.list_submissions"))