"""Layout blueprint (Phase 6).

Lets the teacher draw answer boxes on a processed sheet image.
Each box is stored as fractions (0.0-1.0) of the image size so that
the same coordinates work whatever size the sheet was scanned at.

Routes
------
GET  /layout/<assessment_id>                 - show the interactive editor
POST /layout/<assessment_id>/save_region     - save one box (JSON API)
"""
from flask import (
    Blueprint, g, jsonify, redirect, render_template, request, url_for
)

from auth import login_required
from assessments import get_owned_assessment
from database import get_db

layout_bp = Blueprint("layout", __name__, url_prefix="/layout")


@layout_bp.route("/<int:assessment_id>")
@login_required
def editor(assessment_id):
    """Show the layout editor for this assessment."""
    assessment = get_owned_assessment(assessment_id)
    db = get_db()

    questions = db.execute(
        "SELECT * FROM questions WHERE assessment_id = ? ORDER BY question_no",
        (assessment_id,),
    ).fetchall()

    # Use the most recent processed submission as the reference image
    ref_submission = db.execute(
        """SELECT id, processed_image_path
           FROM submissions
           WHERE assessment_id = ? AND processed_image_path IS NOT NULL
           ORDER BY id DESC LIMIT 1""",
        (assessment_id,),
    ).fetchone()

    return render_template(
        "layout_editor.html",
        assessment=assessment,
        questions=questions,
        ref_submission=ref_submission,
    )


@layout_bp.route("/<int:assessment_id>/save_region", methods=("POST",))
@login_required
def save_region(assessment_id):
    """JSON API: save (or clear) an answer box for one question."""
    get_owned_assessment(assessment_id)      # 404 if wrong teacher
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Expected JSON body"}), 400

    question_id = data.get("question_id")
    rx = data.get("region_x")
    ry = data.get("region_y")
    rw = data.get("region_w")
    rh = data.get("region_h")

    # Verify question belongs to this assessment
    db = get_db()
    q = db.execute(
        "SELECT id FROM questions WHERE id = ? AND assessment_id = ?",
        (question_id, assessment_id),
    ).fetchone()
    if q is None:
        return jsonify({"error": "Question not found"}), 404

    db.execute(
        "UPDATE questions SET region_x=?, region_y=?, region_w=?, region_h=? WHERE id=?",
        (rx, ry, rw, rh, question_id),
    )
    db.commit()
    return jsonify({"ok": True})
