"""Evaluate blueprint (Phase 6).

Two routes:
  POST /evaluate/<submission_id>         - run OCR + scoring, save to DB
  GET  /evaluate/<submission_id>/result  - show the detailed result page

The gray image (*_gray.png) is used for OCR because it keeps ink contrast
without the heavy binarization that can break thin letters.
"""
import os
import result as result_engine
from datetime import datetime

from flask import (
    Blueprint, flash, g, redirect, render_template, url_for
)

from auth import login_required
from config import Config
from database import get_db
from modules import ocr as ocr_mod
from modules import scoring
from submissions import get_owned_submission

evaluate_bp = Blueprint("evaluate", __name__, url_prefix="/evaluate")


def _gray_path(binary_filename):
    """Turn  abc_binary.png  into  abc_gray.png ."""
    return binary_filename.replace("_binary.png", "_gray.png")


def _full(filename):
    return os.path.join(Config.UPLOAD_FOLDER, filename)


@evaluate_bp.route("/<int:submission_id>", methods=("POST",))
@login_required
def run(submission_id):
    """Run OCR + scoring on every question of this submission."""
    print(f"[DEBUG] Evaluation started for submission {submission_id}")

    submission = get_owned_submission(submission_id)
    db = get_db()

    questions = db.execute(
        "SELECT * FROM questions WHERE assessment_id = ? ORDER BY question_no",
        (submission["assessment_id"],),
    ).fetchall()

    if not questions:
        flash("This assessment has no questions.", "error")
        return redirect(url_for(
            "submissions.detail", submission_id=submission_id
        ))

    has_any_region = any(q["region_x"] is not None for q in questions)
    if not has_any_region:
        flash("No answer boxes are defined yet. Define Answer Boxes first.", "error")
        return redirect(url_for(
            "submissions.detail", submission_id=submission_id
        ))

    binary_name = submission["processed_image_path"]
    if not binary_name:
        flash("Preprocess the sheet before evaluating.", "error")
        return redirect(url_for(
            "submissions.detail", submission_id=submission_id
        ))

    gray_name = _gray_path(binary_name)
    gray_full = _full(gray_name)

    if not os.path.exists(gray_full):
        flash("Gray image not found. Re-run preprocessing.", "error")
        return redirect(url_for(
            "submissions.detail", submission_id=submission_id
        ))

    total_possible = 0.0
    total_awarded = 0.0
    any_needs_review = False
    wrong_q_labels = []

    print(f"[DEBUG] Processing {len(questions)} questions")

    for q in questions:
        total_possible += q["max_marks"]

        # ocr_result = ocr_mod.ocr_question(gray_full, q)
        print(
            f"[DEBUG] Starting OCR for Q{q['question_no']}",
            flush=True
        )

        ocr_result = ocr_mod.ocr_question(gray_full, q)

        print(
            f"[DEBUG] OCR finished for Q{q['question_no']}: "
            f"{ocr_result['recognized_text']!r}",
            flush=True
        )
        score_result = scoring.score_answer(
            q["question_type"],
            ocr_result["recognized_text"],
            q["correct_answer"],
            q["max_marks"],
        )

        # Incorrect answers may be caused by OCR misreading handwriting.
        # if not score_result["is_correct"]:
        #     ocr_result["needs_review"] = True
        if not score_result["is_correct"] or not ocr_result["recognized_text"].strip():
            ocr_result["needs_review"] = True


        if ocr_result["needs_review"]:
            any_needs_review = True

        total_awarded += score_result["marks_awarded"]

        if not score_result["is_correct"]:
            wrong_q_labels.append(
                f"Q{q['question_no']} "
                f"({q['question_type'].replace('_', ' ')})"
            )

        print(
            f"[RESULT CHECK] Q{q['question_no']} | "
            f"expected={q['correct_answer']!r} | "
            f"OCR={ocr_result['recognized_text']!r} | "
            f"marks={score_result['marks_awarded']}/{q['max_marks']} | "
            f"correct={score_result['is_correct']}"
        )

        # Save EVERY answer, not only answers needing review.
        db.execute(
            """
            INSERT INTO answers
            (submission_id, question_id, recognized_text, confidence,
                marks_awarded, is_correct, needs_review, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(submission_id, question_id) DO UPDATE SET
                recognized_text = excluded.recognized_text,
                confidence = excluded.confidence,
                marks_awarded = excluded.marks_awarded,
                is_correct = excluded.is_correct,
                needs_review = excluded.needs_review,
                feedback = excluded.feedback
            """,
            (
                submission_id,
                q["id"],
                ocr_result["recognized_text"],
                ocr_result["confidence"],
                score_result["marks_awarded"],
                1 if score_result["is_correct"] else 0,
                1 if ocr_result["needs_review"] else 0,
                score_result["feedback"],
            ),
        )

    print("[DEBUG] All questions processed; updating submission")

    #new_status = "needs_review" if any_needs_review else "evaluated"
    new_status = "evaluated"
    fb = scoring.overall_feedback(
        total_awarded, total_possible, wrong_q_labels
    )

    db.execute(
        """
        UPDATE submissions
        SET status = ?, total_score = ?, overall_feedback = ?, evaluated_at = ?
        WHERE id = ?
        """,
        (
            new_status,
            round(total_awarded, 2),
            fb,
            datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            submission_id,
        ),
    )

    print("[DEBUG] Committing results")
    db.commit()
    print("[DEBUG] Database commit completed")

    msg = (
        f"Evaluation complete. Score: "
        f"{total_awarded:.1f} / {total_possible:.1f}."
    )
    if any_needs_review:
        msg += " Some answers need teacher review."

    flash(msg, "success")

    print("[DEBUG] Redirecting to result page")
    return redirect(url_for(
        "evaluate.result", submission_id=submission_id
    ))



@evaluate_bp.route("/<int:submission_id>/result")
@login_required
def result(submission_id):
    """Show results, teacher recommendations, worksheets and feedback."""
    submission = get_owned_submission(submission_id)
    db = get_db()

    question_answers = db.execute(
        """
        SELECT
            q.id AS question_id,
            q.question_no,
            q.question_text,
            q.question_type,
            q.max_marks,
            q.correct_answer,
            q.region_x,
            a.recognized_text,
            a.confidence,
            a.marks_awarded,
            a.is_correct,
            a.needs_review AS answer_needs_review,
            a.feedback,
            a.teacher_override_text
        FROM questions q
        LEFT JOIN answers a
            ON a.question_id = q.id
            AND a.submission_id = ?
        WHERE q.assessment_id = ?
        ORDER BY q.question_no
        """,
        (submission_id, submission["assessment_id"]),
    ).fetchall()

    return render_template(
    "evaluate_result.html",
    submission=submission,
    question_answers=question_answers,
)


@evaluate_bp.route("/<int:submission_id>/personalized")
@login_required
def personalized(submission_id):
    submission = get_owned_submission(submission_id)
    db = get_db()

    
    question_answers = db.execute(
        """
        SELECT
            q.question_no,
            q.question_text,
            q.question_type,
            q.max_marks,
            q.correct_answer,
            a.marks_awarded,
            a.is_correct,
            a.feedback
        FROM questions q
        LEFT JOIN answers a
            ON a.question_id = q.id
            AND a.submission_id = ?
        WHERE q.assessment_id = ?
        ORDER BY q.question_no
        """,
        (submission_id, submission["assessment_id"]),
    ).fetchall()

    qa_rows = []

    for qa in question_answers:
        qa_rows.append({
            "question_no": qa["question_no"],
            "question": qa["question_text"] or "",
            "question_type": qa["question_type"] or "",
            "correct_answer": qa["correct_answer"] or "",
            "marks_awarded": qa["marks_awarded"] or 0,
            "max_marks": qa["max_marks"] or 0,
            "is_correct": bool(qa["is_correct"]),
            "feedback": qa["feedback"] or "",
        })

    report = result_engine.build_phase9(
        qa_rows,
        student_name=submission["student_name"],
    )


    return render_template(
        "personalized_report.html",
        submission=submission,
        report=report
    )
