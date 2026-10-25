"""Scoring engine: compares OCR-recognized text to the correct answer (Phase 6).

One function per question type, plus a single entry-point score_answer().
All functions return a dict: { is_correct, marks_awarded, feedback }
"""
import re


# ── Helpers ──────────────────────────────────────────────────────────────────

def _norm(text):
    """Lowercase, strip, collapse spaces. Keep letters, digits, hyphen, pipe."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-|]", "", text)
    return re.sub(r"\s+", " ", text)


# ── Shape aliases ─────────────────────────────────────────────────────────────
# When a student draws a circle, OCR often reads it as "O" or "0".
# This maps each shape name → set of OCR outputs we treat as correct.
_SHAPE_ALIASES = {
    "circle":    {"circle", "o", "0", "oval", "round"},
    "square":    {"square", "sq"},
    "triangle":  {"triangle", "tri"},
    "rectangle": {"rectangle", "rect"},
    "oval":      {"oval", "o", "0", "ellipse"},
    "star":      {"star"},
    "heart":     {"heart"},
    "diamond":   {"diamond", "rhombus"},
}


# ── Per-type scorers ─────────────────────────────────────────────────────────

def _score_fill_blank(recognized, correct, max_marks):
    """Accepted answers in 'correct' are separated by |  e.g.  colour|color"""
    rec_norm = _norm(recognized)
    accepted = [_norm(a) for a in correct.split("|") if a.strip()]
    primary = correct.split("|")[0].strip()

    is_correct = bool(rec_norm) and (rec_norm in accepted)
    return {
        "is_correct": is_correct,
        "marks_awarded": float(max_marks) if is_correct else 0.0,
        "feedback": (
            f"Correct! The answer is '{primary}'."
            if is_correct
            else (
                f"Incorrect. Expected '{primary}', but got '{recognized or 'nothing'}'."
                " Practise fill-in-the-blank questions on this topic."
            )
        ),
    }


def _score_numerical(recognized, correct, max_marks):
    """Extract the first number from recognized text and compare to correct."""
    digits = re.sub(r"[^\d.\-]", "", recognized)
    try:
        rec_val = float(digits)
        cor_val = float(correct)
        is_correct = abs(rec_val - cor_val) < 0.01
    except (ValueError, TypeError):
        is_correct = False
        cor_val = correct

    return {
        "is_correct": is_correct,
        "marks_awarded": float(max_marks) if is_correct else 0.0,
        "feedback": (
            f"Correct! The answer is {correct}."
            if is_correct
            else (
                f"Incorrect. The answer is {correct},"
                f" but '{recognized or 'nothing'}' was written."
                " Practise this calculation."
            )
        ),
    }


def _score_shape(recognized, correct, max_marks):
    """Check whether the correct shape was drawn.

    OCR often reads a drawn circle as 'O' or '0', a square as 'sq', etc.
    We use _SHAPE_ALIASES to accept these common misreadings.
    """
    rec_norm = _norm(recognized)        # e.g. "o" or "0"
    cor_norm = _norm(correct)           # e.g. "circle"

    # Check exact match first (e.g. student wrote the word "circle")
    is_correct = bool(rec_norm) and (
        cor_norm in rec_norm
        or rec_norm == cor_norm
    )

    # If not exact, check the alias set for this shape
    if not is_correct:
        aliases = _SHAPE_ALIASES.get(cor_norm, {cor_norm})
        is_correct = bool(rec_norm) and any(
            alias == rec_norm or alias in rec_norm
            for alias in aliases
        )

    return {
        "is_correct": is_correct,
        "marks_awarded": float(max_marks) if is_correct else 0.0,
        "feedback": (
            f"Correct! A {correct} was identified."
            if is_correct
            else (
                f"Incorrect. Expected a {correct}."
                f" The system read '{recognized or 'nothing'}'."
                " Practise drawing this shape clearly."
            )
        ),
    }


def _score_exact(recognized, correct, max_marks):
    """Exact match after normalising both strings.
    Used for pattern, clock, true/false, number_name, table_fill.
    """
    rec_norm = _norm(recognized)
    cor_norm = _norm(correct)
    is_correct = rec_norm == cor_norm and bool(rec_norm)
    return {
        "is_correct": is_correct,
        "marks_awarded": float(max_marks) if is_correct else 0.0,
        "feedback": (
            f"Correct! The answer is '{correct}'."
            if is_correct
            else f"Incorrect. Expected '{correct}', but got '{recognized or 'nothing'}'."
        ),
    }

def _fix_matching_text(text):
    """Recover missing commas between matching pairs.

    OCR often drops the comma so '1-C,2-A,3-B' becomes '1-C2-A3-B'.
    We detect the boundary where a letter is immediately followed by a digit
    (end of one pair, start of the next) and insert a comma there.

    Examples:
      '1-C2-A3-B'    →  '1-C,2-A,3-B'
      '1-C, 2-A, 3-B' →  unchanged (commas already there)
      'A-1B-2C-3'     →  'A-1,B-2,C-3'
    """
    # Insert comma between letter→digit or digit→letter at pair boundaries
    text = re.sub(r"([A-Za-z])(\d)", r"\1,\2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1,\2", text)
    return text


def _score_matching(recognized, correct, max_marks):
    """Partial credit: each correct pair earns marks / total_pairs.

    Both 'recognized' and 'correct' use the format:  1-C,2-A,3-B
    Missing commas in the OCR output are recovered automatically.
    """
    def parse_pairs(text):
        """Return dict {left_upper: right_upper} from text like '1-C,2-A'."""
        text = _fix_matching_text(text)
        pairs = {}
        for m in re.finditer(r"([A-Za-z0-9]+)\s*[-:]\s*([A-Za-z0-9]+)", text):
            left  = m.group(1).upper().strip()
            right = m.group(2).upper().strip()
            # Only keep single-token keys/values to avoid greedy mismatches
            if left and right:
                pairs[left] = right
        return pairs

    correct_pairs = parse_pairs(correct)
    total = len(correct_pairs)

    if total == 0:
        return {
            "is_correct": False,
            "marks_awarded": 0.0,
            "feedback": "Could not read the answer key for this question.",
        }

    rec_pairs = parse_pairs(recognized)
    right_count = sum(
        1 for k, v in correct_pairs.items() if rec_pairs.get(k) == v
    )
    marks = round((right_count / total) * float(max_marks), 2)
    is_correct = right_count == total

    if is_correct:
        fb = f"Correct! All {total} pairs matched."
    elif right_count == 0:
        fb = (
            f"All {total} pairs are incorrect."
            f" Correct pairings: {correct}."
            " Practise the match-the-following activity."
        )
    else:
        fb = (
            f"{right_count} of {total} pairs correct"
            f" (partial marks: {marks}/{max_marks})."
            f" Correct pairings: {correct}."
        )

    return {"is_correct": is_correct, "marks_awarded": marks, "feedback": fb}


# ── Public entry point ────────────────────────────────────────────────────────

def score_answer(question_type, recognized_text, correct_answer, max_marks):
    """Compare recognized_text to correct_answer for the given question type.

    Returns dict: { is_correct: bool, marks_awarded: float, feedback: str }
    """
    rec = (recognized_text or "").strip()

    # ── Scoring map ────────────────────────────────────────────────────────
    # Updated to handle new arithmetic, pattern, clock, true/false, number_name,
    # table_fill, and shape_name question types.
    scorers = {
        "fill_blank": _score_fill_blank,
        "numerical":  _score_numerical,
        "shape":      _score_shape,
        "matching":   _score_matching,
        # Arithmetic types reuse numerical scorer (exact numeric answer expected)
        "addition": _score_numerical,
        "subtraction": _score_numerical,
        "multiplication": _score_numerical,
        "division": _score_numerical,
        # Simple exact‑match types use a generic exact scorer
        "pattern": _score_exact,
        "clock": _score_exact,
        "true_false": _score_exact,
        "number_name": _score_exact,
        "table_fill": _score_exact,
        "shape_name": _score_shape,
    }
    fn = scorers.get(question_type, _score_fill_blank)
    result = fn(rec, correct_answer, max_marks)

    result.setdefault("is_correct", False)
    result.setdefault("marks_awarded", 0.0)
    result.setdefault("feedback", "")
    return result


def overall_feedback(total_awarded, total_marks, wrong_questions):
    """Return a one-line teacher-facing feedback string."""
    pct = (total_awarded / total_marks * 100) if total_marks > 0 else 0

    if pct >= 90:
        grade, tip = "Excellent", "Encourage the student to keep up this standard."
    elif pct >= 70:
        grade, tip = "Good", "Solid effort overall."
    elif pct >= 50:
        grade, tip = "Average", "Some topics need more practice."
    else:
        grade, tip = "Needs Improvement", "The teacher should provide extra support."

    wrong_str = ""
    if wrong_questions:
        wrong_str = " Focus on: " + ", ".join(wrong_questions) + "."

    return (
        f"{grade}. Score {total_awarded:.1f}/{total_marks:.1f}"
        f" ({pct:.0f}%).{wrong_str} {tip}"
    )
