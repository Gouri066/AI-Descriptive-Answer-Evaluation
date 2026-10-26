"""Quick diagnostic: show what OCR actually reads for the latest submission."""
import sqlite3, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config

db = sqlite3.connect(Config.DATABASE_PATH)
db.row_factory = sqlite3.Row

print("=== Recent Submissions ===")
subs = db.execute(
    "SELECT id, student_name, total_score, status FROM submissions ORDER BY id DESC LIMIT 5"
).fetchall()
for s in subs:
    print(f"  ID={s['id']}  name={s['student_name']}  score={s['total_score']}  status={s['status']}")

if not subs:
    print("No submissions found.")
    sys.exit()

latest_id = subs[0]["id"]
print(f"\n=== Answers for latest submission (ID={latest_id}) ===")
rows = db.execute(
    """
    SELECT q.question_no, q.question_type, q.correct_answer, q.max_marks,
           a.recognized_text, a.marks_awarded, a.confidence, a.is_correct
    FROM answers a
    JOIN questions q ON q.id = a.question_id
    WHERE a.submission_id = ?
    ORDER BY q.question_no
    """,
    (latest_id,),
).fetchall()

if not rows:
    print("  No answers stored yet for this submission.")
else:
    print(f"  {'Q':>2}  {'type':<14} {'correct':<16} {'OCR read':<20} {'marks':>5}  {'conf':>5}  {'ok'}")
    for r in rows:
        print(
            f"  {r['question_no']:>2}  {r['question_type']:<14} "
            f"{str(r['correct_answer'])[:15]:<16} "
            f"{str(r['recognized_text'])[:19]:<20} "
            f"{r['marks_awarded']:>5.1f}  {r['confidence']:>5.1f}  {'YES' if r['is_correct'] else 'NO'}"
        )

print("\n=== Questions with NO region (answer box missing) ===")
no_box = db.execute(
    """
    SELECT q.question_no, q.question_type, q.correct_answer
    FROM questions q
    JOIN submissions s ON s.assessment_id = q.assessment_id
    WHERE s.id = ? AND q.region_x IS NULL
    ORDER BY q.question_no
    """,
    (latest_id,),
).fetchall()
if no_box:
    for q in no_box:
        print(f"  Q{q['question_no']} ({q['question_type']}): correct={q['correct_answer']}")
else:
    print("  All questions have answer boxes defined.")
