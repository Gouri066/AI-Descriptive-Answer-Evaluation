import sys
sys.path.insert(0, r'c:\Users\Gouri K\OneDrive\Desktop\Major_project_2026-27')
from modules import scoring

tests = [
    ('fill_blank', 'Februery',  'February|february', 1),
    ('fill_blank', '8O',        '80',                1),
    ('fill_blank', 'Novembar',  'November',          1),
    ('fill_blank', '3O',        '30',                1),
    ('fill_blank', 'Tue',       'True',              1),
    ('fill_blank', 'februry',   'February',          1),
    ('fill_blank', 'squaer',    'square',            1),
    ('numerical',  '72',        '72',                1),
    ('numerical',  '7Z',        '72',                1),
    ('numerical',  '8O',        '80',                1),
]
print(f"{'Type':<12} {'OCR':<15} {'Correct':<15} {'Result':<6} {'Marks'}")
print("-" * 60)
for qtype, ocr_text, correct, marks in tests:
    r = scoring.score_answer(qtype, ocr_text, correct, marks)
    verdict = "PASS" if r["is_correct"] else "FAIL"
    print(f"{qtype:<12} {ocr_text!r:<15} {correct!r:<15} {verdict:<6} {r['marks_awarded']}")
