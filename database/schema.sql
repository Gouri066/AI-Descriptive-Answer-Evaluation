    -- Add column for uploaded question paper image (if not already present)\n    ALTER TABLE assessments ADD COLUMN IF NOT EXISTS paper_image TEXT;
CREATE TABLE IF NOT EXISTS teachers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name     TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- An assessment (test) created by a teacher
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    class_level INTEGER NOT NULL CHECK (class_level BETWEEN 1 AND 5),
    subject TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    paper_image TEXT,  -- optional path to uploaded question paper image
    FOREIGN KEY (teacher_id) REFERENCES teachers(id) ON DELETE CASCADE
);

-- Questions and the answer key for an assessment.
-- region_* columns store where the answer box is on the sheet,
-- as fractions of image width/height (0.0 to 1.0). Used from Phase 5.
CREATE TABLE IF NOT EXISTS questions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id  INTEGER NOT NULL,
    question_no    INTEGER NOT NULL,
    question_text  TEXT NOT NULL,
    question_type  TEXT NOT NULL CHECK (
        question_type IN ('fill_blank', 'numerical', 'shape', 'matching')
    ),
    max_marks      REAL NOT NULL CHECK (max_marks > 0),
    correct_answer TEXT NOT NULL,
    region_x       REAL,
    region_y       REAL,
    region_w       REAL,
    region_h       REAL,
    FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE,
    UNIQUE (assessment_id, question_no)
);

-- One uploaded answer sheet of one student
CREATE TABLE IF NOT EXISTS submissions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id        INTEGER NOT NULL,
    student_name         TEXT NOT NULL,
    student_roll_no      TEXT NOT NULL,
    image_path           TEXT NOT NULL,
    processed_image_path TEXT,
    status               TEXT NOT NULL DEFAULT 'uploaded' CHECK (
        status IN ('uploaded', 'processed', 'evaluated', 'needs_review')
    ),
    total_score          REAL,
    overall_feedback     TEXT,
    uploaded_at          TEXT NOT NULL DEFAULT (datetime('now')),
    evaluated_at         TEXT,
    FOREIGN KEY (assessment_id) REFERENCES assessments(id) ON DELETE CASCADE
);

-- Per-question result inside a submission
CREATE TABLE IF NOT EXISTS answers (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id         INTEGER NOT NULL,
    question_id           INTEGER NOT NULL,
    recognized_text       TEXT,
    confidence            REAL,
    marks_awarded         REAL,
    is_correct            INTEGER,
    needs_review          INTEGER NOT NULL DEFAULT 0,
    teacher_override_text TEXT,
    feedback              TEXT,
    FOREIGN KEY (submission_id) REFERENCES submissions(id) ON DELETE CASCADE,
    FOREIGN KEY (question_id)   REFERENCES questions(id)   ON DELETE CASCADE,
    UNIQUE (submission_id, question_id)
);