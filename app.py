from flask import Flask, render_template, jsonify, g, redirect, url_for
from config import Config
import database
from auth import auth_bp, login_required
from assessments import assessments_bp
from submissions import submissions_bp
from layout import layout_bp
from evaluate import evaluate_bp

# Create the Flask application and load settings from config.py
app = Flask(__name__)
app.config.from_object(Config)

# Connect the database (creates tables on first run)
database.init_app(app)

# Add the login/register/logout routes from auth.py
app.register_blueprint(auth_bp)
app.register_blueprint(submissions_bp)
app.register_blueprint(assessments_bp)
app.register_blueprint(layout_bp)
app.register_blueprint(evaluate_bp)


@app.route("/")
def index():
    """Landing page. Logged-in teachers go straight to the dashboard."""
    if g.teacher is not None:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/dashboard")
@login_required
def dashboard():
    """Teacher dashboard with live counts for the logged-in teacher only."""
    db = database.get_db()
    teacher_id = g.teacher["id"]

    stats = {
        "assessments": db.execute(
            "SELECT COUNT(*) AS c FROM assessments WHERE teacher_id = ?",
            (teacher_id,),
        ).fetchone()["c"],
        "submissions": db.execute(
            "SELECT COUNT(*) AS c FROM submissions s "
            "JOIN assessments a ON a.id = s.assessment_id "
            "WHERE a.teacher_id = ?",
            (teacher_id,),
        ).fetchone()["c"],
        "evaluated": db.execute(
            "SELECT COUNT(*) AS c FROM submissions s "
            "JOIN assessments a ON a.id = s.assessment_id "
            "WHERE a.teacher_id = ? AND s.status = 'evaluated'",
            (teacher_id,),
        ).fetchone()["c"],
        "needs_review": db.execute(
            "SELECT COUNT(*) AS c FROM submissions s "
            "JOIN assessments a ON a.id = s.assessment_id "
            "WHERE a.teacher_id = ? AND s.status = 'needs_review'",
            (teacher_id,),
        ).fetchone()["c"],
    }
    return render_template("dashboard.html", stats=stats)


@app.route("/db-check")
@login_required
def db_check():
    """Temporary route: shows every table and its row count.
    Now login-protected. We will remove this before final submission."""
    db = database.get_db()
    tables = db.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()

    result = {}
    for t in tables:
        name = t["name"]
        count = db.execute(f"SELECT COUNT(*) AS c FROM {name}").fetchone()["c"]
        result[name] = count
    return jsonify(result)


if __name__ == "__main__":
    # debug=True restarts the server automatically when you save changes
    # and shows detailed error pages. Turn it off for final use.
    app.run(debug=True)