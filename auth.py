import functools
import sqlite3

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

from config import Config
from database import get_db

# A Blueprint is a group of related routes kept in their own file
auth_bp = Blueprint("auth", __name__)


@auth_bp.before_app_request
def load_logged_in_teacher():
    """Runs before every request. Puts the logged-in teacher (or None)
    into g.teacher so every page and route can use it."""
    teacher_id = session.get("teacher_id")
    if teacher_id is None:
        g.teacher = None
    else:
        g.teacher = get_db().execute(
            "SELECT id, full_name, email FROM teachers WHERE id = ?",
            (teacher_id,),
        ).fetchone()


def login_required(view):
    """Decorator: put @login_required under a route to protect it.
    Visitors who are not logged in are sent to the login page."""
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.teacher is None:
            flash("Please log in to continue.", "error")
            return redirect(url_for("auth.login"))
        return view(**kwargs)
    return wrapped_view


@auth_bp.route("/register", methods=("GET", "POST"))
def register():
    if g.teacher is not None:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        code = request.form.get("signup_code", "").strip()

        error = None
        if not full_name or not email or not password:
            error = "Please fill in all the fields."
        elif "@" not in email or "." not in email:
            error = "Please enter a valid email address."
        elif len(password) < 8:
            error = "Password must be at least 8 characters long."
        elif password != confirm:
            error = "The two passwords do not match."
        elif code != Config.SIGNUP_CODE:
            error = "Invalid teacher signup code."

        if error is None:
            db = get_db()
            try:
                db.execute(
                    "INSERT INTO teachers (full_name, email, password_hash) "
                    "VALUES (?, ?, ?)",
                    (full_name, email, generate_password_hash(password)),
                )
                db.commit()
            except sqlite3.IntegrityError:
                # The email column is UNIQUE, so a duplicate raises this error
                error = "This email is already registered."
            else:
                flash("Account created. Please log in.", "success")
                return redirect(url_for("auth.login"))

        flash(error, "error")

    return render_template("register.html")


@auth_bp.route("/login", methods=("GET", "POST"))
def login():
    if g.teacher is not None:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        teacher = get_db().execute(
            "SELECT * FROM teachers WHERE email = ?", (email,)
        ).fetchone()

        # Same message for "no such email" and "wrong password" on purpose,
        # so nobody can use the form to find out which emails exist.
        if teacher is None or not check_password_hash(
            teacher["password_hash"], password
        ):
            flash("Invalid email or password.", "error")
        else:
            session.clear()
            session["teacher_id"] = teacher["id"]
            return redirect(url_for("dashboard"))

    return render_template("login.html")


@auth_bp.route("/logout", methods=("POST",))
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))