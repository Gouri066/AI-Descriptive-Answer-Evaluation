import os
import sqlite3

from flask import g

from config import Config, BASE_DIR


def get_db():
    """Return the database connection for the current request.
    A connection is created once per request and reused."""
    if "db" not in g:
        os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
        g.db = sqlite3.connect(Config.DATABASE_PATH)
        # Lets us read columns by name: row["title"] instead of row[2]
        g.db.row_factory = sqlite3.Row
        # SQLite ignores foreign keys unless this is switched on every time
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(error=None):
    """Close the connection at the end of each request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create all tables by running database/schema.sql."""
    db = get_db()
    schema_path = os.path.join(BASE_DIR, "database", "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        db.executescript(f.read())
    db.commit()


def init_app(app):
    """Connect the database helpers to the Flask app."""
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()