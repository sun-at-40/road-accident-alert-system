from functools import wraps
from pathlib import Path
import sqlite3

from flask import abort, g, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


class UserStore:
    def __init__(self, database_path):
        self.database_path = Path(database_path)

    def initialize(self):
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT,
                    google_subject TEXT UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def create_email_user(self, email, display_name, password):
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (email, display_name, password_hash)
                    VALUES (?, ?, ?)
                    """,
                    (email, display_name, generate_password_hash(password)),
                )
        except sqlite3.IntegrityError:
            return None
        return self.get_by_id(cursor.lastrowid)

    def authenticate_email_user(self, email, password):
        user = self.get_by_email(email)
        if user is None or user["password_hash"] is None:
            return None
        return user if check_password_hash(user["password_hash"], password) else None

    def find_or_create_google_user(self, subject, email, display_name):
        user = self.get_by_google_subject(subject)
        if user is not None:
            return user

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
            if existing is not None:
                connection.execute(
                    "UPDATE users SET google_subject = ? WHERE id = ?",
                    (subject, existing["id"]),
                )
                return self.get_by_id(existing["id"])

            cursor = connection.execute(
                """
                INSERT INTO users (email, display_name, google_subject)
                VALUES (?, ?, ?)
                """,
                (email, display_name, subject),
            )
        return self.get_by_id(cursor.lastrowid)

    def get_by_id(self, user_id):
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()

    def get_by_email(self, email):
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()

    def get_by_google_subject(self, subject):
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM users WHERE google_subject = ?", (subject,)
            ).fetchone()

    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection


def login_user(user):
    session.clear()
    session["user_id"] = user["id"]


def logout_user():
    session.clear()


def load_current_user(user_store):
    user_id = session.get("user_id")
    g.current_user = user_store.get_by_id(user_id) if user_id is not None else None
    if user_id is not None and g.current_user is None:
        session.clear()


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.get("current_user") is None:
            return redirect(url_for("login", next=request.full_path))
        return view(*args, **kwargs)

    return wrapped_view


def safe_next_url(candidate):
    if candidate and candidate.startswith("/") and not candidate.startswith("//"):
        return candidate
    return url_for("index")
