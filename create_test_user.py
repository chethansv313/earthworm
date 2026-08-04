"""
create_test_user.py
---------------------
One-time helper script to create a login you can actually use to test
the website. Run this AFTER running app.py at least once (so users.db
exists), or it will create the database itself too.

USAGE:
    python create_test_user.py

This creates a user with:
    Username: farmer1
    Password: password123

Change USERNAME / PASSWORD below before running if you want different
test credentials.
"""

from app import app, db, User
from werkzeug.security import generate_password_hash

USERNAME = "farme"
PASSWORD = "passwor23"

with app.app_context():
    db.create_all()  # ensures the table exists

    existing = User.query.filter_by(username=USERNAME).first()
    if existing:
        print(f"User '{USERNAME}' already exists - nothing to do.")
    else:
        new_user = User(
            username=USERNAME,
            password_hash=generate_password_hash(PASSWORD),
        )
        db.session.add(new_user)
        db.session.commit()
        print(f"Created user '{USERNAME}' with password '{PASSWORD}'.")
        print("You can now log in with these credentials at http://127.0.0.1:5000")