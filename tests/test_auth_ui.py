"""Automated UI tests for the VerifiRAG authentication flow."""

import os
import pytest
from streamlit.testing.v1 import AppTest

# ==========================================
# FIXTURE: CLEAN DATABASE FOR TESTING
# ==========================================
@pytest.fixture(autouse=True)
def isolated_test_db(tmp_path):
    """
    Forces the app to use a temporary SQLite database for tests.
    This ensures your real data/legalverifirag.db is never touched or overwritten.
    """
    test_db_path = tmp_path / "test_auth.db"
    os.environ["DATABASE_PATH"] = str(test_db_path)
    yield
    # The temporary directory and DB are automatically cleaned up by pytest

# ==========================================
# AUTHENTICATION UI TESTS
# ==========================================

def test_successful_account_creation():
    """Verify a user can successfully register a new account."""
    at = AppTest.from_file("src/app.py").run()
    
    # 1. Click the "Register" toggle in the header
    at.button(key="header_login_Register").click().run()
    
    # 2. Fill out the registration form
    at.text_input(key="login_username").input("new_lawyer")
    at.text_input(key="login_password").input("secure_pass_123")
    at.button(key="login_submit").click().run()
    
    # 3. Assert the success message appears
    assert at.success, "Expected a success message to appear."
    assert "Registration successful!" in at.success[0].value
    # 4. Assert the app flips back to the login state automatically
    assert at.session_state["show_register_notice"] is False


def test_duplicate_account_rejection():
    """Verify the system blocks registering an already existing username."""
    at = AppTest.from_file("src/app.py").run()
    
    # 1. Register the first user
    at.button(key="header_login_Register").click().run()
    at.text_input(key="login_username").input("taken_user")
    at.text_input(key="login_password").input("passwordA")
    at.button(key="login_submit").click().run()
    
    # 2. Attempt to register the EXACT SAME username again
    at.button(key="header_login_Register").click().run()
    at.text_input(key="login_username").input("taken_user")
    at.text_input(key="login_password").input("passwordB")
    at.button(key="login_submit").click().run()
    
    # 3. Assert the error message blocks them
    assert at.error, "Expected an error message to appear."
    assert "Username already exists" in at.error[0].value


def test_successful_login():
    """Verify a user can log in with correct credentials and reach the home page."""
    at = AppTest.from_file("src/app.py").run()
    
    # 1. Register a test user so they exist in the DB
    at.button(key="header_login_Register").click().run()
    at.text_input(key="login_username").input("valid_client")
    at.text_input(key="login_password").input("correct_password")
    at.button(key="login_submit").click().run()
    
    # 2. Fill out the Sign In form with those exact credentials
    at.text_input(key="login_username").input("valid_client")
    at.text_input(key="login_password").input("correct_password")
    at.button(key="login_submit").click().run()
    
    # 3. Assert the session state updates and redirects to the Home page
    assert at.session_state["authenticated"] is True
    assert at.session_state["page"] == "home"
    assert at.session_state["user"]["username"] == "valid_client"


def test_rejected_incorrect_password():
    """Verify logging in with a wrong password fails gracefully."""
    at = AppTest.from_file("src/app.py").run()
    
    # 1. Register a test user
    at.button(key="header_login_Register").click().run()
    at.text_input(key="login_username").input("real_user")
    at.text_input(key="login_password").input("real_password")
    at.button(key="login_submit").click().run()
    
    # 2. Try to log in with the right username but WRONG password
    at.text_input(key="login_username").input("real_user")
    at.text_input(key="login_password").input("totally_wrong_password")
    at.button(key="login_submit").click().run()
    
    # 3. Assert they are blocked and get an error
    assert at.error, "Expected an error message."
    assert "Invalid username or password" in at.error[0].value
    assert at.session_state["authenticated"] is False


def test_rejected_nonexistent_account():
    """Verify logging in with an account that doesn't exist fails gracefully."""
    at = AppTest.from_file("src/app.py").run()
    
    # 1. Try to log in immediately without registering
    at.text_input(key="login_username").input("ghost_user")
    at.text_input(key="login_password").input("ghost_password")
    at.button(key="login_submit").click().run()
    
    # 2. Assert they are blocked
    assert at.error, "Expected an error message."
    assert "Invalid username or password" in at.error[0].value
    assert at.session_state["authenticated"] is False