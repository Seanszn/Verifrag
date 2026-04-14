"""Password hashing helpers for the server-side auth flow.

New passwords are stored with PBKDF2-HMAC. Verification also accepts legacy
bcrypt hashes produced by the older client-side auth path so existing users
can still sign in after auth moves behind the API.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os

try:
    import bcrypt
except ImportError:  # pragma: no cover - optional compatibility path
    bcrypt = None


_ITERATIONS = 120_000
_SALT_BYTES = 16


def hash_password(password: str) -> str:
    """Hash a password as pbkdf2_sha256$iterations$salt$hash."""
    if not password:
        raise ValueError("Password must not be empty.")

    salt = os.urandom(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _ITERATIONS,
    )
    salt_b64 = base64.b64encode(salt).decode("ascii")
    digest_b64 = base64.b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${_ITERATIONS}${salt_b64}${digest_b64}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a serialized PBKDF2 or legacy bcrypt hash."""
    if password_hash.startswith("$2"):
        if bcrypt is None:
            return False
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )

    try:
        algorithm, iterations, salt_b64, digest_b64 = password_hash.split("$", 3)
    except ValueError:
        return False

    if algorithm != "pbkdf2_sha256":
        return False

    salt = base64.b64decode(salt_b64.encode("ascii"))
    expected = base64.b64decode(digest_b64.encode("ascii"))
    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        int(iterations),
    )
    return hmac.compare_digest(actual, expected)
