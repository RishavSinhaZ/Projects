import time
import json
import random
import string
from datetime import datetime

MASTER_PASSWORD = "Skyler@123"
MAX_ATTEMPTS = 3
LOCKOUT_TIME = 30  # seconds

def check_master_password():
    """Check master password with attempts + lockout."""
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        pwd = input("Enter Master Password: ")
        if pwd == MASTER_PASSWORD:
            log_action("Login Success")
            return True
        else:
            attempts += 1
            print(f"❌ Wrong Password! Attempts left: {MAX_ATTEMPTS - attempts}")
    print(f"🔒 Too many wrong attempts! Locked for {LOCKOUT_TIME} seconds.")
    time.sleep(LOCKOUT_TIME)
    return False


def log_action(action: str):
    """Log actions to a file with timestamp."""
    with open("activity.log", "a") as f:
        f.write(f"{datetime.now()} - {action}\n")


def check_password_strength(password: str) -> str:
    """Basic password strength checker."""
    length = len(password) >= 8
    upper = any(c.isupper() for c in password)
    lower = any(c.islower() for c in password)
    digit = any(c.isdigit() for c in password)
    special = any(c in string.punctuation for c in password)

    if all([length, upper, lower, digit, special]):
        return "✅ Strong"
    elif length and (upper or lower) and (digit or special):
        return "⚠️ Medium"
    else:
        return "❌ Weak"


def generate_otp():
    """Simulate 2FA with a 6-digit OTP."""
    otp = ''.join(random.choices(string.digits, k=6))
    print(f"📲 Human Verification Code: {otp}")  # In real-world, send via email/SMS
    user_otp = input("Enter the code: ")
    if user_otp == otp:
        log_action("OTP Verified")
        return True
    else:
        print("❌ Invalid OTP!")
        return False
