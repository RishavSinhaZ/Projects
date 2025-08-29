import json
import os
from cryptography.fernet import Fernet

DATA_FILE = "passwords.json"
KEY_FILE = "key.key"

# ------------------ Encryption Helpers ------------------ #
def load_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as key_file:
            key_file.write(key)
    else:
        with open(KEY_FILE, "rb") as key_file:
            key = key_file.read()
    return Fernet(key)

fernet = load_key()

def encrypt_data(data: str) -> str:
    return fernet.encrypt(data.encode()).decode()

def decrypt_data(data: str) -> str:
    return fernet.decrypt(data.encode()).decode()

# ------------------ Storage Helpers ------------------ #
def load_data():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ------------------ Password Manager ------------------ #
def add_password(service, username, password):
    data = load_data()

    if service not in data:
        data[service] = []

    # Prevent duplicate usernames under same service
    for account in data[service]:
        if account["username"] == username:
            print(f"⚠️ Username '{username}' already exists under {service}!")
            return

    encrypted_pass = encrypt_data(password)
    data[service].append({"username": username, "password": encrypted_pass})
    save_data(data)
    print(f"✅ Password for {username} added under {service}!")

def view_password(service):
    data = load_data()
    if service not in data or len(data[service]) == 0:
        print(f"❌ No accounts found for {service}")
        return

    print(f"\n🔑 Accounts under {service}:")
    for account in data[service]:
        try:
            decrypted_pass = decrypt_data(account["password"])
            print(f" 👤 {account['username']} | 🔐 {decrypted_pass}")
        except Exception:
            print(f"❌ Error decrypting password for {account['username']}")

def delete_password(service, username):
    data = load_data()
    if service not in data:
        print(f"❌ No service named {service}")
        return

    updated_accounts = [acc for acc in data[service] if acc["username"] != username]

    if len(updated_accounts) == len(data[service]):
        print(f"❌ Username '{username}' not found under {service}")
        return

    data[service] = updated_accounts
    if not data[service]:
        del data[service]  # remove service if no accounts left
    save_data(data)
    print(f"🗑️ Deleted {username} from {service}")
    
def edit_password(service, username, new_username=None, new_password=None):
    data = load_data()

    if service not in data:
        print(f"❌ No service named {service}")
        return

    for account in data[service]:
        if account["username"] == username:
            if new_username:
                account["username"] = new_username
            if new_password:
                account["password"] = encrypt_data(new_password)

            save_data(data)
            print(f"✏️ Updated account '{username}' in {service}")
            return

    print(f"❌ Username '{username}' not found under {service}")

def list_services():
    data = load_data()
    if not data:
        print("📭 No services stored yet.")
        return
    print("\n📂 Stored Services:")
    for service in data.keys():
        print(f" - {service} ({len(data[service])} accounts)")

def export_passwords(filename="export.json"):
    data = load_data()
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"📤 Passwords exported to {filename}")

def import_passwords(filename="export.json"):
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found!")
        return

    with open(filename, "r") as f:
        imported_data = json.load(f)

    data = load_data()

    # merge without overwriting duplicates
    for service, accounts in imported_data.items():
        if service not in data:
            data[service] = accounts
        else:
            for acc in accounts:
                if not any(existing["username"] == acc["username"] for existing in data[service]):
                    data[service].append(acc)

    save_data(data)
    print(f"📥 Passwords imported from {filename}")
    
def change_master_password(old_password: str, new_password: str):
    # Load old key
    try:
        with open(KEY_FILE, "rb") as key_file:
            old_key = key_file.read()
        old_fernet = Fernet(old_key)
    except Exception:
        print("❌ Could not load old encryption key!")
        return

    # Generate new key from new password (hashed into 32 bytes for Fernet)
    from base64 import urlsafe_b64encode
    import hashlib
    new_key = urlsafe_b64encode(hashlib.sha256(new_password.encode()).digest())
    new_fernet = Fernet(new_key)

    # Load existing data
    data = load_data()

    # Re-encrypt everything
    for service, accounts in data.items():
        for account in accounts:
            try:
                decrypted_pass = old_fernet.decrypt(account["password"].encode()).decode()
                account["password"] = new_fernet.encrypt(decrypted_pass.encode()).decode()
            except Exception:
                print(f"⚠️ Failed to re-encrypt password for {account['username']} in {service}")

    # Save data with new encryption
    save_data(data)

    # Replace key file
    with open(KEY_FILE, "wb") as key_file:
        key_file.write(new_key)

    print("🔑 Master password changed and all data re-encrypted!")
