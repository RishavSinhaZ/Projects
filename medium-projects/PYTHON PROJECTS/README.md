# 🔐 Password Manager

A secure and customizable **Password Manager** built in Python.  
It allows you to **store, view, edit, delete, import, and export passwords** with encryption.  
The project also includes activity logging, backup support, and master password protection.

---

## 📂 Project Structure

```
📁 password-manager
│── main.py              # Entry point of the application (menu-driven interface)
│── manager.py           # Core logic for password management (add, edit, delete, list, etc.)
│── encryption.py        # Handles encryption and decryption of data
│── utils.py             # Helper functions (file handling, validations, etc.)
│── data.json            # Encrypted storage for passwords
│── backup.json          # Backup file of saved passwords
│── key.key              # Encryption key (DO NOT share or delete)
│── secret.key           # Master password key file
│── activity.log         # Logs all password manager activities
│── README.md            # Project documentation
```

## ⚡ Features

- **Master Password Protection** 🔑
- **Add / View / Edit / Delete Passwords**
- **List All Stored Services**
- **Export & Import Passwords**
- **Encrypted Storage with Fernet**
- **Activity Logging (`activity.log`)**
- **Automatic Backup (`backup.json`)**
- **Retry Lockout (after 3 wrong inputs, wait timer applies)**

---

## 📌 Menu Options

1. Add Password

2. View Password

3. Delete Password

4. List Services

5. Export Passwords

6. Import Passwords

7. Edit Existing Password

8. Change Master Password

9. Exit

----

## Install dependencies:

    pip install cryptography


## Run the program:

    python main.py

---

🔒 Security Notes

    Do not share your key.key or secret.key files.

    The data.json file contains encrypted passwords – it is safe but still should be backed up.

Always use a strong master password.


🧑‍💻 Contribution

    Feel free to fork, improve, and submit pull requests.
    Suggestions for more features (like cloud sync, 2FA, GUI) are welcome 🚀.


---
📜 License

**This project is open-source under the MIT License.**
