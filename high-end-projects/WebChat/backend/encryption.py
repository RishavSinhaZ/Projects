from Crypto.Cipher import AES
import base64
import os

# 16-byte key for AES (in real app: generate dynamically per session)
SECRET_KEY = b"ThisIsASecretKey"

def pad(text):
    return text + (16 - len(text) % 16) * chr(16 - len(text) % 16)

def unpad(text):
    return text[:-ord(text[-1])]

def encrypt_message(message):
    cipher = AES.new(SECRET_KEY, AES.MODE_ECB)
    encrypted = cipher.encrypt(pad(message).encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")

def decrypt_message(encrypted_text):
    cipher = AES.new(SECRET_KEY, AES.MODE_ECB)
    decrypted = cipher.decrypt(base64.b64decode(encrypted_text))
    return unpad(decrypted.decode("utf-8"))
