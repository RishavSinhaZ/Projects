from flask import Flask, request, jsonify
from encryption import encrypt_message, decrypt_message

app = Flask(__name__)

# Temporary in-memory store (for testing)
messages = {}

@app.route("/send", methods=["POST"])
def send_message():
    data = request.json
    sender = data.get("sender")
    receiver = data.get("receiver")
    message = data.get("message")

    if not (sender and receiver and message):
        return jsonify({"error": "Missing fields"}), 400

    encrypted = encrypt_message(message)
    if receiver not in messages:
        messages[receiver] = []
    messages[receiver].append({"from": sender, "text": encrypted})
    return jsonify({"status": "Message sent", "encrypted": encrypted})

@app.route("/receive/<username>", methods=["GET"])
def receive_message(username):
    user_messages = messages.get(username, [])
    messages[username] = []  # clear inbox after fetching
    decrypted_msgs = [{"from": m["from"], "text": decrypt_message(m["text"])} for m in user_messages]
    return jsonify({"messages": decrypted_msgs})

if __name__ == "__main__":
    app.run(debug=True)
