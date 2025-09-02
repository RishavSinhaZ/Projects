let username = "";
let friendname = "";

function createUser() {
  username = document.getElementById("username").value;
  if (username) {
    localStorage.setItem("username", username);
    alert("Username created: " + username);
  }
}

function joinChat() {
  friendname = document.getElementById("friendname").value;
  username = localStorage.getItem("username");
  if (friendname && username) {
    window.location.href = "chat.html";
  } else {
    alert("Please create username first.");
  }
}

async function sendMessage() {
  const message = document.getElementById("message").value;
  if (!message) return;

  await fetch("http://127.0.0.1:5000/send", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sender: username, receiver: friendname, message })
  });
  document.getElementById("message").value = "";
}

async function pollMessages() {
  const res = await fetch(`http://127.0.0.1:5000/receive/${username}`);
  const data = await res.json();
  const box = document.getElementById("chat-box");
  data.messages.forEach(m => {
    let p = document.createElement("p");
    p.textContent = m.from + ": " + m.text;
    box.appendChild(p);
  });
}

setInterval(() => {
  if (localStorage.getItem("username")) {
    pollMessages();
  }
}, 2000);
