# MyShell - Mini Linux Shell in C

## Features
- Custom prompt with colors + current working directory
- Built-in commands: `cd`, `exit`, `clear`, `help`, `history`
- Execute external commands (`ls`, `pwd`, etc.)
- Supports background processes (`command &`)
- Persistent command history saved in `history/history.txt`

## Project Structure
---
myshell/
├── src/ # C source files
├── include/ # Header files
├── history/ # History persistence
├── Makefile
└── README.md
---


## Build & Run
```bash
make
./myshell

MyShell:/home/mrsky$ ls -l
MyShell:/home/mrsky$ cd projects
MyShell:/home/mrsky/projects$ history


---

⚡ With this structure + features, your project is no longer a “toy” shell → it’s a **mini terminal emulator** with persistence, history, and process management. Interviewers will *definitely* ask about `fork`, `execvp`, and history handling.  

---

👉 Do you want me to actually **write the Makefile and headers** too so you can copy-paste and run it directly, or do you want to start with this skeleton first and then I’ll expand?


