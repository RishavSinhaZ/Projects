#include <stdio.h>
#include <string.h>
#include <stdlib.h>   // for exit(), system()
#include <unistd.h>
#include "commands.h"
#include "utils.h"    // ✅ needed for show_history()

int handle_builtin(char *input) {
    if (strcmp(input, "exit") == 0) {
        printf("Exiting MyShell...\n");
        exit(0);
    }

    if (strcmp(input, "clear") == 0) {
        system("clear");
        return 1;
    }

    if (strcmp(input, "help") == 0) {
        printf("MyShell - Available commands:\n");
        printf("  cd <dir>    - Change directory\n");
        printf("  history     - Show command history\n");
        printf("  clear       - Clear terminal\n");
        printf("  exit        - Exit shell\n");
        return 1;
    }

    if (strncmp(input, "cd ", 3) == 0) {
        char *path = input + 3;
        if (chdir(path) != 0) perror("cd failed");
        return 1;
    }

    if (strcmp(input, "history") == 0) {
        show_history();   // ✅ now works because of utils.h
        return 1;
    }

    return 0; // not a builtin
}
