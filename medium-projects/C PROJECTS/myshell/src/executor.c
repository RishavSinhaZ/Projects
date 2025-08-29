#include <stdio.h>
#include <stdlib.h>   // ✅ added for exit()
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include "executor.h"

void execute_command(char *input) {
    char *args[20];
    int argc = 0;

    // tokenize input
    char *token = strtok(input, " ");
    while (token != NULL) {
        args[argc++] = token;
        token = strtok(NULL, " ");
    }
    args[argc] = NULL;

    if (argc == 0) return;

    // background execution
    int background = 0;
    if (strcmp(args[argc - 1], "&") == 0) {
        background = 1;
        args[argc - 1] = NULL;
    }

    pid_t pid = fork();

    if (pid == 0) {
        // child
        execvp(args[0], args);
        perror("exec failed");
        exit(1);
    } else if (pid > 0) {
        if (!background) wait(NULL); // wait only if not background
    } else {
        perror("fork failed");
    }
}
