#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "utils.h"

#define HISTORY_FILE "history/history.txt"

// keep in-memory history
static char *history[100];
static int history_count = 0;

void add_history(const char *cmd) {
    if (history_count < 100) {
        history[history_count++] = strdup(cmd);
    }
}

void show_history() {
    for (int i = 0; i < history_count; i++) {
        printf("%d %s\n", i + 1, history[i]);
    }
}

void print_prompt() {
    char cwd[1024];
    getcwd(cwd, sizeof(cwd));
    printf("\033[1;32mMyShell\033[0m:\033[1;34m%s\033[0m$ ", cwd);
    fflush(stdout);
}

void load_history() {
    FILE *file = fopen(HISTORY_FILE, "r");
    if (!file) return;
    char line[100];
    while (fgets(line, sizeof(line), file)) {
        line[strcspn(line, "\n")] = 0;
        add_history(line);
    }
    fclose(file);
}

void save_history(const char *cmd) {
    FILE *file = fopen(HISTORY_FILE, "a");
    if (file) {
        fprintf(file, "%s\n", cmd);
        fclose(file);
    }
    add_history(cmd);
}
