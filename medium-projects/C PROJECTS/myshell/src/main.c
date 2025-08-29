#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "commands.h"
#include "executor.h"
#include "utils.h"

#define MAX_INPUT 100

int main() {
    char input[MAX_INPUT];

    // load previous history
    load_history();

    while (1) {
        // print colored prompt with current dir
        print_prompt();

        // read user input
        if (fgets(input, sizeof(input), stdin) == NULL) break;
        input[strcspn(input, "\n")] = 0; // remove newline

        if (strlen(input) == 0) continue;

        // save to history
        save_history(input);

        // check for built-in commands
        if (handle_builtin(input)) continue;

        // else, execute external command
        execute_command(input);
    }

    return 0;
}
