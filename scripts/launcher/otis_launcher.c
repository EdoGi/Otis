/*
 * Otis.app's executable: start scripts/run.sh detached, then exit.
 *
 * It replaces an osacompile AppleScript applet whose only job was the same
 * `do shell script "exec run.sh >/dev/null 2>&1 &"`. The AppleScript runtime
 * cost 4-14 s before run.sh even started (a bare applet: 0.7-2.7 s on this
 * Mac), with nothing on screen meanwhile. This takes milliseconds.
 *
 * Its stderr — inherited by bash — goes to ~/.otis/launch.log instead of
 * /dev/null, so a launch that dies before run.sh is read (moved repo, a TCC
 * denial on ~/Documents) still leaves a line. OTIS_LAUNCHED_BY_APP tells
 * run.sh to keep its own launch log.
 *
 * Build (scripts/build_app.sh does this):
 *   clang -O2 -DOTIS_RUN_SH='"/abs/path/to/scripts/run.sh"' -o Otis otis_launcher.c
 */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef OTIS_RUN_SH
#error "build with -DOTIS_RUN_SH=\"/absolute/path/to/scripts/run.sh\""
#endif

extern char **environ;

static int open_launch_log(void) {
    const char *home = getenv("HOME");
    char path[PATH_MAX];
    if (home == NULL || home[0] == '\0') {
        return -1;
    }
    snprintf(path, sizeof path, "%s/.otis", home);
    if (mkdir(path, 0700) != 0 && errno != EEXIST) {
        return -1;
    }
    snprintf(path, sizeof path, "%s/.otis/launch.log", home);
    return open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
}

int main(void) {
    int null_fd = open("/dev/null", O_RDWR | O_CLOEXEC);
    int log_fd = open_launch_log();
    int err_fd = log_fd >= 0 ? log_fd : null_fd;

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    if (null_fd >= 0) {
        posix_spawn_file_actions_adddup2(&actions, null_fd, STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions, null_fd, STDOUT_FILENO);
    }
    if (err_fd >= 0) {
        posix_spawn_file_actions_adddup2(&actions, err_fd, STDERR_FILENO);
    }

    posix_spawnattr_t attrs;
    posix_spawnattr_init(&attrs);
    posix_spawnattr_setflags(&attrs, POSIX_SPAWN_SETSID);

    setenv("OTIS_LAUNCHED_BY_APP", "1", 1);
    char *argv[] = {"/bin/bash", OTIS_RUN_SH, NULL};
    pid_t pid;
    int rc = posix_spawn(&pid, "/bin/bash", &actions, &attrs, argv, environ);
    if (rc != 0 && log_fd >= 0) {
        dprintf(log_fd, "Otis.app launcher: could not start %s: %s\n",
                OTIS_RUN_SH, strerror(rc));
    }
    return rc == 0 ? 0 : 1;
}
