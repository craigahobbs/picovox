#!/bin/bash
# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# make train's stage bookkeeping (the training dashboard reads it): each stage's start, end, and failure as a line of
# TRAIN_DIR/stages.jsonl ({"stage", "event": start|end|fail, "time": epoch seconds}), and each command's output (stdout
# and stderr) also in TRAIN_DIR/logs/STAGE.log
#
# tools/trainstage.sh TRAIN_DIR STAGE MAKE_PID start|end
# tools/trainstage.sh TRAIN_DIR STAGE MAKE_PID -- [VAR=value]... COMMAND [ARG]...
#
# MAKE_PID is the running make's process id (the Makefile's STAGE passes it)

set -u
dir=$1
stage=$2
make_pid=$3
shift 3
mkdir -p "$dir/logs"

# Append an event (with optional extra JSON members)
event() {
    printf '{"stage": "%s", "event": "%s", "time": %s%s}\n' "$stage" "$1" "$(date +%s)" "${2:-}" >> "$dir/stages.jsonl"
}

# The start and end events - a start also starts the utilization sampler for the life of the make, if none runs
if [ "${1:-}" = start ] || [ "${1:-}" = end ]; then
    event "$1"
    if [ "$1" = start ]; then
        nohup "$(dirname "$0")/trainutil.sh" "$dir" "$make_pid" >/dev/null 2>&1 &
    fi
    exit 0
fi

# The command
if [ "${1:-}" = -- ]; then
    shift
fi

# Run the command (in the background, so a signal - make interrupted - is handled at once: the command is stopped and
# the stage marked failed), its output tee'd to the stage's log; a failure is marked failed with its exit status. The
# command is stopped with TERM even when make is interrupted: a background command starts with SIGINT ignored.
trap 'kill -TERM "$pid" 2>/dev/null; wait "$pid"; event fail ", \"signal\": \"terminated\""; exit 143' TERM
trap 'kill -TERM "$pid" 2>/dev/null; wait "$pid"; event fail ", \"signal\": \"interrupted\""; exit 130' INT
PYTHONUNBUFFERED=1 env "$@" > >(tee -a "$dir/logs/$stage.log") 2>&1 &
pid=$!
wait "$pid"
status=$?
trap - TERM INT
if [ "$status" -ne 0 ]; then
    event fail ", \"status\": $status"
fi
exit "$status"
