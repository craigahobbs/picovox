#!/bin/bash
# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

# make train's stage bookkeeping (the training dashboard reads it): each stage's start, end, and failure as a line of
# TRAIN_DIR/stages.jsonl ({"stage", "event": start|end|fail, "time": epoch seconds}), and each command's output (stdout
# and stderr) also in TRAIN_DIR/logs/STAGE.log and any --log file (a training run's train.log)
#
# tools/trainstage.sh TRAIN_DIR STAGE start|end
# tools/trainstage.sh TRAIN_DIR STAGE [--log FILE]... -- [VAR=value]... COMMAND [ARG]...

set -u
dir=$1
stage=$2
shift 2
mkdir -p "$dir/logs"

# Append an event (with optional extra JSON members)
event() {
    printf '{"stage": "%s", "event": "%s", "time": %s%s}\n' "$stage" "$1" "$(date +%s)" "${2:-}" >> "$dir/stages.jsonl"
}

# The start and end events
if [ "${1:-}" = start ] || [ "${1:-}" = end ]; then
    event "$1"
    exit 0
fi

# The command's logs
logs=("$dir/logs/$stage.log")
while [ "${1:-}" = --log ]; do
    mkdir -p "$(dirname "$2")"
    logs+=("$2")
    shift 2
done
if [ "${1:-}" = -- ]; then
    shift
fi

# Run the command (in the background, so a signal - make interrupted - is handled at once: the command is stopped and
# the stage marked failed), its output tee'd to the logs; a failure is marked failed with its exit status
trap 'kill -TERM "$pid" 2>/dev/null; wait "$pid"; event fail ", \"signal\": \"terminated\""; exit 143' TERM
trap 'kill -INT "$pid" 2>/dev/null; wait "$pid"; event fail ", \"signal\": \"interrupted\""; exit 130' INT
PYTHONUNBUFFERED=1 env "$@" > >(tee -a "${logs[@]}") 2>&1 &
pid=$!
wait "$pid"
status=$?
trap - TERM INT
if [ "$status" -ne 0 ]; then
    event fail ", \"status\": $status"
fi
exit "$status"
