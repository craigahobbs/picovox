#!/bin/bash
# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

# make train's utilization sampler (the training dashboard charts it): every PERIOD seconds (default 20) while the make
# process MAKE_PID runs, a line of TRAIN_DIR/util.log - the time, the CPU busy % (top), the GPU utilization % (ioreg's
# IOAccelerator "Device Utilization %" - no sudo), the memory used %, and the running stages (TRAIN_DIR/stages.jsonl).
# It exits when make does (finished, failed, or interrupted); one sampler per TRAIN_DIR.
#
# tools/trainutil.sh TRAIN_DIR MAKE_PID [PERIOD]

set -u
dir=$1
make_pid=$2
period=${3:-20}
log=$dir/util.log
pid_file=$dir/util.pid
mkdir -p "$dir"

# One sampler per TRAIN_DIR
if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    exit 0
fi
echo $$ > "$pid_file"
trap 'rm -f "$pid_file"' EXIT

# Each start writes the header (it marks a new make train in a resumed log)
echo 'time cpu_busy% gpu% mem_used% stages' >> "$log"

# The running stages - those whose last event is a start - comma-separated
running_stages() {
    if [ -f "$dir/stages.jsonl" ]; then
        sed -n 's/^{"stage": "\([^"]*\)", "event": "\([a-z]*\)".*/\1 \2/p' "$dir/stages.jsonl" |
            awk '{last[$1] = $2; if (!($1 in order)) {order[$1] = ++count; names[count] = $1}}
                END {for (i = 1; i <= count; i++) if (last[names[i]] == "start") {printf "%s%s", sep, names[i]; sep = ","}}'
    fi
}

# Sample until make exits
while kill -0 "$make_pid" 2>/dev/null; do
    stats=$(top -l 2 -n 0 -s 1 | awk '
        function bytes(text) {
            unit = substr(text, length(text)); value = substr(text, 1, length(text) - 1)
            return value * (unit == "G" ? 1024 : unit == "K" ? 1 / 1024 : 1)
        }
        /CPU usage/ {for (i = 1; i <= NF; i++) if ($i == "idle") idle = $(i - 1); sub(/%/, "", idle); busy = 100 - idle}
        /PhysMem/ {used = bytes($2); for (i = 1; i <= NF; i++) if ($i ~ /^unused/) free = bytes($(i - 1))}
        END {printf "%.1f %.1f", busy, (used + free > 0 ? 100 * used / (used + free) : 0)}')
    gpu=$(ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2)
    stages=$(running_stages)
    set -- $stats
    echo "$(date '+%Y-%m-%dT%H:%M:%S') $1 ${gpu:-0} $2 ${stages:--}" >> "$log"
    for _ in $(seq "$period"); do
        kill -0 "$make_pid" 2>/dev/null || break
        sleep 1
    done
done
