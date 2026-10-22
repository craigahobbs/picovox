#!/bin/bash
# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

# make train's utilization sampler (the training dashboard charts it): every PERIOD seconds (default 20) while the make
# process MAKE_PID runs, a line of TRAIN_DIR/util.log - the time, the CPU busy % (top), the GPU utilization % (ioreg's
# IOAccelerator "Device Utilization %" - no sudo), the memory used %, the system's power and the power it takes in, in
# watts (a laptop's - ioreg's AppleSmartBattery telemetry, its SystemLoad and SystemPowerIn: power in above the system's
# charges the battery, below it drains it; - without a battery), macOS's thermal pressure level (0 nominal, 1 moderate,
# 2 heavy, 3 trapping, 4 sleeping - notifyutil's com.apple.system.thermalpressurelevel: a 14-inch M3 Max reaches heavy
# after about 2.5 minutes of a GPU stage at 70 W, then holds the system near 40 W - the GPU stages about 50% slower),
# and the running stages (TRAIN_DIR/stages.jsonl). It exits when make does (finished, failed, or interrupted); one
# sampler per TRAIN_DIR.
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
echo 'time cpu_busy% gpu% mem_used% system_w power_in_w thermal stages' >> "$log"

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
    power=$(ioreg -rn AppleSmartBattery 2>/dev/null | awk '
        match($0, /"SystemLoad"=[0-9]+/) {load = substr($0, RSTART + 13, RLENGTH - 13)}
        match($0, /"SystemPowerIn"=[0-9]+/) {power_in = substr($0, RSTART + 16, RLENGTH - 16)}
        END {if (load != "" && power_in != "") printf "%.1f %.1f", load / 1000, power_in / 1000; else printf "- -"}')
    thermal=$(notifyutil -g com.apple.system.thermalpressurelevel 2>/dev/null | awk '{print $2}')
    stages=$(running_stages)
    set -- $stats
    echo "$(date '+%Y-%m-%dT%H:%M:%S') $1 ${gpu:-0} $2 ${power:-- -} ${thermal:--} ${stages:--}" >> "$log"
    for _ in $(seq "$period"); do
        kill -0 "$make_pid" 2>/dev/null || break
        sleep 1
    done
done
