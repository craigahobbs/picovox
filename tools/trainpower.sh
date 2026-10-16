#!/bin/bash
# Licensed under the MIT License
# https://github.com/craigahobbs/voice-synth/blob/main/LICENSE

# make train's power check (macOS laptops): the training run keeps the GPU busy for hours, drawing more than a small
# power adapter supplies (an M3 Max measured ~85 W), so a run on battery or a low-wattage adapter drains the battery and
# the machine slows down. If the adapter is under MIN_WATTS (default 96, TRAIN_MIN_WATTS), say so - and, at a terminal,
# wait for Enter (plug in a bigger adapter first, then Enter re-checks). TRAIN_POWER_CHECK=0 skips the check.
#
# tools/trainpower.sh [MIN_WATTS]

set -u
min_watts=${1:-${TRAIN_MIN_WATTS:-96}}
[ "${TRAIN_POWER_CHECK:-1}" = 0 ] && exit 0

# The adapter's watts (0 on battery); no battery (a desktop Mac, or not macOS) - nothing to check
adapter_watts() {
    local battery
    battery=$(ioreg -rn AppleSmartBattery 2>/dev/null) || return 1
    [ -n "$battery" ] || return 1
    if echo "$battery" | grep -q '"ExternalConnected" = Yes'; then
        echo "$battery" | grep -oE '"Watts"=[0-9]+' | head -1 | cut -d= -f2 | grep . || echo 0
    else
        echo 0
    fi
}
watts=$(adapter_watts) || exit 0
[ "$watts" -ge "$min_watts" ] && exit 0

describe() {
    if [ "$1" = 0 ]; then echo 'on battery'; else echo "on a $1 W power adapter"; fi
}
echo "make train: this Mac is $(describe "$watts") - make train keeps the GPU busy for hours and needs a ${min_watts} W" \
    "or larger adapter; on less the battery drains and training slows down."
if [ -t 0 ] && [ -t 1 ]; then
    read -r -p "make train: plug in a ${min_watts} W+ adapter, then press Enter (or press Enter to continue as is) "
    watts=$(adapter_watts)
    if [ "$watts" -ge "$min_watts" ]; then
        echo "make train: now on a $watts W adapter"
    else
        echo "make train: continuing $(describe "$watts")"
    fi
fi
exit 0
