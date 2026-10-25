#!/usr/bin/env bash
set -eu

umask 077

WALL_LIMIT_SECONDS=5400
GPU_MEMORY_LIMIT_MIB=40000
SELF="$(readlink -f -- "$0")"

usage() {
    echo "usage: $0 launch SOURCE_ROOT RUN_ROOT EXPERIMENT_JSON ISAAC_PY TWIN_STAGE" >&2
    echo "       set FLUIDMIX_F2_QUALIFICATION_MODE=1 for the bounded qualification run" >&2
    echo "       set FLUIDMIX_F2_WATER_PBD_PRESET=1 with qualification mode for A29" >&2
    exit 64
}

atomic_text() {
    target=$1
    value=$2
    temporary="${target}.partial.$$"
    printf '%s\n' "$value" >"$temporary"
    mv -f -- "$temporary" "$target"
}

atomic_copy() {
    source=$1
    target=$2
    temporary="${target}.partial.$$"
    cp -- "$source" "$temporary"
    mv -f -- "$temporary" "$target"
}

process_group() {
    pid=$1
    ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]'
}

valid_positive_integer() {
    case $1 in
        ''|*[!0-9]*) return 1 ;;
        *) test "$1" -gt 1 ;;
    esac
}

claim_stop_reason() {
    run_root=$1
    reason=$2
    if mkdir -- "$run_root/stop.claim" 2>/dev/null
    then
        atomic_text "$run_root/stop-reason.txt" "$reason"
        return 0
    fi
    return 1
}

terminate_recorded_group() {
    pid=$1
    pgid=$2
    valid_positive_integer "$pid" || return 1
    valid_positive_integer "$pgid" || return 1
    actual=$(process_group "$pid" || true)
    test "$actual" = "$pgid" || return 1
    kill -TERM -- "-$pgid" 2>/dev/null || true
    attempts=0
    while kill -0 -- "-$pgid" 2>/dev/null && test "$attempts" -lt 300
    do
        sleep 0.1
        attempts=$((attempts + 1))
    done
    if kill -0 -- "-$pgid" 2>/dev/null
    then
        kill -KILL -- "-$pgid" 2>/dev/null || true
    fi
}

monitor_configuration() {
    MEMORY_MODE=${FLUIDMIX_F2_MEMORY_MODE:-nvidia_smi}
    if test "${FLUIDMIX_F2_TEST_MODE:-0}" = 1
    then
        MONITOR_INTERVAL=${FLUIDMIX_F2_MONITOR_INTERVAL_SECONDS:-0.05}
        case $MONITOR_INTERVAL in
            ''|*[!0-9.]*) return 1 ;;
        esac
    else
        MONITOR_INTERVAL=5
    fi

    case $MEMORY_MODE in
        nvidia_smi)
            if test "${FLUIDMIX_F2_TEST_MODE:-0}" = 1
            then
                NVIDIA_SMI=${FLUIDMIX_F2_NVIDIA_SMI:-}
            else
                NVIDIA_SMI=$(command -v nvidia-smi || true)
            fi
            case $NVIDIA_SMI in
                /*) ;;
                *) return 1 ;;
            esac
            test -x "$NVIDIA_SMI" || return 1
            MEMORY_SOURCE=$NVIDIA_SMI
            ;;
        local_uma)
            # DGX Spark uses unified system memory and does not expose memory usage
            # through nvidia-smi. This local-only mode is deliberately opt-in.
            # https://docs.nvidia.com/dgx/dgx-spark/dgx-spark.pdf
            if test "${FLUIDMIX_F2_TEST_MODE:-0}" = 1
            then
                MEMINFO_PATH=${FLUIDMIX_F2_MEMINFO:-}
            else
                MEMINFO_PATH=/proc/meminfo
            fi
            case $MEMINFO_PATH in
                /*) ;;
                *) return 1 ;;
            esac
            MEMORY_SOURCE=$MEMINFO_PATH
            ;;
        *) return 1 ;;
    esac
    export MEMORY_MODE MEMORY_SOURCE NVIDIA_SMI MEMINFO_PATH MONITOR_INTERVAL
}

sample_nvidia_smi_memory() {
    raw=$(
        "$NVIDIA_SMI" \
            --query-gpu=memory.used \
            --format=csv,noheader,nounits 2>/dev/null
    ) || return 1
    printf '%s\n' "$raw" | awk '
        BEGIN { count = 0; maximum = -1 }
        NF {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
            if ($0 !~ /^[0-9]+$/) exit 2
            value = $0 + 0
            if (value > maximum) maximum = value
            count += 1
        }
        END {
            if (count == 0) exit 3
            print maximum
        }
    '
}

sample_local_uma_memory() {
    awk '
        $1 == "MemTotal:" {
            if (seen_total || NF != 3 || $2 !~ /^[0-9]+$/ || $3 != "kB") invalid = 1
            total = $2
            seen_total = 1
        }
        $1 == "MemAvailable:" {
            if (seen_available || NF != 3 || $2 !~ /^[0-9]+$/ || $3 != "kB") invalid = 1
            available = $2
            seen_available = 1
        }
        END {
            if (invalid || !seen_total || !seen_available || total <= 0 || available > total) exit 2
            used_kib = total - available
            # Round up so conversion to integer MiB cannot hide a limit crossing.
            print int((used_kib + 1023) / 1024)
        }
    ' "$MEMINFO_PATH" 2>/dev/null
}

sample_memory() {
    case $MEMORY_MODE in
        nvidia_smi) sample_nvidia_smi_memory ;;
        local_uma) sample_local_uma_memory ;;
        *) return 1 ;;
    esac
}

monitor_gpu() {
    run_root=$1
    isaac_pid=$2
    isaac_pgid=$3
    partial="$run_root/gpu-memory.csv.partial.$$"
    printf '%s\n' 'timestamp_utc,used_mib' >"$partial"
    sample_count=0
    finalize_monitor() {
        if test "$sample_count" -eq 0
        then
            claim_stop_reason "$run_root" gpu_memory_unreadable || true
            terminate_recorded_group "$isaac_pid" "$isaac_pgid" || true
        fi
        mv -f -- "$partial" "$run_root/gpu-memory.csv"
    }
    trap finalize_monitor EXIT

    monitor_configuration || {
        claim_stop_reason "$run_root" gpu_memory_unreadable || true
        terminate_recorded_group "$isaac_pid" "$isaac_pgid" || true
        return 0
    }
    atomic_text "$run_root/memory-monitor-mode.txt" "$MEMORY_MODE"
    atomic_text "$run_root/memory-monitor-source.txt" "$MEMORY_SOURCE"

    while kill -0 "$isaac_pid" 2>/dev/null
    do
        set +e
        used=$(sample_memory)
        sample_status=$?
        set -e
        if test "$sample_status" -ne 0
        then
            claim_stop_reason "$run_root" gpu_memory_unreadable || true
            terminate_recorded_group "$isaac_pid" "$isaac_pgid" || true
            break
        fi
        printf '%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$used" >>"$partial"
        sample_count=$((sample_count + 1))
        if test "$used" -ge "$GPU_MEMORY_LIMIT_MIB"
        then
            claim_stop_reason "$run_root" gpu_memory_limit_exceeded || true
            terminate_recorded_group "$isaac_pid" "$isaac_pgid" || true
            break
        fi
        if test -f "$run_root/monitor.stop"
        then
            break
        fi
        sleep "$MONITOR_INTERVAL"
    done
}

supervise() {
    source_root=$1
    run_root=$2
    experiment=$3
    isaac_python=$4
    twin_stage=$5
    qualification_mode=$6
    water_pbd_preset=$7

    supervisor_pid=$$
    supervisor_pgid=$(process_group "$supervisor_pid")
    valid_positive_integer "$supervisor_pgid" || exit 70
    atomic_text "$run_root/supervisor.pid" "$supervisor_pid"
    atomic_text "$run_root/supervisor.pgid" "$supervisor_pgid"
    atomic_text "$run_root/started-at.txt" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    runner="$source_root/scripts/physx_three_pour_sequence.py"
    output="$run_root/app-output"
    qualification_args=()
    if test "$qualification_mode" = 1
    then
        qualification_args+=(--qualification-mode)
    fi
    if test "$water_pbd_preset" = 1
    then
        qualification_args+=(--qualification-water-pbd-preset)
    fi
    setsid timeout --signal=TERM --kill-after=30s "${WALL_LIMIT_SECONDS}s" \
        env OMNI_KIT_ACCEPT_EULA=YES \
        "$isaac_python" "$runner" \
        --out "$output" --stage "$twin_stage" --seed 20261025 \
        "${qualification_args[@]}" \
        >"$run_root/isaac.log" 2>&1 </dev/null &
    isaac_pid=$!
    isaac_pgid=$isaac_pid
    atomic_text "$run_root/isaac.pid" "$isaac_pid"
    atomic_text "$run_root/isaac.pgid" "$isaac_pgid"

    setsid "$SELF" _monitor "$run_root" "$isaac_pid" "$isaac_pgid" \
        >"$run_root/monitor.log" 2>&1 </dev/null &
    monitor_pid=$!
    monitor_pgid=$monitor_pid
    atomic_text "$run_root/monitor.pid" "$monitor_pid"
    atomic_text "$run_root/monitor.pgid" "$monitor_pgid"

    set +e
    wait "$isaac_pid"
    exit_code=$?
    set -e
    atomic_text "$run_root/monitor.stop" stop
    set +e
    wait "$monitor_pid"
    set -e

    if test ! -f "$run_root/stop-reason.txt"
    then
        if test "$exit_code" -eq 0
        then
            claim_stop_reason "$run_root" completed || true
        elif test "$exit_code" -eq 124 || test "$exit_code" -eq 137
        then
            claim_stop_reason "$run_root" wall_time_exceeded || true
        else
            claim_stop_reason "$run_root" "application_exit_${exit_code}" || true
        fi
    fi
    stop_reason=$(tr -d '\r\n' <"$run_root/stop-reason.txt")
    atomic_text "$run_root/exit-code.txt" "$exit_code"
    atomic_text "$run_root/finished-at.txt" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    experiment_sha=$(sha256sum -- "$experiment" | awk '{print $1}')
    if test "$exit_code" -eq 0 && test "$stop_reason" = completed
    then
        status=completed
    else
        status=failed
    fi
    completion="$run_root/completion.json.partial.$$"
    printf '{\n' >"$completion"
    printf '  "status": "%s",\n' "$status" >>"$completion"
    printf '  "stop_reason": "%s",\n' "$stop_reason" >>"$completion"
    printf '  "exit_code": %s,\n' "$exit_code" >>"$completion"
    printf '  "wall_limit_seconds": %s,\n' "$WALL_LIMIT_SECONDS" >>"$completion"
    printf '  "gpu_memory_limit_mib": %s,\n' "$GPU_MEMORY_LIMIT_MIB" >>"$completion"
    if test "$qualification_mode" = 1
    then
        printf '  "qualification_mode": true,\n' >>"$completion"
    else
        printf '  "qualification_mode": false,\n' >>"$completion"
    fi
    if test "$water_pbd_preset" = 1
    then
        printf '  "water_pbd_preset": true,\n' >>"$completion"
    fi
    printf '  "experiment_sha256": "%s",\n' "$experiment_sha" >>"$completion"
    printf '  "supervisor_pid": %s,\n' "$supervisor_pid" >>"$completion"
    printf '  "supervisor_pgid": %s,\n' "$supervisor_pgid" >>"$completion"
    printf '  "isaac_pid": %s,\n' "$isaac_pid" >>"$completion"
    printf '  "isaac_pgid": %s,\n' "$isaac_pgid" >>"$completion"
    printf '  "monitor_pid": %s,\n' "$monitor_pid" >>"$completion"
    printf '  "monitor_pgid": %s\n' "$monitor_pgid" >>"$completion"
    printf '}\n' >>"$completion"
    mv -f -- "$completion" "$run_root/completion.json"
}

launch() {
    source_root=$1
    run_root=$2
    experiment=$3
    isaac_python=$4
    twin_stage=$5
    qualification_mode=${FLUIDMIX_F2_QUALIFICATION_MODE:-0}
    water_pbd_preset=${FLUIDMIX_F2_WATER_PBD_PRESET:-0}
    case $qualification_mode in
        0|1) ;;
        *) echo "F2 qualification mode must be 0 or 1" >&2; exit 64 ;;
    esac
    case $water_pbd_preset in
        0|1) ;;
        *) echo "F2 water PBD preset must be 0 or 1" >&2; exit 64 ;;
    esac
    if test "$water_pbd_preset" = 1 && test "$qualification_mode" != 1
    then
        echo "F2 water PBD preset requires qualification mode" >&2
        exit 64
    fi
    case $source_root:$run_root:$experiment:$isaac_python:$twin_stage in
        /*:/*:/*:/*:/*) ;;
        *) echo "F2 launcher paths must be absolute" >&2; exit 64 ;;
    esac
    test -d "$source_root" || { echo "source root is missing" >&2; exit 66; }
    test -f "$source_root/scripts/physx_three_pour_sequence.py" || {
        echo "F2 runner is missing" >&2
        exit 66
    }
    test -f "$experiment" || { echo "experiment JSON is missing" >&2; exit 66; }
    test -x "$isaac_python" || { echo "Isaac Python is not executable" >&2; exit 66; }
    test -f "$twin_stage" || { echo "twin stage is missing" >&2; exit 66; }
    if test -e "$run_root" && { test ! -d "$run_root" || test -L "$run_root"; }
    then
        echo "run root is not a plain directory" >&2
        exit 73
    fi
    mkdir -p -m 700 -- "$run_root"
    if ! mkdir -m 700 -- "$run_root/launch.guard" 2>/dev/null
    then
        echo "F2 already launched; persistent guard retained"
        return 0
    fi
    atomic_copy "$experiment" "$run_root/experiment.json"
    atomic_text "$run_root/qualification-mode.txt" "$qualification_mode"
    if test "$water_pbd_preset" = 1
    then
        atomic_text "$run_root/water-pbd-preset.txt" "$water_pbd_preset"
    fi
    sha256sum -- "$run_root/experiment.json" >"$run_root/experiment.sha256.partial.$$"
    mv -f -- "$run_root/experiment.sha256.partial.$$" "$run_root/experiment.sha256"

    FLUIDMIX_F2_INTERNAL=1 nohup setsid "$SELF" _supervise \
        "$source_root" "$run_root" "$run_root/experiment.json" "$isaac_python" "$twin_stage" \
        "$qualification_mode" "$water_pbd_preset" \
        >"$run_root/supervisor.log" 2>&1 </dev/null &
    bootstrap_pid=$!
    attempts=0
    while test ! -f "$run_root/supervisor.pid" && kill -0 "$bootstrap_pid" 2>/dev/null && test "$attempts" -lt 100
    do
        sleep 0.05
        attempts=$((attempts + 1))
    done
    if test ! -f "$run_root/supervisor.pid"
    then
        echo "F2 supervisor failed before recording its PID" >&2
        return 70
    fi
    echo "F2 launch accepted; supervisor PID $(cat "$run_root/supervisor.pid")"
}

case ${1:-} in
    launch)
        test "$#" -eq 6 || usage
        launch "$2" "$3" "$4" "$5" "$6"
        ;;
    _supervise)
        test "${FLUIDMIX_F2_INTERNAL:-0}" = 1 || exit 64
        test "$#" -eq 8 || exit 64
        supervise "$2" "$3" "$4" "$5" "$6" "$7" "$8"
        ;;
    _monitor)
        test "${FLUIDMIX_F2_INTERNAL:-0}" = 1 || exit 64
        test "$#" -eq 4 || exit 64
        monitor_gpu "$2" "$3" "$4"
        ;;
    *) usage ;;
esac
