#!/usr/bin/env bash
set -eu

umask 077
WALL_LIMIT_SECONDS=1800
GPU_MEMORY_LIMIT_MIB=40000
SELF=$(readlink -f -- "$0")

usage() {
    echo "usage: $0 launch SOURCE_ROOT RUN_ROOT EXPERIMENT_JSON ISAAC_PY TWIN_STAGE" >&2
    exit 64
}

atomic_text() {
    target=$1
    value=$2
    temporary="${target}.partial.$$"
    printf '%s\n' "$value" >"$temporary"
    mv -f -- "$temporary" "$target"
}

process_group() {
    ps -o pgid= -p "$1" 2>/dev/null | tr -d '[:space:]'
}

terminate_group() {
    pid=$1
    pgid=$2
    actual=$(process_group "$pid" || true)
    test -n "$pgid" && test "$actual" = "$pgid" || return 0
    kill -TERM -- "-$pgid" 2>/dev/null || true
    attempts=0
    while kill -0 -- "-$pgid" 2>/dev/null && test "$attempts" -lt 300
    do
        sleep 0.1
        attempts=$((attempts + 1))
    done
    kill -KILL -- "-$pgid" 2>/dev/null || true
}

nvidia_smi_path() {
    if test "${PALETTE_SMOKE_TEST_MODE:-0}" = 1
    then
        value=${PALETTE_SMOKE_NVIDIA_SMI:-}
    else
        value=$(command -v nvidia-smi || true)
    fi
    case $value in
        /*) test -x "$value" && printf '%s\n' "$value" ;;
    esac
}

monitor() {
    run_root=$1
    isaac_pid=$2
    isaac_pgid=$3
    smi=$(nvidia_smi_path || true)
    interval=${PALETTE_SMOKE_MONITOR_INTERVAL:-5}
    partial="$run_root/gpu-memory.csv.partial.$$"
    printf '%s\n' 'timestamp_utc,used_mib' >"$partial"
    samples=0
    reason=''
    while kill -0 "$isaac_pid" 2>/dev/null
    do
        if test -z "$smi"
        then
            reason=gpu_memory_unreadable
            break
        fi
        raw=$($smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true)
        used=$(printf '%s\n' "$raw" | awk '
            BEGIN { count=0; maximum=-1 }
            /^[[:space:]]*[0-9]+[[:space:]]*$/ {
                gsub(/[[:space:]]/, "", $0); if ($0+0 > maximum) maximum=$0+0; count+=1
            }
            END { if (count == 0) exit 1; print maximum }
        ' || true)
        if test -z "$used"
        then
            reason=gpu_memory_unreadable
            break
        fi
        printf '%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$used" >>"$partial"
        samples=$((samples + 1))
        if test "$used" -ge "$GPU_MEMORY_LIMIT_MIB"
        then
            reason=gpu_memory_limit_exceeded
            break
        fi
        test ! -f "$run_root/monitor.stop" || break
        sleep "$interval"
    done
    mv -f -- "$partial" "$run_root/gpu-memory.csv"
    if test -n "$reason"
    then
        atomic_text "$run_root/monitor-failure.txt" "$reason"
        terminate_group "$isaac_pid" "$isaac_pgid"
    elif test "$samples" -eq 0
    then
        atomic_text "$run_root/monitor-failure.txt" gpu_memory_unreadable
        terminate_group "$isaac_pid" "$isaac_pgid"
    fi
}

supervise() {
    source_root=$1
    run_root=$2
    experiment=$3
    isaac_python=$4
    twin_stage=$5
    atomic_text "$run_root/supervisor.pid" "$$"
    atomic_text "$run_root/supervisor.pgid" "$(process_group $$)"
    atomic_text "$run_root/started-at.txt" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    runner="$source_root/scripts/physx_particle_palette_smoke.py"
    setsid timeout --signal=TERM --kill-after=30s "${WALL_LIMIT_SECONDS}s" \
        env OMNI_KIT_ACCEPT_EULA=YES \
        "$isaac_python" "$runner" --stage "$twin_stage" --out "$run_root/app-output" \
        --seed 20261025 \
        >"$run_root/isaac.log" 2>&1 </dev/null &
    isaac_pid=$!
    isaac_pgid=$isaac_pid
    atomic_text "$run_root/isaac.pid" "$isaac_pid"
    atomic_text "$run_root/isaac.pgid" "$isaac_pgid"

    PALETTE_SMOKE_INTERNAL=1 setsid "$SELF" _monitor "$run_root" "$isaac_pid" "$isaac_pgid" \
        >"$run_root/monitor.log" 2>&1 </dev/null &
    monitor_pid=$!
    atomic_text "$run_root/monitor.pid" "$monitor_pid"
    atomic_text "$run_root/monitor.pgid" "$monitor_pid"
    set +e
    wait "$isaac_pid"
    exit_code=$?
    set -e
    atomic_text "$run_root/monitor.stop" stop
    set +e
    wait "$monitor_pid"
    set -e
    if test -f "$run_root/monitor-failure.txt"
    then
        stop_reason=$(tr -d '\r\n' <"$run_root/monitor-failure.txt")
    elif test "$exit_code" -eq 0
    then
        stop_reason=completed
    elif test "$exit_code" -eq 124 || test "$exit_code" -eq 137
    then
        stop_reason=wall_time_exceeded
    else
        stop_reason="application_exit_${exit_code}"
    fi
    atomic_text "$run_root/exit-code.txt" "$exit_code"
    atomic_text "$run_root/stop-reason.txt" "$stop_reason"
    atomic_text "$run_root/finished-at.txt" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    status=failed
    test "$exit_code" -ne 0 || test "$stop_reason" != completed || status=completed
    experiment_sha=$(sha256sum -- "$experiment" | awk '{print $1}')
    temporary="$run_root/completion.json.partial.$$"
    printf '{\n  "status": "%s",\n  "stop_reason": "%s",\n  "exit_code": %s,\n  "wall_limit_seconds": %s,\n  "gpu_memory_limit_mib": %s,\n  "experiment_sha256": "%s"\n}\n' \
        "$status" "$stop_reason" "$exit_code" "$WALL_LIMIT_SECONDS" \
        "$GPU_MEMORY_LIMIT_MIB" "$experiment_sha" >"$temporary"
    mv -f -- "$temporary" "$run_root/completion.json"
}

launch() {
    source_root=$1
    run_root=$2
    experiment=$3
    isaac_python=$4
    twin_stage=$5
    case $source_root:$run_root:$experiment:$isaac_python:$twin_stage in
        /*:/*:/*:/*:/*) ;;
        *) echo "palette smoke paths must be absolute" >&2; exit 64 ;;
    esac
    test -f "$source_root/scripts/physx_particle_palette_smoke.py" || exit 66
    test -f "$experiment" || exit 66
    test -x "$isaac_python" || exit 66
    test -f "$twin_stage" || exit 66
    if test -e "$run_root" && { test ! -d "$run_root" || test -L "$run_root"; }
    then
        echo "run root is not a plain directory" >&2
        exit 73
    fi
    mkdir -p -m 700 -- "$run_root"
    if ! mkdir -m 700 -- "$run_root/launch.guard" 2>/dev/null
    then
        echo "palette smoke already launched; persistent guard retained"
        return 0
    fi
    cp -- "$experiment" "$run_root/experiment.json"
    PALETTE_SMOKE_INTERNAL=1 nohup setsid "$SELF" _supervise \
        "$source_root" "$run_root" "$run_root/experiment.json" "$isaac_python" "$twin_stage" \
        >"$run_root/supervisor.log" 2>&1 </dev/null &
    bootstrap_pid=$!
    attempts=0
    while test ! -f "$run_root/supervisor.pid" && kill -0 "$bootstrap_pid" 2>/dev/null && test "$attempts" -lt 100
    do
        sleep 0.05
        attempts=$((attempts + 1))
    done
    test -f "$run_root/supervisor.pid" || exit 70
    echo "palette smoke launch accepted; supervisor PID $(cat "$run_root/supervisor.pid")"
}

case ${1:-} in
    launch)
        test "$#" -eq 6 || usage
        launch "$2" "$3" "$4" "$5" "$6"
        ;;
    _supervise)
        test "${PALETTE_SMOKE_INTERNAL:-0}" = 1 && test "$#" -eq 6 || exit 64
        supervise "$2" "$3" "$4" "$5" "$6"
        ;;
    _monitor)
        test "${PALETTE_SMOKE_INTERNAL:-0}" = 1 && test "$#" -eq 4 || exit 64
        monitor "$2" "$3" "$4"
        ;;
    *) usage ;;
esac
