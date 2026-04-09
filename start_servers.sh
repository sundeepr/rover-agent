#!/usr/bin/env bash
# start_servers.sh — launch web server and OmniVLA model server
#
# Both servers run as background processes with logs written to
# logs/web_server.log and logs/omnivla_server.log.
#
# Usage:
#   ./start_servers.sh            # start both
#   ./start_servers.sh --stop     # kill both
#   ./start_servers.sh --status   # show running PIDs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
WEB_PIDFILE="$SCRIPT_DIR/.web_server.pid"
OMNIVLA_PIDFILE="$SCRIPT_DIR/.omnivla_server.pid"

mkdir -p "$LOG_DIR"

# ── Helpers ────────────────────────────────────────────────────────────────────

is_running() {
    local pidfile="$1"
    [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null
}

stop_server() {
    local name="$1" pidfile="$2"
    if is_running "$pidfile"; then
        kill "$(cat "$pidfile")"
        rm -f "$pidfile"
        echo "Stopped $name"
    else
        echo "$name is not running"
        rm -f "$pidfile"
    fi
}

start_server() {
    local name="$1" pidfile="$2" logfile="$3"
    shift 3
    if is_running "$pidfile"; then
        echo "$name already running (PID $(cat "$pidfile"))"
        return
    fi
    nohup python "$@" >> "$logfile" 2>&1 &
    echo $! > "$pidfile"
    echo "Started $name (PID $!) — logs: $logfile"
}

# ── Commands ───────────────────────────────────────────────────────────────────

cmd="${1:-start}"

case "$cmd" in
    --stop)
        stop_server "web server"    "$WEB_PIDFILE"
        stop_server "OmniVLA server" "$OMNIVLA_PIDFILE"
        ;;
    --status)
        if is_running "$WEB_PIDFILE"; then
            echo "web server    : running (PID $(cat "$WEB_PIDFILE"))"
        else
            echo "web server    : stopped"
        fi
        if is_running "$OMNIVLA_PIDFILE"; then
            echo "OmniVLA server: running (PID $(cat "$OMNIVLA_PIDFILE"))"
        else
            echo "OmniVLA server: stopped"
        fi
        ;;
    *)
        cd "$SCRIPT_DIR"
        start_server "web server"     "$WEB_PIDFILE"     "$LOG_DIR/web_server.log" \
            web_server.py
        start_server "OmniVLA server" "$OMNIVLA_PIDFILE" "$LOG_DIR/omnivla_server.log" \
            omnivla_server.py
        echo
        echo "Both servers started. To stop:"
        echo "  ./start_servers.sh --stop"
        ;;
esac
