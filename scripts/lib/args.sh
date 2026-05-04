#!/bin/bash
# Arguments module for ralph.sh
# CLI argument parsing
# Dependencies: constants.sh, agents.sh

# Parse command line arguments
# Sets: MAX_ITERATIONS, ONCE_FLAG, RALPH_AGENT, RALPH_ACTION, AGENT_EXTRA_ARGS
# Usage: parse_arguments "$@"
parse_arguments() {
  MAX_ITERATIONS=10
  ONCE_FLAG=false
  RALPH_AGENT="claude"
  RALPH_ACTION="run"
  AGENT_EXTRA_ARGS=()

  while [[ $# -gt 0 ]]; do
    case $1 in
      --)
        shift
        AGENT_EXTRA_ARGS=("$@")
        break
        ;;
      --help|-h)
        show_help
        ;;
      --once)
        ONCE_FLAG=true
        shift
        ;;
      --print-name)
        RALPH_ACTION="print-name"
        shift
        ;;
      --login)
        RALPH_ACTION="login"
        shift
        ;;
      --agent|-a)
        if [ -z "$2" ]; then
          echo "Error: --agent requires a value" >&2
          exit 1
        fi
        RALPH_AGENT=$(normalize_agent_name "$2")
        shift 2
        ;;
      --agent=*)
        RALPH_AGENT=$(normalize_agent_name "${1#*=}")
        shift
        ;;
      --max-iterations|-n)
        if [ -z "$2" ]; then
          echo "Error: --max-iterations requires a value" >&2
          exit 1
        fi
        MAX_ITERATIONS="$2"
        shift 2
        ;;
      --max-iterations=*)
        MAX_ITERATIONS="${1#*=}"
        shift
        ;;
      [0-9]*)
        MAX_ITERATIONS="$1"
        shift
        ;;
      *)
        shift
        ;;
    esac
  done

  # --once overrides max-iterations
  if [ "$ONCE_FLAG" = true ]; then
    MAX_ITERATIONS=1
  fi

  if ! is_supported_agent "$RALPH_AGENT"; then
    echo "Error: unsupported agent '$RALPH_AGENT'" >&2
    echo "Supported agents: $(supported_agents_list)" >&2
    exit 1
  fi
}
