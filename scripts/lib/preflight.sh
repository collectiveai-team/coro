#!/bin/bash
# Preflight module for ralph.sh
# Pre-flight checks before starting the main loop
# Dependencies: constants.sh, logging.sh
#
# Verify required files exist before starting the main loop.
# Required files cause script exit if missing.
# Optional files trigger a warning but allow continuation.

# Check if running inside a git repository
# Usage: check_git_repo
# Exits with code 1 if not in a git repository
check_git_repo() {
  if ! git rev-parse --git-dir > /dev/null 2>&1; then
    log_error "ralph.sh must be run inside a git repository"
    exit 1
  fi
}

# Ensure history directory exists for storing iteration logs
# Usage: check_history_dir
# Creates .agent/history/ directory if it doesn't exist
check_history_dir() {
  mkdir -p "$HISTORY_DIR"
  if [ ! -d "$HISTORY_DIR" ]; then
    log_error "Failed to create history directory: $HISTORY_DIR"
    exit 1
  fi
}

# Check for required files and exit if missing, warn for optional files
# Usage: check_required_files
check_required_files() {
  local missing_required=false

  # Required files - exit if missing
  if [ ! -f "$SCRIPT_DIR/.agent/tasks.json" ]; then
    log_error "Required file missing: .agent/tasks.json"
    missing_required=true
  fi

  if [ ! -f "$SCRIPT_DIR/.agent/PROMPT.md" ]; then
    log_error "Required file missing: .agent/PROMPT.md"
    missing_required=true
  fi

  # Exit if any required files are missing
  if [ "$missing_required" = true ]; then
    log_error "Please create the required files before running Ralph"
    exit 1
  fi

  # Optional files - warn if missing but continue
  if [ ! -f "$SCRIPT_DIR/.agent/prd/SUMMARY.md" ]; then
    log_warn "Optional file missing: .agent/prd/SUMMARY.md"
  fi
}
