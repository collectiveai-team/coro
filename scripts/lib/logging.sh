#!/bin/bash
# Logging module for ralph.sh
# Consistent, colored logging for different message types.
# Dependencies: constants.sh
#
# Usage:
#   log_info "Starting process..."
#   log_success "Task completed"
#   log_warn "This might take a while"
#   log_error "Something went wrong"

# Log info message with blue [INFO] prefix
log_info() {
  echo -e "${B}[INFO]${R} $1"
}

# Log success message with green [OK] prefix
log_success() {
  echo -e "${GR}[OK]${R} $1"
}

# Log warning message with yellow [WARN] prefix
log_warn() {
  echo -e "${Y}[WARN]${R} $1"
}

# Log error message with red [ERROR] prefix to stderr
log_error() {
  echo -e "${RD}[ERROR]${R} $1" >&2
}
