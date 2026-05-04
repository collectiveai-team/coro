#!/bin/bash
# Terminal module for ralph.sh
# ANSI support detection and terminal width utilities
# Dependencies: constants.sh

# Check if terminal supports ANSI escape sequences
check_ansi_support() {
  # Check if $TERM is set and not dumb
  if [ -z "$TERM" ] || [ "$TERM" = "dumb" ]; then
    ANSI_SUPPORTED=false
    return
  fi
  # Check for known terminal types that don't support ANSI
  case "$TERM" in
    dumb|unknown|network)
      ANSI_SUPPORTED=false
      return
      ;;
  esac
  # Check if stdout is a terminal
  if [ ! -t 1 ]; then
    ANSI_SUPPORTED=false
    return
  fi
  ANSI_SUPPORTED=true
}

# Get terminal width (default to 80 if unavailable)
get_terminal_width() {
  if command -v tput &> /dev/null; then
    local width=$(tput cols 2>/dev/null)
    if [ -n "$width" ] && [ "$width" -gt 0 ]; then
      echo "$width"
      return
    fi
  fi
  # Fallback to stty
  if command -v stty &> /dev/null; then
    local width=$(stty size 2>/dev/null | cut -d' ' -f2)
    if [ -n "$width" ] && [ "$width" -gt 0 ]; then
      echo "$width"
      return
    fi
  fi
  echo "80"
}

# Truncate line to fit terminal width
truncate_line() {
  local line="$1"
  local max_width="$2"
  if [ ${#line} -gt $max_width ]; then
    echo "${line:0:$((max_width - 3))}..."
  else
    echo "$line"
  fi
}
