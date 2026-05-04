#!/bin/bash
# Spinner module for ralph.sh
# Animated spinner display with step tracking
# Dependencies: constants.sh, timing.sh, terminal.sh

# Spinner characters
SPINNER='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
SPINNER_PID=""
CURRENT_STEP="Thinking"

# Spinner function - runs in background, reads step from STEP_FILE
# Usage: start_spinner; do_work; stop_spinner
start_spinner() {
  echo "Thinking" > "$STEP_FILE"
  echo "" > "$PREVIEW_LINE_FILE"

  (
    # Spinner chars as array for proper unicode handling
    local -a SPIN_CHARS=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏)
    local i=0
    local spin_len=${#SPIN_CHARS[@]}
    local term_width=$(tput cols 2>/dev/null || echo 80)
    local preview_max=$((term_width - 6))  # Leave room for "    > "
    while true; do
      local step=$(cat "$STEP_FILE" 2>/dev/null || echo "Thinking")
      local preview=$(cat "$PREVIEW_LINE_FILE" 2>/dev/null || echo "")
      local char="${SPIN_CHARS[$i]}"
      # Truncate preview if needed
      if [ ${#preview} -gt $preview_max ]; then
        preview="${preview:0:$((preview_max - 3))}..."
      fi
      # Line 1: spinner + step, Line 2: dimmed preview
      printf "\r\033[K  ${C}%s${R} %s" "$char" "$step"
      if [ -n "$preview" ]; then
        printf "\n\033[K    ${D}▸ %s${R}\033[A" "$preview"
      fi
      i=$(( (i + 1) % spin_len ))
      sleep 0.1
    done
  ) &
  SPINNER_PID=$!
}

# Stop the spinner
stop_spinner() {
  if [ -n "$SPINNER_PID" ] && kill -0 "$SPINNER_PID" 2>/dev/null; then
    kill "$SPINNER_PID" 2>/dev/null
    wait "$SPINNER_PID" 2>/dev/null || true
    # Clear both lines (spinner and preview)
    printf "\r\033[K\n\033[K\033[A\r"
  fi
  SPINNER_PID=""
}

# Update spinner step based on output line and record step timing
update_spinner_step() {
  local line="$1"
  local detected=$(detect_step "$line")
  if [ -n "$detected" ]; then
    echo "$detected" > "$STEP_FILE"

    # Clean step name (remove trailing spaces) for timing tracking
    local clean_step=$(echo "$detected" | LC_ALL=C sed 's/ *$//' 2>/dev/null)

    # Record timing if step changed
    if [ "$clean_step" != "$CURRENT_STEP_NAME" ]; then
      record_step_time "$clean_step"
    fi
  fi
}

# Update the preview line shown under the spinner
update_preview_line() {
  local line="$1"
  # Skip empty lines or lines that are just whitespace
  [ -z "$line" ] && return
  [[ "$line" =~ ^[[:space:]]*$ ]] && return
  # Sanitize: replace newlines/tabs with spaces, collapse multiple spaces
  # This prevents multi-line content from breaking cursor positioning
  line=$(echo "$line" | LC_ALL=C tr '\n\t\r' ' ' 2>/dev/null | LC_ALL=C sed 's/  */ /g; s/^ *//; s/ *$//' 2>/dev/null | head -1)
  # Skip if sanitization resulted in empty string
  [ -z "$line" ] && return
  # Write to preview file (spinner reads this)
  echo "$line" > "$PREVIEW_LINE_FILE"
}
