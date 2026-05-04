#!/bin/bash
# Preview module for ralph.sh
# Rolling preview buffer display
# Dependencies: constants.sh, terminal.sh

# Rolling preview settings
PREVIEW_LINES=50
PREVIEW_WIDTH=80
LINE_BUFFER=()
LINES_DISPLAYED=0
ANSI_SUPPORTED=true

# Initialize the rolling preview display
init_rolling_preview() {
  LINE_BUFFER=()
  LINES_DISPLAYED=0
  PREVIEW_WIDTH=$(get_terminal_width)
}

# Add line to rolling buffer and update display
add_to_rolling_preview() {
  local line="$1"

  # Skip empty lines
  [ -z "$line" ] && return

  # Truncate to terminal width
  line=$(truncate_line "$line" $PREVIEW_WIDTH)

  # Add to buffer
  LINE_BUFFER+=("$line")

  # Keep only last PREVIEW_LINES
  if [ ${#LINE_BUFFER[@]} -gt $PREVIEW_LINES ]; then
    LINE_BUFFER=("${LINE_BUFFER[@]:1}")
  fi

  # Update display
  if [ "$ANSI_SUPPORTED" = true ]; then
    redraw_rolling_preview
  else
    # Fallback: just print the line
    echo "$line"
  fi
}

# Redraw the rolling preview using ANSI cursor control
redraw_rolling_preview() {
  # Move cursor up to start of preview area
  if [ $LINES_DISPLAYED -gt 0 ]; then
    printf "\033[%dA" $LINES_DISPLAYED
  fi

  # Clear and redraw each line
  local count=0
  for line in "${LINE_BUFFER[@]}"; do
    printf "\033[K%s\n" "$line"
    count=$((count + 1))
  done

  # Update count of displayed lines
  LINES_DISPLAYED=$count
}

# Clear the rolling preview area
clear_rolling_preview() {
  if [ "$ANSI_SUPPORTED" = true ] && [ $LINES_DISPLAYED -gt 0 ]; then
    # Move up and clear each line
    printf "\033[%dA" $LINES_DISPLAYED
    for ((i=0; i<$LINES_DISPLAYED; i++)); do
      printf "\033[K\n"
    done
    printf "\033[%dA" $LINES_DISPLAYED
  fi
  LINE_BUFFER=()
  LINES_DISPLAYED=0
}
