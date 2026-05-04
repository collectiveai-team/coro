#!/bin/bash
# Output module for ralph.sh
# JSON parsing and ANSI stripping utilities
# Dependencies: constants.sh, terminal.sh

# Parse JSON stream and extract text content
# Handles Agent's stream-json format
parse_json_content() {
  local json_line="$1"
  # Try to extract text content from various JSON formats
  # For example, Claude stream-json outputs {"type":"content_block_delta","delta":{"text":"..."}}
  # or {"type":"text","text":"..."} etc.

  # Extract text from delta (use jq if available for proper escaped quote handling)
  local text=""
  if command -v jq &>/dev/null; then
    text=$(echo "$json_line" | jq -r '.delta.text // .text // empty' 2>/dev/null)
  else
    text=$(echo "$json_line" | grep -o '"text":"[^"]*"' | head -1 | LC_ALL=C sed 's/"text":"//;s/"$//' 2>/dev/null)
  fi
  if [ -n "$text" ]; then
    # Unescape common JSON escapes (only needed for grep fallback, jq handles this)
    if ! command -v jq &>/dev/null; then
      text=$(echo "$text" | LC_ALL=C sed 's/\\n/\'$'\n''/g; s/\\t/\'$'\t''/g; s/\\"/"/g; s/\\\\/\\/g' 2>/dev/null)
    fi
    echo "$text"
    return
  fi

  # Extract tool_use events (have "name" but no "text")
  # Produces strings like "Read file_path=/src/main.ts" or "Bash command=npm test"
  # that detect_step() patterns can match
  local tool_name=$(echo "$json_line" | grep -o '"name":"[^"]*"' | head -1 | LC_ALL=C sed 's/"name":"//;s/"$//' 2>/dev/null)
  if [ -n "$tool_name" ]; then
    local file_path=$(echo "$json_line" | grep -o '"file_path":"[^"]*"' | head -1 | LC_ALL=C sed 's/"file_path":"//;s/"$//' 2>/dev/null)
    local pattern=$(echo "$json_line" | grep -o '"pattern":"[^"]*"' | head -1 | LC_ALL=C sed 's/"pattern":"//;s/"$//' 2>/dev/null)
    local cmd=$(echo "$json_line" | grep -o '"command":"[^"]*"' | head -1 | LC_ALL=C sed 's/"command":"//;s/"$//' 2>/dev/null)
    if [ -n "$file_path" ]; then
      echo "$tool_name file_path=$file_path"
    elif [ -n "$pattern" ]; then
      echo "$tool_name pattern=$pattern"
    elif [ -n "$cmd" ]; then
      echo "$tool_name command=$cmd"
    else
      echo "$tool_name"
    fi
    return
  fi

  # If it doesn't look like JSON, return as-is
  if ! echo "$json_line" | grep -q '^{'; then
    echo "$json_line"
  fi
}

# Strip ANSI escape sequences and control characters from text
# Usage: clean_text=$(strip_ansi "$text")
# Handles:
#   - CSI sequences: ESC[...m (colors), ESC[...H (cursor), ESC[?...h/l (modes)
#   - OSC sequences: ESC]0;...(BEL or ESC\) for window titles
#   - OSC-like sequences without ESC prefix (from script command)
#   - Caret notation for control chars: ^@ through ^_ (from script command)
#   - Other escapes: ESC followed by various characters
#   - Control chars: backspace, BEL, carriage return, etc.
strip_ansi() {
  local input="$1"
  # First use sed to handle ESC sequences (must be done before tr removes ESC)
  # Then use tr to remove remaining raw control characters
  # tr removes: 0x00-0x08 (NUL through BS), 0x0B-0x0C (VT, FF), 0x0E-0x1A (SO through SUB),
  #             0x1C-0x1F (FS through US) - excludes ESC (0x1B) for sed to process
  # sed handles: ESC sequences, OSC-like patterns, caret notation
  # Note: OSC pattern uses ^ anchor but also handles after caret removal via second pass
  echo "$input" | LC_ALL=C sed \
    -e 's/\x1b\[[0-9;?]*[A-Za-z]//g' \
    -e 's/\x1b\][^\x07]*\x07//g' \
    -e 's/\x1b\][^\x1b]*\x1b\\//g' \
    -e 's/\x1b[()][AB012]//g' \
    -e 's/\x1b[>=]//g' \
    -e 's/\x1b.//g' \
    -e 's/\^[][A-Z@\\^_]//g' \
    -e 's/^0;[^]]*]//g' \
    -e 's/<u0;//g' 2>/dev/null \
    | LC_ALL=C tr -d '\000-\010\013\014\016-\032\034-\037' 2>/dev/null \
    | LC_ALL=C sed -e 's/^0;[^]]*]//g' 2>/dev/null
}

# Strip ANSI from a file and write to output file
# Usage: strip_ansi_file "$input_file" "$output_file"
# First sed processes ESC sequences, then tr removes remaining control characters
strip_ansi_file() {
  local input_file="$1"
  local output_file="$2"
  # First sed processes ESC sequences (before tr removes anything)
  # Then tr removes remaining control characters (excluding newline 0x0A, CR 0x0D)
  # tr range excludes ESC (0x1B = octal 033) which sed already handled
  # Final sed pass cleans up OSC patterns that were hidden by control chars
  LC_ALL=C sed \
    -e 's/\x1b\[[0-9;?]*[A-Za-z]//g' \
    -e 's/\x1b\][^\x07]*\x07//g' \
    -e 's/\x1b\][^\x1b]*\x1b\\//g' \
    -e 's/\x1b[()][AB012]//g' \
    -e 's/\x1b[>=]//g' \
    -e 's/\x1b.//g' \
    -e 's/\^[][A-Z@\\^_]//g' \
    -e 's/^0;[^]]*]//g' \
    -e 's/<u0;//g' \
    "$input_file" 2>/dev/null | LC_ALL=C tr -d '\000-\010\013\014\016-\032\034-\037' 2>/dev/null \
    | LC_ALL=C sed -e 's/^0;[^]]*]//g' 2>/dev/null \
    > "$output_file"
}

# Extract final summary from JSON stream output
# Looks for the result type message which contains the final output
extract_final_summary() {
  local output_file="$1"
  local result_line=""
  local summary=""

  # Look for the result type message in the JSON stream
  if [ -f "$output_file" ]; then
    result_line=$(grep '"type":"result"' "$output_file" 2>/dev/null | tail -1)

    if [ -n "$result_line" ]; then
      # Extract the result text using jq
      if command -v jq &> /dev/null; then
        summary=$(echo "$result_line" | jq -r '.result // ""' 2>/dev/null)
      fi
    fi
  fi

  echo "$summary"
}

# Display the final summary with simple separators
display_final_summary() {
  local summary="$1"
  local max_lines="${2:-15}"  # Default to max 15 lines

  if [ -z "$summary" ]; then
    return
  fi

  # Display with simple separators
  echo ""
  echo -e "${C}─────────────────────────────────────────────────────────────────${R}"
  echo -e "${Y}Iteration Summary${R}"
  echo -e "${C}─────────────────────────────────────────────────────────────────${R}"

  # Wrap and display lines (max 70 chars wide, max 15 lines)
  local line_count=0
  local width=70
  while IFS= read -r line || [ -n "$line" ]; do
    while [ ${#line} -gt $width ] && [ $line_count -lt $max_lines ]; do
      # Find last space within width for word wrap
      local cut_at=$width
      local segment="${line:0:$width}"
      local last_space="${segment% *}"
      if [ "$last_space" != "$segment" ] && [ ${#last_space} -gt 20 ]; then
        cut_at=${#last_space}
      fi
      echo "${line:0:$cut_at}"
      line="${line:$cut_at}"
      line="${line# }"  # Remove leading space
      ((line_count++))
    done
    if [ $line_count -lt $max_lines ] && [ -n "$line" ]; then
      echo "$line"
      ((line_count++))
    fi
  done <<< "$summary"

  echo -e "${C}─────────────────────────────────────────────────────────────────${R}"
}
