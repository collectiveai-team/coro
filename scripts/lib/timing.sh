#!/bin/bash
# Timing module for ralph.sh
# Step timing tracking and duration formatting
# Dependencies: constants.sh
#
# Steps: Thinking, Planning, Reading code, Web research, Implementing, Debugging,
#        Writing tests, Testing, Linting, Typechecking, Installing, Verifying,
#        Waiting, Committing

# Step timing tracking (using indexed arrays for bash 3.x compatibility)
STEP_NAMES=("Thinking" "Planning" "Reading code" "Web research" "Implementing" "Debugging" "Writing tests" "Testing" "Linting" "Typechecking" "Installing" "Verifying" "Waiting" "Committing")
ITERATION_STEP_VALUES=(0 0 0 0 0 0 0 0 0 0 0 0 0 0)   # Step times for current iteration
SESSION_STEP_VALUES=(0 0 0 0 0 0 0 0 0 0 0 0 0 0)     # Accumulated step times across all iterations
CURRENT_STEP_NAME=""              # Name of current step being timed
CURRENT_STEP_START=0              # Timestamp when current step started

# Get step emoji by name (bash 3.x compatible)
get_step_emoji() {
  case "$1" in
    "Thinking") echo "🤔" ;;
    "Planning") echo "🗺️" ;;
    "Reading code") echo "📖" ;;
    "Web research") echo "🌐" ;;
    "Implementing") echo "⚡" ;;
    "Debugging") echo "🐛" ;;
    "Writing tests") echo "✍️" ;;
    "Testing") echo "🧪" ;;
    "Linting") echo "🧹" ;;
    "Typechecking") echo "📝" ;;
    "Installing") echo "📦" ;;
    "Verifying") echo "✅" ;;
    "Waiting") echo "⏳" ;;
    "Committing") echo "🚀" ;;
    *) echo "" ;;
  esac
}

# Get step index by name (-1 if not found)
get_step_index() {
  local name="$1"
  for i in "${!STEP_NAMES[@]}"; do
    if [ "${STEP_NAMES[$i]}" = "$name" ]; then
      echo "$i"
      return
    fi
  done
  echo "-1"
}

# Format duration in seconds to MM:SS or HH:MM:SS
format_duration() {
  local seconds=$1
  local hours=$((seconds / 3600))
  local mins=$(((seconds % 3600) / 60))
  local secs=$((seconds % 60))

  if [ $hours -gt 0 ]; then
    printf "%02d:%02d:%02d" $hours $mins $secs
  else
    printf "%02d:%02d" $mins $secs
  fi
}

# Format delta with color coding (stock market style)
# Returns: colored string with +/- prefix, or empty if first iteration
format_delta() {
  local current=$1
  local previous=$2

  # Skip if first iteration (no previous time)
  if [ "$previous" -eq 0 ]; then
    echo ""
    return
  fi

  local delta=$((current - previous))

  if [ $delta -lt 0 ]; then
    # Faster (green with - prefix)
    local abs_delta=$((-delta))
    printf "${GR}-%ds${R}" $abs_delta
  elif [ $delta -gt 0 ]; then
    # Slower (red with + prefix)
    printf "${RD}+%ds${R}" $delta
  else
    # Same time
    printf "~"
  fi
}

# Record time spent in current step when transitioning to a new step
# Usage: record_step_time "new_step_name"
# Pass empty string to finalize the last step without starting a new one
record_step_time() {
  local new_step="$1"

  # If we have a current step, record its elapsed time
  if [ -n "$CURRENT_STEP_NAME" ] && [ "$CURRENT_STEP_START" -gt 0 ]; then
    local now=$(date +%s)
    local elapsed=$((now - CURRENT_STEP_START))

    # Only record if there's actual time (more than 0 seconds)
    if [ "$elapsed" -gt 0 ]; then
      local idx=$(get_step_index "$CURRENT_STEP_NAME")
      if [ "$idx" -ge 0 ]; then
        # Add to iteration step times
        local current_time=${ITERATION_STEP_VALUES[$idx]}
        ITERATION_STEP_VALUES[$idx]=$((current_time + elapsed))

        # Add to session step times
        local session_time=${SESSION_STEP_VALUES[$idx]}
        SESSION_STEP_VALUES[$idx]=$((session_time + elapsed))
      fi
    fi
  fi

  # Start timing the new step (if provided)
  if [ -n "$new_step" ]; then
    CURRENT_STEP_NAME="$new_step"
    CURRENT_STEP_START=$(date +%s)
  else
    # No new step - clear tracking
    CURRENT_STEP_NAME=""
    CURRENT_STEP_START=0
  fi
}

# Format step duration for display
# Usage: format_step_duration 45  # -> "45s"
# Usage: format_step_duration 125 # -> "02:05"
format_step_duration() {
  local seconds=$1
  if [ "$seconds" -lt 60 ]; then
    echo "${seconds}s"
  else
    local mins=$((seconds / 60))
    local secs=$((seconds % 60))
    printf "%02d:%02d" $mins $secs
  fi
}

# Format step times as inline display string
# Usage: format_step_times "ITERATION" or "SESSION"
# Returns: "│ 🧪 Testing: 45s │ 🧹 Linting: 12s" (sorted by duration descending)
format_step_times() {
  local mode="$1"
  local output=""
  local sorted_steps=()

  # Build array of "duration:step_name" for sorting
  for i in "${!STEP_NAMES[@]}"; do
    local step="${STEP_NAMES[$i]}"
    local time
    if [ "$mode" = "SESSION" ]; then
      time=${SESSION_STEP_VALUES[$i]}
    else
      time=${ITERATION_STEP_VALUES[$i]}
    fi
    if [ "$time" -gt 0 ]; then
      sorted_steps+=("$time:$step")
    fi
  done

  # Sort by duration descending (reverse numeric sort on first field)
  if [ ${#sorted_steps[@]} -gt 0 ]; then
    IFS=$'\n' sorted_steps=($(sort -t: -k1 -rn <<<"${sorted_steps[*]}"))
    unset IFS
  fi

  # Build output string
  for item in "${sorted_steps[@]}"; do
    local time="${item%%:*}"
    local step="${item#*:}"
    local emoji=$(get_step_emoji "$step")
    local formatted=$(format_step_duration $time)

    if [ -n "$output" ]; then
      output="$output ${C}│${R} "
    fi
    output="$output$emoji $step: ${Y}$formatted${R}"
  done

  echo "$output"
}

# Initialize iteration step times at start of new iteration
# Usage: init_iteration_step_times
init_iteration_step_times() {
  # Clear iteration step times (reset all to 0)
  ITERATION_STEP_VALUES=(0 0 0 0 0 0 0 0 0 0 0 0 0 0)

  # Start timing with "Thinking" as default initial step
  CURRENT_STEP_NAME="Thinking"
  CURRENT_STEP_START=$(date +%s)
}

# Display session step totals at end of run
# Usage: display_session_step_totals
display_session_step_totals() {
  local step_output=$(format_step_times "SESSION")

  if [ -n "$step_output" ]; then
    echo -e "  📊 Session totals: $step_output"
  fi
}

# Detect current step from output line
# Returns: step name based on output patterns (exact match to STEP_NAMES)
# Priority: Implementing first, then other steps in order of specificity
detect_step() {
  local line="$1"

  # IMPLEMENTING - highest priority (includes building)
  # Tool-based: Write, Edit tools with file paths
  # Natural: creating, writing, editing, modifying, updating files
  # Building: npm run build, vite build, compiling, bundling
  if echo "$line" | grep -qiE "(Write|Edit).*file_path"; then
    echo "Implementing"
  elif echo "$line" | grep -qiE "(creating|writing new|editing|modifying|updating|changing).*\.(ts|tsx|js|jsx|sh|py|go|rs|json|yaml|yml|toml|css|scss|html)"; then
    echo "Implementing"
  elif echo "$line" | grep -qiE "(npm|yarn|pnpm|bun) run build"; then
    echo "Implementing"
  elif echo "$line" | grep -qiE "(vite|webpack|esbuild|rollup|turbo|tsc) build"; then
    echo "Implementing"
  elif echo "$line" | grep -qiE "(compiling|bundling|transpiling)"; then
    echo "Implementing"
  elif echo "$line" | grep -qiE "Bash.*command=.*(build|compile|bundle)"; then
    echo "Implementing"

  # COMMITTING - git operations
  elif echo "$line" | grep -qiE "(git commit|git add|committing|staged for commit)"; then
    echo "Committing"

  # TESTING - running test suites
  elif echo "$line" | grep -qiE "(npm|yarn|pnpm) (run )?(test|e2e|spec)"; then
    echo "Testing"
  elif echo "$line" | grep -qiE "(jest|vitest|playwright|cypress|mocha|pytest)"; then
    echo "Testing"
  elif echo "$line" | grep -qiE "(test|spec).*(pass|fail|skip|pending)"; then
    echo "Testing"
  elif echo "$line" | grep -qiE "(running|executing) (tests|test suite)"; then
    echo "Testing"
  elif echo "$line" | grep -qiE "Bash.*command=.*(test|e2e|spec|jest|vitest|playwright|pytest)"; then
    echo "Testing"

  # DEBUGGING - investigating errors, fixing issues
  elif echo "$line" | grep -qiE "(the|this) (error|issue|problem|bug) (is|seems|appears|was)"; then
    echo "Debugging"
  elif echo "$line" | grep -qiE "(investigating|debugging|diagnosing|troubleshooting)"; then
    echo "Debugging"
  elif echo "$line" | grep -qiE "(fails|failed|failing|broken) because"; then
    echo "Debugging"
  elif echo "$line" | grep -qiE "(let me|i'll) (check|see|figure out|understand) why"; then
    echo "Debugging"
  elif echo "$line" | grep -qiE "(root cause|stack trace|traceback|exception)"; then
    echo "Debugging"

  # LINTING - code style and formatting
  elif echo "$line" | grep -qiE "(eslint|biome|lint|prettier|formatting|stylelint)"; then
    echo "Linting"
  elif echo "$line" | grep -qiE "Bash.*command=.*(lint|eslint|biome|prettier|stylelint)"; then
    echo "Linting"

  # TYPECHECKING - type validation
  elif echo "$line" | grep -qiE "(npm run typecheck|tsc|typescript|type.?check|mypy|pyright)"; then
    echo "Typechecking"
  elif echo "$line" | grep -qiE "Bash.*command=.*(typecheck|tsc|mypy|pyright)"; then
    echo "Typechecking"

  # WRITING TESTS - creating test files
  elif echo "$line" | grep -qiE "(\.test\.|\.spec\.|test file|writing test|adding test)"; then
    echo "Writing tests"
  elif echo "$line" | grep -qiE "(creating|writing).*(test|spec)"; then
    echo "Writing tests"

  # INSTALLING - package/dependency management
  elif echo "$line" | grep -qiE "(npm|yarn|pnpm|bun) (install|add|i )"; then
    echo "Installing"
  elif echo "$line" | grep -qiE "(pip|poetry|cargo|go get|brew) install"; then
    echo "Installing"
  elif echo "$line" | grep -qiE "(installing|adding|updating) (dependency|dependencies|package)"; then
    echo "Installing"
  elif echo "$line" | grep -qiE "Bash.*command=.*(npm install|yarn add|pnpm add|pip install|brew install)"; then
    echo "Installing"

  # WEB RESEARCH - fetching docs, searching web
  elif echo "$line" | grep -qiE "(WebFetch|WebSearch)"; then
    echo "Web research"
  elif echo "$line" | grep -qiE "(fetching|looking up|searching).*(docs|documentation|api|web)"; then
    echo "Web research"
  elif echo "$line" | grep -qiE "(let me|i'll) search (for|the web|online)"; then
    echo "Web research"

  # VERIFYING - checking work, validation
  elif echo "$line" | grep -qiE "(verifying|confirming|validating) (the|that|it)"; then
    echo "Verifying"
  elif echo "$line" | grep -qiE "(let me|i'll) (verify|confirm|make sure|double.?check)"; then
    echo "Verifying"
  elif echo "$line" | grep -qiE "(looks correct|works as expected|successful)"; then
    echo "Verifying"

  # WAITING - blocked on user input
  elif echo "$line" | grep -qiE "(AskUserQuestion|waiting for|blocked on)"; then
    echo "Waiting"
  elif echo "$line" | grep -qiE "(need|require|awaiting) (input|clarification|confirmation|response)"; then
    echo "Waiting"

  # PLANNING - designing approach
  elif echo "$line" | grep -qiE "(EnterPlanMode|ExitPlanMode|plan mode)"; then
    echo "Planning"
  elif echo "$line" | grep -qiE "(let me|i'll|i need to) (plan|outline|design|architect)"; then
    echo "Planning"
  elif echo "$line" | grep -qiE "(my|the) (approach|strategy|plan) (is|will be)"; then
    echo "Planning"
  elif echo "$line" | grep -qiE "(step [0-9]|first,|second,|finally,).*(i should|i need|i will|we need)"; then
    echo "Planning"

  # READING CODE - exploring codebase
  elif echo "$line" | grep -qiE "Read.*file_path|Glob.*pattern|Grep.*pattern"; then
    echo "Reading code"
  elif echo "$line" | grep -qiE "(let me|i'll) (read|examine|look at|inspect|check) (the|this)"; then
    echo "Reading code"
  elif echo "$line" | grep -qiE "(searching|finding|looking) (for|at|through)"; then
    echo "Reading code"
  elif echo "$line" | grep -qiE "(reviewing|scanning|analyzing) (the )?(code|file|implementation|codebase)"; then
    echo "Reading code"

  # THINKING - deliberation, analysis (lowest priority - default fallback category)
  elif echo "$line" | grep -qiE "(let me|i need to|i'll) (think|consider|analyze|understand)"; then
    echo "Thinking"
  elif echo "$line" | grep -qiE "(hmm|interesting|notably|this suggests|the question is)"; then
    echo "Thinking"

  else
    echo ""
  fi
}
