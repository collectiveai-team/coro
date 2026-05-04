#!/bin/bash
# Promise tags module for ralph.sh
# Semantic signals for agent communication
# Dependencies: None
#
# Promise tags allow the AI agent to communicate status to the loop controller.
# The script detects these tags in the agent's output to determine next actions.
#
# Tag Format: <promise>TYPE:content</promise>
#
# Available Tags:
#   <promise>COMPLETE</promise>     - All tasks finished successfully
#   <promise>BLOCKED:reason</promise>  - Agent is blocked and needs human help
#   <promise>DECIDE:question</promise> - Agent needs human decision/clarification
#   <promise>TASK-1:DONE</promise> - Task completed during this iteration
#
# Examples:
#   <promise>COMPLETE</promise>
#   <promise>BLOCKED:Missing API credentials for external service</promise>
#   <promise>DECIDE:Should we use REST or GraphQL for the new endpoint?</promise>
#   <promise>TASK-1:DONE</promise>

# Regex patterns for promise tags
PROMISE_COMPLETE_PATTERN='<promise>COMPLETE</promise>'
PROMISE_BLOCKED_PATTERN='<promise>BLOCKED:[^<]*</promise>'
PROMISE_DECIDE_PATTERN='<promise>DECIDE:[^<]*</promise>'
PROMISE_TASK_DONE_PATTERN='<promise>TASK-[A-Za-z0-9._-]+:DONE</promise>'

# Check if output contains a COMPLETE tag
# Usage: if has_complete_tag "$output"; then ...
has_complete_tag() {
  local output="$1"
  echo "$output" | grep -q "$PROMISE_COMPLETE_PATTERN"
}

# Check if output contains a BLOCKED tag
# Usage: if has_blocked_tag "$output"; then ...
has_blocked_tag() {
  local output="$1"
  echo "$output" | grep -qE "$PROMISE_BLOCKED_PATTERN"
}

# Check if output contains a DECIDE tag
# Usage: if has_decide_tag "$output"; then ...
has_decide_tag() {
  local output="$1"
  echo "$output" | grep -qE "$PROMISE_DECIDE_PATTERN"
}

# Extract the reason from a BLOCKED tag
# Usage: reason=$(extract_blocked_reason "$output")
# Returns: the reason string, or empty if no tag found
extract_blocked_reason() {
  local output="$1"
  local match=$(echo "$output" | grep -oE "$PROMISE_BLOCKED_PATTERN" | head -1)
  if [ -n "$match" ]; then
    # Extract content between BLOCKED: and </promise>
    echo "$match" | sed 's/<promise>BLOCKED://;s/<\/promise>//'
  fi
}

# Extract the question from a DECIDE tag
# Usage: question=$(extract_decide_question "$output")
# Returns: the question string, or empty if no tag found
extract_decide_question() {
  local output="$1"
  local match=$(echo "$output" | grep -oE "$PROMISE_DECIDE_PATTERN" | head -1)
  if [ -n "$match" ]; then
    # Extract content between DECIDE: and </promise>
    echo "$match" | sed 's/<promise>DECIDE://;s/<\/promise>//'
  fi
}

# Extract completed task IDs from TASK-DONE tags
# Usage: extract_completed_task_ids "$output"
# Returns: one task ID per line, de-duplicated in first-seen order
extract_completed_task_ids() {
  local output="$1"
  echo "$output" \
    | grep -oE "$PROMISE_TASK_DONE_PATTERN" \
    | sed 's/<promise>//;s/:DONE<\/promise>//' \
    | awk '!seen[$0]++'
}

# Format completed task IDs for compact display
# Usage: task_ids=$(format_completed_task_ids "$output")
# Returns: comma-separated task IDs, or empty if no task completion tags exist
format_completed_task_ids() {
  local output="$1"
  extract_completed_task_ids "$output" | awk '
    NF {
      printf "%s%s", separator, $0
      separator = ", "
    }
    END {
      if (separator != "") {
        printf "\n"
      }
    }
  '
}

# Check if output contains any help-needed tag (BLOCKED or DECIDE)
# Usage: if needs_help "$output"; then ...
needs_help() {
  local output="$1"
  has_blocked_tag "$output" || has_decide_tag "$output"
}
