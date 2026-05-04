#!/bin/bash
# Agent module for ralph.sh
# Supported Docker Sandboxes agents and command construction.
#
# Sandbox naming + login command construction has a JS twin in
# `bin/lib/sandbox-name.js` used by the install CLI to precompute names
# at the end of `npx @pageai/ralph-loop`. Both implementations MUST
# produce byte-identical sandbox names for the same (agent, project)
# pair. See RALPH.md ("Keeping Bash and JavaScript in sync") and the
# parity test in `tests/test-sandbox-name-js.js`.

# Mirrored in bin/lib/sandbox-name.js (SUPPORTED_AGENTS).
SUPPORTED_AGENTS="claude codex copilot cursor gemini opencode"

# Print the supported agent list for help and validation messages.
supported_agents_list() {
  echo "$SUPPORTED_AGENTS"
}

# Normalize user-provided agent names for case-insensitive matching.
normalize_agent_name() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

# Return success when the agent slug is one Ralph knows how to run.
is_supported_agent() {
  case "$1" in
    claude|codex|copilot|cursor|gemini|opencode)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

# Human-friendly agent name for status and error messages.
agent_display_name() {
  case "$1" in
    claude) echo "Claude" ;;
    codex) echo "Codex" ;;
    copilot) echo "Copilot" ;;
    cursor) echo "Cursor" ;;
    gemini) echo "Gemini" ;;
    opencode) echo "OpenCode" ;;
    *) echo "$1" ;;
  esac
}

# Claude's stream-json output is currently the only fully parsed output path.
agent_uses_stream_json() {
  [ "$1" = "claude" ]
}

# Authentication command users can run before starting Ralph.
# Mirrored in bin/lib/sandbox-name.js (agentLoginCommand).
agent_login_command() {
  local agent="${1:-claude}"
  local sandbox_name="${2:-}"

  if [ -n "$sandbox_name" ]; then
    echo "sbx run --name $sandbox_name $agent ."
  else
    echo "sbx run $agent ."
  fi
}

# Print every supported agent login command for this project.
print_login_suggestions() {
  local selected_agent="${1:-claude}"
  local project_path="${2:-$PWD}"
  local agent
  local sandbox_name
  local command
  local label

  echo -e "  ${Y}Login commands for this project:${R}"
  for agent in $(supported_agents_list); do
    sandbox_name=$(build_sandbox_name "$agent" "$project_path")
    command=$(agent_login_command "$agent" "$sandbox_name")
    label="$(agent_display_name "$agent")"

    if [ "$agent" = "$selected_agent" ]; then
      echo -e "  ${GR}→ ${label}:${R} ${C}${command}${R}"
    else
      echo -e "    ${label}: ${D}${command}${R}"
    fi
  done
}

# Authentication failure patterns seen across supported agent CLIs.
agent_auth_error_patterns() {
  case "$1" in
    claude)
      echo "Invalid API key|Please run /login|not authenticated|authentication required"
      ;;
    codex)
      echo "codex login|not authenticated|authentication required|Unauthorized|Invalid API key"
      ;;
    copilot)
      echo "gh auth login|not authenticated|authentication required|Unauthorized"
      ;;
    cursor)
      echo "cursor login|not authenticated|authentication required|Unauthorized|Invalid API key"
      ;;
    gemini)
      echo "gemini auth|not authenticated|authentication required|API key not valid|Invalid API key"
      ;;
    opencode)
      echo "opencode auth login|not authenticated|authentication required|Unauthorized|Invalid API key"
      ;;
    *)
      echo "Invalid API key|not authenticated|authentication required|Unauthorized"
      ;;
  esac
}

# Convert project and agent values into Docker Sandboxes-safe name segments.
# Mirrored in bin/lib/sandbox-name.js (sanitizeSandboxNameSegment).
sanitize_sandbox_name_segment() {
  local value="${1:-sandbox}"

  value=$(printf "%s" "$value" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')

  if [ -z "$value" ]; then
    value="sandbox"
  fi

  printf "%s" "$value"
}

# Return an 8-character deterministic hash for an absolute project path.
# Mirrored in bin/lib/sandbox-name.js (sandboxPathHash). The JS side
# uses Node's crypto.createHash('sha256'); this side prefers `shasum`,
# falls back to `sha256sum`, and finally to `cksum` (which would NOT
# match the JS output, so prefer environments where shasum/sha256sum
# is available).
sandbox_path_hash() {
  local project_path="$1"

  if command -v shasum >/dev/null 2>&1; then
    printf "%s" "$project_path" | shasum -a 256 | awk '{print substr($1, 1, 8)}'
  elif command -v sha256sum >/dev/null 2>&1; then
    printf "%s" "$project_path" | sha256sum | awk '{print substr($1, 1, 8)}'
  else
    printf "%s" "$project_path" | cksum | awk '{printf "%08x", $1}'
  fi
}

# Build Ralph's deterministic Docker Sandboxes name for this project and agent.
# Mirrored in bin/lib/sandbox-name.js (buildSandboxName).
build_sandbox_name() {
  local agent="${1:-claude}"
  local project_path="${2:-$PWD}"
  local safe_agent
  local safe_project
  local hash

  safe_agent=$(sanitize_sandbox_name_segment "$agent")
  safe_project=$(sanitize_sandbox_name_segment "$(basename "$project_path")")
  hash=$(sandbox_path_hash "$project_path")

  printf "ralph-%s-%s-%s" "$safe_agent" "$safe_project" "$hash"
}

# Shell-quote any extra args passed after Ralph's own -- separator.
format_agent_extra_args() {
  local output=""
  local arg
  local quoted

  for arg in "${AGENT_EXTRA_ARGS[@]}"; do
    printf -v quoted "%q" "$arg"
    output="$output $quoted"
  done

  printf "%s" "$output"
}

# Build the command executed inside script(1)'s pseudo-TTY.
build_agent_command() {
  local agent="${1:-claude}"
  local sandbox_name="${2:-}"
  local extra_args
  local sbx_run
  extra_args=$(format_agent_extra_args)

  if [ -n "$sandbox_name" ]; then
    sbx_run="sbx run --name $sandbox_name $agent ."
  else
    sbx_run="sbx run $agent ."
  fi

  case "$agent" in
    claude)
      printf '%s -- --output-format stream-json --verbose%s -p "$PROMPT_CONTENT"' "$sbx_run" "$extra_args"
      ;;
    codex)
      printf '%s -- exec%s "$PROMPT_CONTENT"' "$sbx_run" "$extra_args"
      ;;
    copilot)
      printf '%s --%s -p "$PROMPT_CONTENT"' "$sbx_run" "$extra_args"
      ;;
    cursor)
      printf '%s -- -p%s "$PROMPT_CONTENT"' "$sbx_run" "$extra_args"
      ;;
    gemini)
      printf '%s --%s -p "$PROMPT_CONTENT"' "$sbx_run" "$extra_args"
      ;;
    opencode)
      printf '%s -- run%s "$PROMPT_CONTENT"' "$sbx_run" "$extra_args"
      ;;
    *)
      return 1
      ;;
  esac
}
