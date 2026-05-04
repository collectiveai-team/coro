#!/bin/bash
# Display module for ralph.sh
# UI elements, ASCII art, and help display
# Dependencies: constants.sh, timing.sh

# Display BLOCKED message with reason and resume instructions
# Usage: display_blocked_message "reason" iteration_number
display_blocked_message() {
  local reason="$1"
  local iteration="$2"

  echo ""
  echo -e "${RD}░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░${R}"
  echo -e "  🚫 ${RD}Agent is BLOCKED${R}"
  echo -e "${RD}░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░${R}"
  echo ""
  echo -e "  ${Y}Reason:${R}"
  echo -e "    $reason"
  echo ""
  echo -e "  ${C}How to resume:${R}"
  echo -e "    1. Resolve the blocking issue described above"
  echo -e "    2. Run ${GR}./ralph.sh${R} to continue from where you left off"
  echo ""
  echo -e "  ${G}Stopped at iteration ${Y}$iteration${R}"
  echo ""
}

# Display DECIDE message with question and resume instructions
# Usage: display_decide_message "question" iteration_number
display_decide_message() {
  local question="$1"
  local iteration="$2"

  echo ""
  echo -e "${M}░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░${R}"
  echo -e "  ❓ ${M}Agent needs a DECISION${R}"
  echo -e "${M}░░▒▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒░░${R}"
  echo ""
  echo -e "  ${Y}Question:${R}"
  echo -e "    $question"
  echo ""
  echo -e "  ${C}How to answer and resume:${R}"
  echo -e "    1. Make a decision about the question above"
  echo -e "    2. Update the relevant files or configuration"
  echo -e "    3. Run ${GR}./ralph.sh${R} to continue with your decision"
  echo ""
  echo -e "  ${G}Stopped at iteration ${Y}$iteration${R}"
  echo ""
}

# Ralph ASCII art and catchphrases
show_ralph() {

  local catchphrases=(
    "■▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ I'm helping! ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ I'm doing my best! ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ I'm in danger! ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ I'm learnding! ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒▒ My cat's breath smells like cat food. ▒▒▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒▒▒ Me fail English? That's unpossible! ▒▒▒▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒▒▒▒▒ I'm asking Claude to cook pasta! ▒▒▒▒▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒▒▒▒▒ I found a moon rock in my nose! ▒▒▒▒▒▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒▒▒▒▒▒▒▒▒ It tastes like burning! ▒▒▒▒▒▒▒▒▒▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒ When I grow up, I want to be a computer! ▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ I'm a develotron! ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ I'm helpding AI! ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ I'm essential! ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒■▒▒▓■"
    "■▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒■▒▒▓■"
  )

  local random_index=$((RANDOM % 15))
  local phrase="${catchphrases[$random_index]}"

  echo ""
  echo -e "${Y}"
  cat << 'RALPH'

██████╗  █████╗ ██╗     ██████╗ ██╗  ██╗
██╔══██╗██╔══██╗██║     ██╔══██╗██║  ██║
██████╔╝███████║██║     ██████╔╝███████║
██╔══██╗██╔══██║██║     ██╔═══╝ ██╔══██║
██║  ██║██║  ██║███████╗██║     ██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝

██╗      ██████╗  ██████╗ ██████╗
██║     ██╔═══██╗██╔═══██╗██╔══██╗
██║     ██║   ██║██║   ██║██████╔╝
██║     ██║   ██║██║   ██║██╔═══╝
███████╗╚██████╔╝╚██████╔╝██║
╚══════╝ ╚═════╝  ╚═════╝ ╚═╝

■■■■■■■■■■■■■■■■■■■■■■■■■■■▓▓■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
■■■■■■■■■■■■■■■■▓▒▓▒▓▒▒▓▒▒▓▢▒▒▓▒▓▒▓▓■■■■■■■■■■■■■■■■■■■■■■■
■■■■■■■■■■■▓▓▓▒▓▒▒▓▒▒▓▒▒▓▒▢▓▒▢▓▢▒▓▒▒▓▒▓▓▓▓■■■■■■■■■■■■■■■■■
■■■■■■■■■▒▢▒▒▓▒▢▓▒▢▒▓▢▒▓▢▢▒▒▢▒▓▢▢▒▒▢▢▓▒▒▓▒▓▒■■■■■■■■■■■■■■■
■■■■■■■■■▓▒▒▓▢▒▒▒▢▒▓▢▢▓▢▢▢▓▢▢▒▒▢▢▢▓▒▢▒▒▒▒▓▒▓▒▒▓▓■■■■■■■■■■■
■■■■■■■■▓▢▒▓▢▢▓▒▢▒▓▢▢▓▒▢▢▢▒▢▢▒▒▢▢▢▒▒▢▢▒▒▢▒▓▢▒▒▒▓■■■■■■■■■■■
■■■■■■■▓▒▒▓▢▢▓▒▢▢▓▢▢▢▒▢▢▢▢▢▒▒▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▒▒▓■■■■■■■■■■
■■■■■■▓▒▒▓▢▢▒▓▢▢▒▒▢▢▢▢▢▒▒▢    ▢▒▒▢▢▢▢▢▢▒▒▢   ▢▒▒▒■■■■■■■■■■
■■■■■■▓▢▓▒▢▢▓▒▢▢▢▢▢▢▢▢▒▒        ▢▒▢▢▢▢▒        ▢▒▓■■■■■■■■■
■■■■■▓▒▒▓▢▢▢▓▢▢▢▢▢▢▢▢▢▒  ▓👁️▓     ▒▢▢▢▢▒  ▓👁️▓  ▢▒▓■■■■■■■■■
■■■■▓▒▒▒▒▒▢▢▢▢▢▢▢▢▢▢▢▢▢▓        ▒▒▢▢▢▢▒▒▢     ▢▒▒▓■■■■■■■■■
■■■■▢▒▓▒▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▒▓▒  ▢▓▒▢▢▒▢▢▒▒▓▢▢▢▢▢▢▢▒▒▓■■■■■■■■■
■■■■▓▢▒▒▒▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▒▒▒▒▒▒▢▢▢▢▢▢▢▢▢▢▒▒▓■■■■■■■■
■■■■▓▢▒▒▒▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▒▒▒▒▒▒▒▒▒▒▢▢▢▢▢▢▢▢▢▒▒▓■■■■■■■
■■■■■▓▓▒▒▒▒▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▓▓▒▢▒▓▒▢▒▒▓▢▢▢▢▢▢▢▢▢▢▒▒■■■■■■■■
■■■■■▓▓▒▒▒▒▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▓▓▒▒▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▒▒■■■■■■■■
■■■■■■■■▓▒▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▒▓▒▢▢▒▒▓▒▢▢▢▢▢▢▢▢▢▢▢▢▒▒▒▓■■■■■■
■■■■■■■■▓■▒▒▢▢▢▢▢▢▢▢▢▢▢▒▒▢▢▓▒▢▢▢▒▓▓▓▓▒▒▒▢▢▢▢▢▢▢▢▒▒▒▒▓■■■■■■
■■■■■■■■■■■■▒▢▢▢▢▢▢▢▢▢▢▓▓■■▒▢▢▢▢▒▒▢▢▢▒▓▒▒▢▢▢▒▒▒▒■■■■■■■■■■■
■■■■■■■■■■▒▒▒▓▒▒▢▢▢▢▢▢▢▢▢▢▓▒▢▢▢▢▢▢▒▒▓▒▒▓▓▒▒▒▓■■■■■■■■■■■■■■
■■■■■■■■■▓▒▒▒▒▒▒▒▓▓▒▢▢▢▢▢▢▓▒▢▢▢▢▢▢▢▢▢▢▢▒▓▒▒▓■■■■■■■■■■■■■■■
■■■■■■■■■▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▒▒▓▓▒▢▢▢▢▢▢▢▒▓▒▓▓▓▒▓■■■■■■■■■■■■■■
■■■■■■▓▒▒▒▒▓▒▒▒▒▒▒▒▒▒▒▒▓▒▒▒▒▒▒▒▓▒▢▢▢▒▒▓▒▒▓▓▒▓▓▓■■■■■■■■■■■■
■■■■■▒▒▒▒▒▒▒▒▒▓▒▒▒▒▒▒▒▓▒▒▒▒▒▒▒▒▒▒▓▒▒▒▒▓▒▒▒▓▓■▒▒▓■■■■■■■■■■■
■■■▓■▒▒▒▒▒▒▒▒▒▒▒▒▓▒▒▓▒▒▒▒▒▒▒▒▒▒▒▒▒▓▒▒▓▒▒▓▓▒▒▒▒▓▓▓■■■■■■■■■■
■■■■■▒▒▒▒▒▒▒▒▒▒▒▒▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▓▓▒■■■■■■■■■
■■■■■▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒■■■■■■■■
■■■▓▢▢▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓■■■■■■■
■■▓▢▢▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓■■■■■
■▓▒▢▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓■■■■
■▓▢▢▒▒▒▒▒▒▒▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▒▓■■■
■▓▢▢▒▒▒▒▒▒▒▒▒▒▒▓▓▓▓▓▓▓▓■▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▒▓■■
■▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒■▒▓■■
RALPH

  echo -e "$phrase"
  echo -e "■■▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▓▒▓■${R}"
  echo -e "═══════════════════════════════════════════════════════════"
  echo -e " Ralph Wiggum Loop ・ Long-running AI agents"
}

# Show help and instructions
show_help() {
  show_ralph
  echo ""
  echo -e "${Y}Usage:${R} ./ralph.sh [options] [max_iterations] [-- agent_options]"
  echo ""
  echo -e "${Y}Options:${R}"
  echo "  --max-iterations N, -n N    Set maximum iterations (default: 10)"
  echo "  --agent AGENT, -a AGENT     Select agent: $(supported_agents_list) (default: claude)"
  echo "  --once                      Run exactly 1 iteration (overrides --max-iterations)"
  echo "  --login                     Show login commands, then open the selected agent sandbox"
  echo "  --print-name                Print the selected agent sandbox name and exit"
  echo "  --help, -h                  Show this help message and exit"
  echo ""
  echo -e "${Y}Arguments:${R}"
  echo "  max_iterations    Maximum number of iterations (positional, default: 10)"
  echo "  agent_options     Extra options passed to the selected agent after --"
  echo ""
  echo -e "${Y}Examples:${R}"
  echo "  ./ralph.sh                      Run with default 10 iterations"
  echo "  ./ralph.sh 5                    Run with 5 iterations max"
  echo "  ./ralph.sh -n 5                 Run with 5 iterations max"
  echo "  ./ralph.sh --max-iterations 5   Same as above"
  echo "  ./ralph.sh --agent codex        Run with Codex"
  echo "  ./ralph.sh -a gemini --once     Run one Gemini iteration"
  echo "  ./ralph.sh -a codex -- --model gpt-5.3-codex"
  echo "  ./ralph.sh --login              Log in to Claude inside Ralph's sandbox"
  echo "  ./ralph.sh --login -a cursor    Log in to Cursor inside Ralph's sandbox"
  echo "  ./ralph.sh --print-name         Print the Claude sandbox name"
  echo "  ./ralph.sh --once               Run exactly 1 iteration"
  echo ""
  echo -e "${Y}Files:${R}"
  echo "  📁 .agent/history/         Iteration output logs"
  echo "  📋 .agent/logs/LOG.md      Progress log file"
  echo "  📄 .agent/prd/PRD.md       PRD file with task definitions"
  echo "  📄 .agent/tasks/           Detailed task descriptions"
  echo "  📝 .agent/PROMPT.md        Prompt sent to the selected agent each iteration"
  echo "  📄 .agent/tasks.json       Task lookup table"
  echo ""
  echo -e "${Y}Behavior:${R}"
  echo "  🤔 Decides on what tasks to pick from .agent/tasks.json"
  echo "  📋 Logs progress to .agent/logs/LOG.md"
  echo "  🎉 Exits early if the agent outputs <promise>COMPLETE</promise>"
  echo "  🖼️ Takes screenshots of progress"
  echo ""
  echo -e "${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${R}"
  echo -e "${Y}📚 Getting Started:${R}"
  echo -e "${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${R}"
  echo ""
  echo -e "  ${C}Step 1:${R} Create a PRD using the prd-creator skill"
  echo -e "         ${G}Dump your requirements and ask AI to use the prd-creator skill${R}"
  echo ""
  echo -e "  ${C}Step 2:${R} Ensure your .agent/ directory has the required files:"
  echo "         📄 .agent/prd/PRD.md        Your product requirements"
  echo "         📄 .agent/prd/SUMMARY.md    Short project overview"
  echo "         📋 .agent/tasks.json        Generated task list"
  echo "         📁 .agent/tasks/            Individual task specs (TASK-{ID}.json)"
  echo "         📝 .agent/PROMPT.md         Agent instructions"
  echo "         📋 .agent/logs/LOG.md       Progress log (auto-created)"
  echo ""
  echo -e "  ${C}Step 3:${R} Run Ralph!"
  echo -e "         ${G}./ralph.sh${R}              # Start the Claude loop"
  echo -e "         ${G}./ralph.sh -a codex${R}     # Or choose another supported agent"
  echo ""
  echo -e "${Y}🔄 How it works:${R}"
  echo "  Each iteration, Ralph will:"
  echo "  1. Find the highest-priority incomplete task in tasks.json"
  echo "  2. Work through the task steps in .agent/tasks/TASK-{ID}.json"
  echo "  3. Run tests, linting, and type checking"
  echo "  4. Update task status and commit changes"
  echo "  5. Repeat until all tasks pass or max iterations reached"
  echo ""
  exit 0
}
