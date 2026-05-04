git update-index --skip-worktree \
  .agent/prd/PRD.md \
  .agent/prd/SUMMARY.md \
  .agent/tasks.json

git ls-files '.agent/tasks/*.json' \
  | xargs -r git update-index --skip-worktree