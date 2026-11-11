# Learnings

What I learned building this — useful for interviews.

## The Claude setup in this repo
- `CLAUDE.md` = shared instructions every agent reads (project rules, commands).
- `.claude/agents/*.md` = specialised agents, each with its own instructions
  and allowed tools: `code-reviewer` (read-only), `experiment-runner`,
  `results-analyst`.
- `.claude/skills/*/SKILL.md` = a repeatable procedure. `run-robustness-eval`
  chains the agents: check → review → run → analyse.
- `notes/` = the vault, long-term memory across sessions.
- Everything is versioned in git, so changes to how Claude works are tracked
  like code.

## ML / evaluation
- 
