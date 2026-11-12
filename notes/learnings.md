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
- **Check how the data is ordered before splitting it.** GTSRB is sorted by
  class and Keras `validation_split` takes the *last* 10%, so validation held
  classes the model never trained on (val accuracy 0.5%, below chance).
- **Log learning curves, not just final numbers.** The BatchNorm bug (98% train,
  ~5% validation) was only visible because training and validation were logged
  per epoch in MLflow.
- **Recognise what overfitting looks like.** In overfitting, validation rises
  first and then falls. A gap from the very start points to a train/inference
  difference instead (BatchNorm, dropout, preprocessing).
- **BatchNorm behaves differently in training and at inference.** It uses batch
  statistics while training and running averages at inference. With few
  training steps the running averages need a lower momentum to keep up.
- **Report spread, not just averages.** With 3 seeds, a gain smaller than about
  twice the seed std (rain, +1.3 pp) cannot be claimed.
- **A baseline can win in places.** HOG+SVM is worse overall but best in
  darkness. Without the baseline we would not know the CNN's weak spot.
- **Change one thing at a time.** One commit and one re-run per fix makes it
  clear what each change did.

## Workflow
- Run the quick smoke test before every full run. It caught both bugs in
  minutes instead of hours.
- Run `ruff check . && ruff format --check .` before committing, so CI stays green.
- Don't keep a file open in the editor while an agent edits it: the editor can
  save its old copy on top.
