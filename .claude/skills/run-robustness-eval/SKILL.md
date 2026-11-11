---
name: run-robustness-eval
description: End-to-end procedure to test a change, run the robustness experiment, review it and update results and notes. Use when Love says "kör hela flödet", "run the full eval", or after changing models, corruptions or config.
---

# Run the robustness evaluation end to end

This skill chains the project's agents. Run the steps in order and stop at
the first failure.

1. **Check the code.** Run `ruff check .` and `pytest -q`. Fix lint/test
   failures only if they are trivial; otherwise stop and report.
2. **Review.** Delegate to the `code-reviewer` agent for the current diff.
   If the verdict is "Ändringar krävs", stop and show Love the findings.
3. **Run.** Delegate to the `experiment-runner` agent: quick run, then full
   run.
4. **Analyse.** Delegate to the `results-analyst` agent. It writes
   `notes/analysis.md` and proposes README text.
5. **Report to Love** in Swedish, in at most five lines: what changed, the
   headline numbers, and the proposed README text. Ask before editing the
   README or committing.

Independent work can run in parallel (for example, the code review while the
quick run is going), but the full run must wait for a passing review.
