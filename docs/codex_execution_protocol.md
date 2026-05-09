# Codex Execution Protocol

This document is a historical execution note from the earlier Storyflow naming
period. For Gate R0 and later work, the active repository is TRUCE-Rec unless
the user states otherwise, the active package is `src/llm4rec/`, and the active
GitHub target is `https://github.com/appleweiping/TRUCE-Rec.git`.

## Preflight

Before editing, Codex must:

1. Confirm the current directory is `D:\Research\TRUCE-Rec`.
2. Confirm the current branch is `main`.
3. Confirm the active GitHub target is
   `https://github.com/appleweiping/TRUCE-Rec.git`. If the local checkout still
   has a historical remote alias, treat it as an alias rather than the project
   identity.
4. Run `git status`.
5. Inspect and report unexpected uncommitted changes before editing.
6. Read `AGENTS.md`, `docs/PROJECT_MEMORY.md`, and `docs/RESEARCH_IDEA.md`
   enough to keep the task aligned.

If the branch is not `main`, Codex must stop and report in Chinese. Codex must
not switch to archive branches and must not read or use
`D:\Research\Uncertainty-LLM4Rec` unless the user explicitly requests a narrow
comparison.

## Scope Control

Codex must implement only the current milestone. For Phase 0, that means
governance and protocol documents only.

Codex must not:

- fabricate experimental results;
- download datasets unless the current task asks for it;
- call paid APIs unless explicitly approved and configured;
- commit `.env`, API keys, sensitive raw responses, raw datasets, large PDFs,
  or reference zip files;
- claim server work was run without user-provided logs or artifacts;
- implement toy models when the task is governance-only.

## Implementation Steps

For each substantial task:

1. Make the requested, tightly scoped edits.
2. Update `README.md` when workflows, commands, status, modules, dependencies,
   or server instructions change.
3. Update relevant docs when assumptions, protocols, or architecture change.
4. Update `docs/PROJECT_MEMORY.md` when big direction, baseline fairness,
   server commands, current status, or future-agent workflow changes.
5. Run relevant tests or basic checks.
6. Write a detailed Chinese local report under
   `local_reports/YYYYMMDD-HHMMSS-task-name.md`.
7. Ensure `local_reports/` is ignored by git.
8. Run `git status`.
9. Commit with a meaningful message.
10. Push to `origin/main`.

## Testing Policy

Use pytest once executable code exists. Tests should cover schemas, data
loading, k-core filtering, splitting, title normalization, grounding,
correctness labels, calibration metrics, popularity buckets, provider parsing
without paid API calls, cache/resume logic, synthetic observation pipeline,
scoring/reranking, triage behavior, and simulation determinism.

For documentation-only Phase 0 tasks, basic checks may include:

```powershell
Get-Location
git branch --show-current
git remote -v
git status --short --branch
```

If a requested test cannot run, Codex must state exactly why and what was run
instead.

## Local Report Requirements

Every substantial task must create a Chinese local report under
`local_reports/`. The report must not be committed.

The report must include:

1. This round's goal.
2. What was completed.
3. Modified file list.
4. New or changed commands.
5. Data download and processing status.
6. API call and cache status.
7. Whether the experiment state is synthetic, pilot, full, or not run.
8. Locally runnable content.
9. Server-only content.
10. Test commands and results.
11. Git commit hash.
12. Whether pushed to `origin/main`.
13. Relationship to `Storyflow.md`.
14. Current risks.
15. Recommended next step.
16. User actions needed.

## Final Response Requirements

At the end of each Codex task, respond in Chinese with:

- what was completed;
- files changed;
- commands run;
- test results;
- whether README was updated;
- whether a local report was written;
- git commit hash;
- push status;
- next recommendation;
- user action needed.

Failure cases must be concrete and include the command or step that failed.
