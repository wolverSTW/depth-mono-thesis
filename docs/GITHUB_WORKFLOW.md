# GitHub workflow

Remote: https://github.com/wolverSTW/depth-mono-thesis.git. Branch: develop.
Existing commits/history remain intact; do not reinitialize or force-push.

Before a milestone commit read status/diff and affected human-edited reports.
Stage only relevant source/docs/config/provenance. Datasets, weights, run outputs,
backups and credentials are ignored. Scan staged paths and content for sensitive
files; Git submodules must remain at recorded clean commits. Use meaningful
chore/feat/docs/fix messages after coherent work.

Inspect `git status --short --branch`, `git diff --cached --check` and staged diff.
Commit, report branch/hash, then ask the owner before `git push -u origin develop`.
No push is currently approved. Re-check `git ls-remote origin` before a later push.
If remote work exists, fetch/review and merge normally without rewriting history.
Do not discard manual edits when resolving conflicts. Code push does not back up
weights/logs/results: use the separately verified artifact workflow.
