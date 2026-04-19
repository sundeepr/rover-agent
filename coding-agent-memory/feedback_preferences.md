---
name: Feedback and preferences
description: How to collaborate on this project — coding style, communication, workflow
type: feedback
originSessionId: 74eb3b7b-c6c1-4d44-b30a-0225d3b8f403
---
Keep responses short and direct — no summaries after making changes.

**Why:** Sundeep reads the diff himself and doesn't need a recap.

**How to apply:** After edits, just say what was done in one line then ask "Commit and push?" or wait for next instruction. Don't list every file changed.

---

Don't add comments, docstrings, or type annotations to code that wasn't changed.

**Why:** Keeps diffs clean and focused.

**How to apply:** Only annotate new code you write, leave surrounding code as-is.

---

Ask before making large refactors or changes beyond what was requested.

**Why:** Sundeep tests on real hardware — unexpected changes can break things mid-session.

**How to apply:** Implement exactly what was asked. If you spot a related bug, mention it separately rather than fixing it silently.
