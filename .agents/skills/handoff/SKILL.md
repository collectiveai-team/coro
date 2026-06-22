---
name: handoff
description: Compact the current conversation into a handoff document for another agent to pick up.
argument-hint: "What will the next session be used for?"
---

Write a handoff document summarising the current conversation so a fresh agent can continue the work. Save to the temporary directory on the workspace .tmp/handoff/ and name the file with the current timestamp and a slugified version of the topic (e.g. `2024-01-01T12-00-00-fix-login-bug.md`).

Gather all the relevant context in order make the next task as easy as possible for the next agent.

Include a "suggested skills" section in the document, which suggests skills that the agent should invoke.

Do not duplicate content already captured in other artifacts (PRDs, plans, ADRs, issues, commits, diffs). Reference them by path or URL instead.

Redact any sensitive information, such as API keys, passwords, or personally identifiable information.

As output write a small handoff prompt on this chat to be copied and pasted into the next session, which references the handoff document and highlights the most important context for the next agent to know.

If the user passed arguments, treat them as a description of what the next session will focus on and tailor the doc accordingly.
