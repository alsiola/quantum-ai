# How this project works its issues

This project lands work straight on `main`. There are no pull requests. These rules override the playbook's steps 4 and 5 and its rule against pushing to the base branch or closing the issue.

A `Closes #<number>` line in a commit pushed to `main` does **not** close the issue here. You must close each issue yourself once its work has landed (see Landing a change).

## Start

Once you have read the issue, post a short start comment with `github.issue.comment`, for example "Working on this. The change will land on `main` and close this issue." For an issue that asks for research, options or an answer, say findings will follow instead.

## Kinds of issue

- **Change:** something is to be built or fixed. Commit it to `main` as below.
- **Investigation:** the issue asks for research, options or an answer ("research X", "which approach…", "check whether…"). Post the findings on the issue and end with `review`; don't close it. If the findings lead to an obvious change the issue asked for, you may make it as a change as well.

## Landing a change

1. Commit on the issue's branch as you work, with clear messages.
2. When the work is done, run what the project's rules say must pass (typecheck, tests, build) and make it pass.
3. `git fetch origin` and rebase onto `origin/main`. Resolve any conflicts and run the checks again.
4. Post the report on the issue with `github.issue.comment` (see below).
5. Push to main: `git push origin HEAD:main`. If the push is refused because `main` moved, fetch, rebase, run the checks and push again. Never force-push `main`.
6. Close the issue yourself; a `Closes #<number>` line in the commit does nothing here. The toolbelt has no close operation, and Open Conduit's repository token can't edit issues, so use the owner's `gh` login, which the owner has approved for this one action:

   ```sh
   gh issue close <number> --repo <owner>/<repo> --reason completed
   ```

   Then check with `github.issue.view` that it shows `closed`. Use `gh` for closing only; comments still go through the toolbelt. If the close fails, say so on the issue and end with `help`. When an issue takes several pushes, close it only after the last one.
7. End with `finished` once the issue has closed. If you pushed with more still to do, leave the issue open and end with `done` or `help` as fits.

## The report

Keep it brief; the commits have the detail:

- a mention of the issue's author and the headline of what now exists, in one line
- `- Commit: <short sha>` (or the range, if there are several commits)
- a short paragraph on what was done
- anything worth flagging (what you couldn't do, a judgement call they might want to reverse, a step only a person can take), each with a bold lead

## Feedback after closing

If the owner comments or reopens the issue after it closed, treat it as new work: make the change, land it the same way (closing the issue again yourself if it was reopened), and reply on the issue with the commit.

## Other agents

When another issue overlaps yours (the same file, module or pipeline step), read it and its recent commits on `main` before you build, and rebase onto its work rather than making the same change twice.

Comment on another agent's issue rarely: only when your work turns up something that helps its work in progress, or needs it to take a step it wouldn't otherwise see. You don't need to answer comments from other agents; act on what they tell you, and reply only when you disagree outright on the way forward.
