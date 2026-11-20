# Releases and deployment

Use a pull request for changes to `main`. The **Verify and deploy Prefero** workflow checks the public source tree, installs the runtime, runs regression and compatibility tests, and starts the Docker app. Check the **verify** job in Actions before merging. A push to `main` can deploy only after that job passes. Keep development on GitHub so that Hugging Face Space updates can remain fast-forward pushes.

## Deploy and verify

Configure the repository's `HF_TOKEN` secret with write access to the Prefero Space. A deployment saves the Space's previous commit on `releases/pre-<run ID>-<attempt>` before pushing the verified commit to Space `main`. If the Space already points at the verified commit, the job succeeds without rewriting it. If Space `main` has diverged, review its changes, reconcile the histories in a pull request and rerun verification.

After a deployment, check that the Space revision matches the GitHub Actions summary. Open the Space, generate a synthetic dataset and complete a conditional logit fit through Results. Tag the verified GitHub commit when publishing a version and attach release notes to the tag.

## Roll back

Use the previous Space revision from the deployment summary to identify the last known-good state. Revert the offending changes in a GitHub pull request, run verification, and let the resulting commit deploy to Space `main`. Keep the deployment history intact and verify the hosted workflow again after the revert.

## Review public files

Keep manuscripts, research data, local databases and credentials outside the public repository. The source tree check examines tracked filenames, so also inspect changes and fixtures before committing. If private content was committed, remove it from reachable history and review tags, branches, forks, pull requests, and deployment mirrors before treating the cleanup as complete. Revoke and replace any exposed credentials.
