# Publish this starter to your GitHub account

The connected GitHub actions available to this session can manage files and branches in accessible repositories, but do not expose the API action required to create a new repository. Use this one-time step:

1. Sign in to GitHub as `3greenhornet`.
2. Open https://github.com/new
3. Repository name: `BuildSleuth`
4. Description: `Evidence-grounded AI assistant for diagnosing CI failures from build logs, code changes, and technical documentation.`
5. Choose **Public** if you want recruiters to view it. Do not initialize it with a README, license, or .gitignore (the starter already includes these).
6. Unzip this archive, open a terminal inside the `buildsleuth_starter` folder, then run:

```bash
git init
git add .
git commit -m "Initial BuildSleuth starter"
git branch -M main
git remote add origin https://github.com/3greenhornet/BuildSleuth.git
git push -u origin main
```

If Git asks you to authenticate, use GitHub's supported browser sign-in or credential manager; do not paste a personal access token into source files or chat.

Before presenting this as completed work on a resume, implement and validate the planned embedding/vector retrieval, evidence-cited model output, and actual GitHub Actions log integration. At present this repository is a starter baseline.
