# Deploying SkyTrust to Streamlit Community Cloud

Free hosting for Streamlit apps, straight from the GitHub repo. About 5 minutes.

## What's already set up in the repo

| Needed by Community Cloud | Where |
|---|---|
| Dependencies | `uv.lock` (repo root). Community Cloud checks for `uv.lock` first and installs from it |
| Entry point | `app/streamlit_app.py` |
| Theme / clean visitor view | `.streamlit/config.toml` (dark theme, developer toolbar hidden) |
| Models + metrics | `artifacts/*.json`: committed, so the app never downloads raw data or retrains |
| Our own package | installed by uv; if it isn't, the app adds `src/` to the import path itself |

Nothing needs secrets or API keys. Open-Meteo is free for non-commercial use.

## Steps

1. Make sure the latest code is on GitHub: in the project folder run `git push`.
2. Go to **https://share.streamlit.io** and click **Continue with GitHub**. Authorize Streamlit
   when GitHub asks.
3. Click **Create app** (top right), then **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `hitenjain22/skytrust`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** pick a subdomain, e.g. `skytrust` → `https://skytrust.streamlit.app`
     (if taken, try `skytrust-hiten`).
5. Open **Advanced settings** → **Python version: 3.12** (matches the project) → Save.
6. Click **Deploy**. The first build installs dependencies and takes a few minutes; you can watch
   the log on the right.
7. When it loads, check all four pages and try another site. Then open the same URL **on your
   phone** (this is the Phase 6 acceptance check).
8. Send the URL to Claude, and it will go into the README's "Live app" line.

## After deploying

- **Updates are automatic:** every `git push` to `main` redeploys the app.
- **Sleeping:** Community Cloud puts apps to sleep after a while with no visitors. The first
  visitor sees a "wake up" button and waits a minute or so. That's normal for the free tier.
- **Saved forecasts:** the "last good forecast" copy lives on the app's temporary disk, so it
  resets when the app restarts. If Open-Meteo is down right after a restart, the Tonight page shows a
  friendly "unavailable" message while Track Record and How It Works keep working.

## If something goes wrong

Open the app, click **Manage app** (bottom right), and read the log.

| Symptom | Fix |
|---|---|
| Error installing dependencies | Check that Python 3.12 is selected (Advanced settings; you can change it under app **Settings**), then **Reboot app**. |
| `ModuleNotFoundError: skytrust` | Shouldn't happen (the app falls back to `src/`). Confirm the main file path is exactly `app/streamlit_app.py`. |
| Tonight page says forecasts are unavailable | Open-Meteo didn't respond; reload in a minute. The other pages don't need it. |
| Anything else | Copy the last ~20 lines of the log and ask. |
