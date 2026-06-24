# Web Demo Deployment

The web version of the human playable Overcooked study lives at:
- Server: `src/multitask_personalization/web/server.py`
- Entry: `scripts/run_human_overcooked_web.py`
- Templates: `src/multitask_personalization/web/templates/`

## Run locally

```bash
PYTHONPATH=src .venv/bin/python scripts/run_human_overcooked_web.py
```

Then open http://localhost:8000 in your browser. Test it works end-to-end before exposing remotely via ngrok.

## Run on the web
### Setup

1. **Install ngrok**:
```bash
brew install ngrok
```

2. **Sign up at ngrok.com** (free tier is fine) and copy your auth token. Run:
```bash
ngrok config add-authtoken YOUR_TOKEN
```

3. **Start the server** in one terminal:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_human_overcooked_web.py
```

4. **Start ngrok** in another terminal:
```bash
ngrok http 8000
```
5. **Copy the public URL** ngrok prints and share to users

## Data location

**Local (ngrok)**: `logs/web/`
- `hbm_state/Raval-Trivedi.pkl` — your family's learned HBM
- `hbm_state/General_Population_Sample.pkl` — sampled group's HBM
- `players.json` — all players
- `snapshots/` — timestamped backups after every episode
