# Theory Rating

Blind rating of 10 pre-ML main-effects theories (Race + Gender pool).

## Login fields (raters)

- **Seed** — which 10 theories (same seed → same set for all raters)
- **Identifier** — whose scores; same seed + identifier resumes progress
- Topic (Race / Gender) is shown; Human vs GenAI is **hidden** from raters

## How you (researcher) get the data

Open **`/admin`** on the deployed site (or locally `http://127.0.0.1:8765/admin`).

1. Enter the **admin token** (`ADMIN_TOKEN` env var; local default: `research-admin-change-me`)
2. Optionally filter by seed
3. **Download CSV**

CSV includes (among others):

| Column | Meaning |
|--------|---------|
| `rater_identifier` | Who scored |
| `seed` | Sample batch |
| `source` / `source_label` | human vs GenAI |
| `group` / `group_label` | PhD Student / Senior Scientist / GenAI |
| `participant_name` | Author name from the survey |
| `topic`, `task`, `text` | Theory content |
| five dimension scores | 1–10 |

APIs (same token):

- `GET /api/admin/export.csv?token=...&seed=optional`
- `GET /api/admin/export.json?token=...`
- `GET /api/admin/sessions?token=...`

## Local run

```bash
cd rating_system
export ADMIN_TOKEN='your-secret'
python3 -m uvicorn app:app --host 127.0.0.1 --port 8765
```

## Stable hosting (Railway)

```bash
cd rating_system
# install CLI once: brew install railway  OR  npm i -g @railway/cli
railway login
railway init
railway volume add --mount /data
railway variables set ADMIN_TOKEN='your-long-secret' RATING_DB_PATH=/data/ratings.db
railway up
railway domain
```

Share the Railway HTTPS URL with raters. Keep `ADMIN_TOKEN` private.

## Temporary tunnel (laptop must stay on)

```bash
cloudflared tunnel --url http://127.0.0.1:8765
```
