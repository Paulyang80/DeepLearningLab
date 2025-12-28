# Backend (FastAPI)

## Run (dev)

```bash
cd /Users/paulyang/Projects/DeepLearningLab
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

If you run `uvicorn` from inside `web/`, Python may not find the top-level `backend` module. In that case either `cd` to the repo root (recommended) or run:

```bash
uvicorn backend.main:app --reload --port 8000 --app-dir /Users/paulyang/Projects/DeepLearningLab
```

## Test

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/predict/line \
  -H 'Content-Type: application/json' \
  -d '{"w":2,"b":1,"x":3}'
```
