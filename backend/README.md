# Backend (FastAPI)

## Run (dev)

```bash
cd /Users/paulyang/Projects/DeepLearningLab
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

## Test

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/predict/line \
  -H 'Content-Type: application/json' \
  -d '{"w":2,"b":1,"x":3}'
```
