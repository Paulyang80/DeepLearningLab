# DeepLearningLab

Interactive deep learning (DL) visualizer playground.

This repo contains:

- `web/`: Next.js (TypeScript + Tailwind) frontend for interactive visualizations
- `backend/`: FastAPI backend for Python-side computation (later: PyTorch inference/training)

## Run (dev)

### Backend (FastAPI)

```bash
cd /Users/paulyang/Projects/DeepLearningLab
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### Frontend (Next.js)

```bash
cd /Users/paulyang/Projects/DeepLearningLab/web
npm install
npm run dev
```
