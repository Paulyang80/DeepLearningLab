from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="DeepLearningLab API")

# Dev-only: allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LineRequest(BaseModel):
    w: float
    b: float
    x: float


class LineResponse(BaseModel):
    y: float


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/predict/line", response_model=LineResponse)
def predict_line(req: LineRequest) -> LineResponse:
    return LineResponse(y=req.w * req.x + req.b)
