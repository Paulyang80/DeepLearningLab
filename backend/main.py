from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List

app = FastAPI(title="DeepLearningLab API")

# Dev-only: allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
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


class Point(BaseModel):
    x: float
    y: float


class DatasetRequest(BaseModel):
    n: int = 40
    seed: int = 42
    x_min: float = -5
    x_max: float = 5
    true_w: float = 1.5
    true_b: float = -0.5
    noise_std: float = 0.8


class DatasetResponse(BaseModel):
    points: List[Point]


class MSERequest(BaseModel):
    w: float
    b: float
    points: List[Point]


class MSEResponse(BaseModel):
    mse: float


class TrainLinearStepRequest(BaseModel):
    w: float
    b: float
    learning_rate: float = 0.05
    points: List[Point]


class TrainLinearStepResponse(BaseModel):
    w: float
    b: float
    loss: float
    grad_w: float
    grad_b: float
    errors: List[float]


class ExplainLinearRequest(BaseModel):
    w: float
    b: float
    points: List[Point]


class ExplainLinearResponse(BaseModel):
    mse: float
    grad_w: float
    grad_b: float
    errors: List[float]


class SurfaceLinearRequest(BaseModel):
    points: List[Point]
    w_min: float = -5
    w_max: float = 5
    b_min: float = -5
    b_max: float = 5
    w_steps: int = 25
    b_steps: int = 25


class SurfaceLinearResponse(BaseModel):
    w_values: List[float]
    b_values: List[float]
    mse_grid: List[List[float]]


def _linear_forward(w: float, b: float, x: np.ndarray) -> np.ndarray:
    return w * x + b


def _mse_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
    err = y_hat - y
    return float(np.mean(err**2))


def _mse_gradients(x: np.ndarray, y_hat: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    err = y_hat - y
    grad_w = float(2.0 * np.mean(err * x))
    grad_b = float(2.0 * np.mean(err))
    return grad_w, grad_b, err


def _sgd_update(w: float, b: float, grad_w: float, grad_b: float, lr: float) -> tuple[float, float]:
    new_w = float(w - lr * grad_w)
    new_b = float(b - lr * grad_b)
    return new_w, new_b


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/predict/line", response_model=LineResponse)
def predict_line(req: LineRequest) -> LineResponse:
    return LineResponse(y=req.w * req.x + req.b)


@app.post("/dataset/linear", response_model=DatasetResponse)
def dataset_linear(req: DatasetRequest) -> DatasetResponse:
    n = int(max(1, min(req.n, 5000)))
    rng = np.random.default_rng(int(req.seed))

    x = rng.uniform(req.x_min, req.x_max, size=n)
    y = req.true_w * x + req.true_b + rng.normal(0.0, req.noise_std, size=n)

    points = [Point(x=float(xi), y=float(yi)) for xi, yi in zip(x, y)]
    return DatasetResponse(points=points)


@app.post("/metrics/mse", response_model=MSEResponse)
def metrics_mse(req: MSERequest) -> MSEResponse:
    if not req.points:
        return MSEResponse(mse=0.0)

    x = np.array([p.x for p in req.points], dtype=np.float64)
    y = np.array([p.y for p in req.points], dtype=np.float64)
    y_hat = req.w * x + req.b
    mse = float(np.mean((y_hat - y) ** 2))
    return MSEResponse(mse=mse)


@app.post("/explain/linear", response_model=ExplainLinearResponse)
def explain_linear(req: ExplainLinearRequest) -> ExplainLinearResponse:
    if not req.points:
        return ExplainLinearResponse(mse=0.0, grad_w=0.0, grad_b=0.0, errors=[])

    x = np.array([p.x for p in req.points], dtype=np.float64)
    y = np.array([p.y for p in req.points], dtype=np.float64)

    # forward
    y_hat = _linear_forward(req.w, req.b, x)

    # loss
    mse = _mse_loss(y_hat, y)

    # gradients
    grad_w, grad_b, err = _mse_gradients(x, y_hat, y)

    return ExplainLinearResponse(
        mse=mse,
        grad_w=grad_w,
        grad_b=grad_b,
        errors=[float(e) for e in err.tolist()],
    )


@app.post("/train/linear/step", response_model=TrainLinearStepResponse)
def train_linear_step(req: TrainLinearStepRequest) -> TrainLinearStepResponse:
    if not req.points:
        return TrainLinearStepResponse(
            w=req.w,
            b=req.b,
            loss=0.0,
            grad_w=0.0,
            grad_b=0.0,
            errors=[],
        )

    lr = float(req.learning_rate)
    if not np.isfinite(lr) or lr <= 0:
        lr = 0.05

    x = np.array([p.x for p in req.points], dtype=np.float64)
    y = np.array([p.y for p in req.points], dtype=np.float64)

    # forward
    y_hat = _linear_forward(req.w, req.b, x)

    # loss
    loss = _mse_loss(y_hat, y)

    # gradients
    grad_w, grad_b, err = _mse_gradients(x, y_hat, y)

    # update
    new_w, new_b = _sgd_update(req.w, req.b, grad_w, grad_b, lr)

    return TrainLinearStepResponse(
        w=new_w,
        b=new_b,
        loss=loss,
        grad_w=grad_w,
        grad_b=grad_b,
        errors=[float(e) for e in err.tolist()],
    )


@app.post("/surface/linear/mse", response_model=SurfaceLinearResponse)
def surface_linear_mse(req: SurfaceLinearRequest) -> SurfaceLinearResponse:
    if not req.points:
        return SurfaceLinearResponse(w_values=[], b_values=[], mse_grid=[])

    w_steps = int(max(2, min(req.w_steps, 80)))
    b_steps = int(max(2, min(req.b_steps, 80)))

    w_min = float(req.w_min)
    w_max = float(req.w_max)
    b_min = float(req.b_min)
    b_max = float(req.b_max)

    if not np.isfinite(w_min) or not np.isfinite(w_max) or w_min == w_max:
        w_min, w_max = -5.0, 5.0
    if not np.isfinite(b_min) or not np.isfinite(b_max) or b_min == b_max:
        b_min, b_max = -5.0, 5.0

    x = np.array([p.x for p in req.points], dtype=np.float64)
    y = np.array([p.y for p in req.points], dtype=np.float64)

    w_values = np.linspace(w_min, w_max, num=w_steps, dtype=np.float64)
    b_values = np.linspace(b_min, b_max, num=b_steps, dtype=np.float64)

    # Compute MSE grid; vectorize over b for each w.
    mse_grid: list[list[float]] = []
    for w in w_values:
        # y_hat shape: (b_steps, n)
        y_hat = w * x[None, :] + b_values[:, None]
        err = y_hat - y[None, :]
        mse_row = np.mean(err**2, axis=1)
        mse_grid.append([float(v) for v in mse_row.tolist()])

    return SurfaceLinearResponse(
        w_values=[float(v) for v in w_values.tolist()],
        b_values=[float(v) for v in b_values.tolist()],
        mse_grid=mse_grid,
    )
