from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Import CNN module
from backend.cnn import SimpleCNN, CNNPredictor

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


class LogisticDatasetRequest(BaseModel):
    n: int = 80
    seed: int = 7
    x_min: float = -5
    x_max: float = 5
    true_w: float = 1.2
    true_b: float = -0.3


class LogisticDatasetResponse(BaseModel):
    points: List[Point]


class ExplainLogisticRequest(BaseModel):
    w: float
    b: float
    activation: str = "sigmoid"  # sigmoid | tanh | linear
    points: List[Point]


class ExplainLogisticResponse(BaseModel):
    loss: float
    grad_w: float
    grad_b: float
    probs: List[float]
    errors: List[float]


class TrainLogisticStepRequest(BaseModel):
    w: float
    b: float
    activation: str = "sigmoid"
    learning_rate: float = 0.2
    points: List[Point]


class TrainLogisticStepResponse(BaseModel):
    w: float
    b: float
    loss: float
    grad_w: float
    grad_b: float
    probs: List[float]
    errors: List[float]


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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _logistic_probs_and_dpdz(z: np.ndarray, activation: str) -> tuple[np.ndarray, np.ndarray]:
    act = (activation or "sigmoid").strip().lower()

    if act == "tanh":
        t = np.tanh(z)
        p = (t + 1.0) / 2.0
        dpdz = 0.5 * (1.0 - t**2)
        return p, dpdz

    if act == "linear":
        eps = 1e-6
        p = np.clip(z, eps, 1.0 - eps)
        dpdz = ((z > eps) & (z < (1.0 - eps))).astype(np.float64)
        return p, dpdz

    # default: sigmoid
    p = _sigmoid(z)
    dpdz = p * (1.0 - p)
    return p, dpdz


def _bce_loss_and_gradients(
    x: np.ndarray,
    y: np.ndarray,
    w: float,
    b: float,
    activation: str,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    z = _linear_forward(w, b, x)
    p, dpdz = _logistic_probs_and_dpdz(z, activation)

    eps = 1e-9
    p_clip = np.clip(p, eps, 1.0 - eps)
    loss = float(np.mean(-(y * np.log(p_clip) + (1.0 - y) * np.log(1.0 - p_clip))))

    # dL/dp for BCE
    dldp = (p_clip - y) / (p_clip * (1.0 - p_clip) + eps)
    delta = dldp * dpdz  # dL/dz

    grad_w = float(np.mean(delta * x))
    grad_b = float(np.mean(delta))
    errors = p - y
    return loss, grad_w, grad_b, p, errors


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


@app.post("/dataset/logistic_1d", response_model=LogisticDatasetResponse)
def dataset_logistic_1d(req: LogisticDatasetRequest) -> LogisticDatasetResponse:
    n = int(max(1, min(req.n, 5000)))
    rng = np.random.default_rng(int(req.seed))

    x = rng.uniform(req.x_min, req.x_max, size=n)
    z = req.true_w * x + req.true_b
    p = _sigmoid(z)
    y = (rng.uniform(0.0, 1.0, size=n) < p).astype(np.float64)

    points = [Point(x=float(xi), y=float(yi)) for xi, yi in zip(x, y)]
    return LogisticDatasetResponse(points=points)


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


@app.post("/explain/logistic_1d", response_model=ExplainLogisticResponse)
def explain_logistic_1d(req: ExplainLogisticRequest) -> ExplainLogisticResponse:
    if not req.points:
        return ExplainLogisticResponse(loss=0.0, grad_w=0.0, grad_b=0.0, probs=[], errors=[])

    x = np.array([p.x for p in req.points], dtype=np.float64)
    y = np.array([p.y for p in req.points], dtype=np.float64)

    loss, grad_w, grad_b, p, err = _bce_loss_and_gradients(x, y, req.w, req.b, req.activation)
    return ExplainLogisticResponse(
        loss=loss,
        grad_w=grad_w,
        grad_b=grad_b,
        probs=[float(v) for v in p.tolist()],
        errors=[float(v) for v in err.tolist()],
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


@app.post("/train/logistic_1d/step", response_model=TrainLogisticStepResponse)
def train_logistic_1d_step(req: TrainLogisticStepRequest) -> TrainLogisticStepResponse:
    if not req.points:
        return TrainLogisticStepResponse(
            w=req.w,
            b=req.b,
            loss=0.0,
            grad_w=0.0,
            grad_b=0.0,
            probs=[],
            errors=[],
        )

    lr = float(req.learning_rate)
    if not np.isfinite(lr) or lr <= 0:
        lr = 0.2

    x = np.array([p.x for p in req.points], dtype=np.float64)
    y = np.array([p.y for p in req.points], dtype=np.float64)

    loss, grad_w, grad_b, p, err = _bce_loss_and_gradients(x, y, req.w, req.b, req.activation)
    new_w, new_b = _sgd_update(req.w, req.b, grad_w, grad_b, lr)

    return TrainLogisticStepResponse(
        w=new_w,
        b=new_b,
        loss=loss,
        grad_w=grad_w,
        grad_b=grad_b,
        probs=[float(v) for v in p.tolist()],
        errors=[float(v) for v in err.tolist()],
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


# ============================================================================
# CNN / MNIST Section
# ============================================================================


class MNISTPredictRequest(BaseModel):
    image: str  # base64-encoded PNG/JPEG or data URL
    return_feature_maps: bool = False


class MNISTPredictResponse(BaseModel):
    predicted_digit: int
    probabilities: List[float]
    logits: List[float]
    conv1_maps: Optional[List[List[List[float]]]] = None  # (16, H, W)
    conv2_maps: Optional[List[List[List[float]]]] = None  # (32, H, W)


# Initialize CNN predictor
MODEL_DIR = Path(__file__).parent / "models"
cnn_predictor = CNNPredictor(MODEL_DIR)


@app.post("/predict/mnist", response_model=MNISTPredictResponse)
def predict_mnist(req: MNISTPredictRequest) -> MNISTPredictResponse:
    """
    Predict MNIST digit from base64 image.
    Optionally return conv1 and conv2 feature maps for visualization.
    """
    try:
        result = cnn_predictor.predict(
            image_str=req.image,
            return_feature_maps=req.return_feature_maps
        )
        return MNISTPredictResponse(**result)
    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")


@app.get("/model/cnn/info")
def cnn_model_info() -> dict:
    """Return basic info about the CNN model."""
    return cnn_predictor.get_model_info()
