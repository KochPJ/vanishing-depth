from __future__ import annotations

import base64
import io
import math
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from starlette.requests import Request

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import datasets.transforms as vd_transforms
from model.vanishing_depth import build_model

try:
    from utils.perlin_noise import random_binary_perlin_noise

    HAS_PROJECT_PERLIN = True
except Exception:
    random_binary_perlin_noise = None
    HAS_PROJECT_PERLIN = False


# ---------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------

DEMO_DIR = ROOT / "data" / "demo"
UPLOAD_DIR = ROOT / ".web_uploads"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

CHECKPOINT_PATH = (
    ROOT
    / "data"
    / "trained_models"
    / "20251026_1761460596_Dino_HalfPerlin448"
    / "best_auc_checkpoint.ckpt"
)

MAX_UPLOAD_BYTES = 80 * 1024 * 1024
UPLOAD_TTL_SECONDS = 2 * 60 * 60

MAX_STAGE_COUNT = 4
STAGE_TTL_SECONDS = 60 * 60

PREVIEW_MAX_SIDE = 1000

# Processing size slider settings.
PROCESS_MIN_SIDE_MIN = 224
PROCESS_MIN_SIDE_MAX = 896

RASTER_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

DEPTH_EXTRA_EXTENSIONS = {
    ".npy",
    ".npz",
}

ALL_DEPTH_EXTENSIONS = RASTER_EXTENSIONS | DEPTH_EXTRA_EXTENSIONS

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required. This application only supports cuda:0."
    )

torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(title="Vanishing Depth Analyzer")

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR)
)

MODEL: Optional[torch.nn.Module] = None
MODEL_ARGS: Optional[SimpleNamespace] = None
DPE = None

MODEL_LOCK = threading.Lock()
UPLOAD_LOCK = threading.Lock()
STAGE_LOCK = threading.Lock()


# ---------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------

class SourceRef(BaseModel):
    kind: Literal["demo", "upload"]
    id: str


class DrawStroke(BaseModel):
    """
    Browser stores coordinates normalized into [0, 1].

    This ensures that a line drawn on a downscaled preview still maps correctly
    to native RGB/depth resolution on the backend.
    """

    points: List[Tuple[float, float]]
    depth_m: float = 1.0
    radius_norm: float = 0.01


class GenerateMissingDepthRequest(BaseModel):
    rgb: SourceRef
    depth: Optional[SourceRef] = None

    depth_scalar: float = 1.0

    clear_depth: bool = False
    strokes: List[DrawStroke] = Field(default_factory=list)

    use_perlin: bool = True
    use_pixel_noise: bool = False
    use_patch_cuts: bool = False

    threshold: float = 0.50
    patch_min: int = 32
    patch_max: int = 128

    random_seed: Optional[int] = None


class ProcessStageRequest(BaseModel):
    stage_id: str
    process_min_size: int


# ---------------------------------------------------------------------
# Runtime cache data structures
# ---------------------------------------------------------------------

@dataclass
class UploadRecord:
    path: Path
    purpose: str
    label: str
    created_at: float


@dataclass
class GeneratedDepthStage:
    created_at: float

    # Native RGB resolution.
    rgb: np.ndarray

    # Loaded original/reference depth, in meters.
    reference_depth_m: np.ndarray

    # Reference depth after clear action plus manual depth drawing.
    editable_depth_m: np.ndarray

    # Final frozen sparse depth after Perlin/pixel/patch removal.
    generated_depth_m: np.ndarray

    # Mask caused by random corruption only.
    corruption_mask: np.ndarray

    # False for RGB-only/manual-only use.
    has_reference_depth: bool


UPLOADS: Dict[str, UploadRecord] = {}
STAGES: Dict[str, GeneratedDepthStage] = {}


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def source_to_dict(source: SourceRef) -> dict:
    return {
        "kind": source.kind,
        "id": source.id,
    }


def safe_float(value, default=None):
    try:
        value = float(value)

        if math.isfinite(value):
            return value
    except Exception:
        pass

    return default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def safe_demo_path(relative_path: str) -> Path:
    if not relative_path:
        raise HTTPException(
            status_code=400,
            detail="Missing demo file path.",
        )

    path = (DEMO_DIR / relative_path).resolve()

    if not is_inside(path, DEMO_DIR):
        raise HTTPException(
            status_code=400,
            detail="Invalid demo path.",
        )

    if not path.exists() or not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Demo file not found: {relative_path}",
        )

    return path


def scan_demo_files() -> dict:
    rgb_files = []
    depth_files = []

    if not DEMO_DIR.exists():
        return {
            "rgb_files": [],
            "depth_files": [],
        }

    for path in sorted(DEMO_DIR.rglob("*")):
        if not path.is_file():
            continue

        extension = path.suffix.lower()
        relative_path = path.relative_to(DEMO_DIR).as_posix()

        if extension in RASTER_EXTENSIONS:
            rgb_files.append(relative_path)

        if extension in ALL_DEPTH_EXTENSIONS:
            depth_files.append(relative_path)

    return {
        "rgb_files": rgb_files,
        "depth_files": depth_files,
    }


# ---------------------------------------------------------------------
# Upload and temporary stage cleanup
# ---------------------------------------------------------------------

def cleanup_uploads():
    now = time.time()
    expired = []

    with UPLOAD_LOCK:
        for upload_id, record in UPLOADS.items():
            if now - record.created_at > UPLOAD_TTL_SECONDS:
                expired.append((upload_id, record))

        for upload_id, record in expired:
            UPLOADS.pop(upload_id, None)

            try:
                record.path.unlink(missing_ok=True)
            except Exception:
                pass


def cleanup_stages():
    now = time.time()

    with STAGE_LOCK:
        expired_ids = [
            stage_id
            for stage_id, stage in STAGES.items()
            if now - stage.created_at > STAGE_TTL_SECONDS
        ]

        for stage_id in expired_ids:
            STAGES.pop(stage_id, None)

        if len(STAGES) > MAX_STAGE_COUNT:
            ordered = sorted(
                STAGES.items(),
                key=lambda item: item[1].created_at,
            )

            for stage_id, _ in ordered[:-MAX_STAGE_COUNT]:
                STAGES.pop(stage_id, None)


def store_stage(stage: GeneratedDepthStage) -> str:
    cleanup_stages()

    stage_id = uuid.uuid4().hex

    with STAGE_LOCK:
        STAGES[stage_id] = stage

    return stage_id


def get_stage(stage_id: str) -> GeneratedDepthStage:
    cleanup_stages()

    with STAGE_LOCK:
        stage = STAGES.get(stage_id)

    if stage is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Generated sparse-depth stage is unavailable or expired. "
                "Please generate missing depth again."
            ),
        )

    return stage


def resolve_source(
    source: SourceRef,
    expected_purpose: str,
) -> Tuple[Path, str]:
    cleanup_uploads()

    if source.kind == "demo":
        path = safe_demo_path(source.id)
        return path, source.id

    with UPLOAD_LOCK:
        record = UPLOADS.get(source.id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Uploaded file is no longer available. "
                "Please upload it again."
            ),
        )

    if record.purpose != expected_purpose:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Uploaded file is not a valid {expected_purpose} source."
            ),
        )

    return record.path, record.label


# ---------------------------------------------------------------------
# RGB and depth loading
# ---------------------------------------------------------------------

def load_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            rgb = np.asarray(image)
    except Exception as exc:
        raise ValueError(
            f"Could not read RGB image '{path.name}': {exc}"
        ) from exc

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(
            f"RGB image '{path.name}' does not have shape H×W×3."
        )

    return np.ascontiguousarray(rgb.astype(np.uint8))


def load_depth(path: Path) -> np.ndarray:
    extension = path.suffix.lower()

    try:
        if extension == ".npy":
            depth = np.load(path)

        elif extension == ".npz":
            archive = np.load(path)

            try:
                if not archive.files:
                    raise ValueError("NPZ file contains no arrays.")

                depth = archive[archive.files[0]]
            finally:
                archive.close()

        else:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                depth = np.asarray(image)

    except Exception as exc:
        raise ValueError(
            f"Could not read depth file '{path.name}': {exc}"
        ) from exc

    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[..., 0]

        elif depth.shape[-1] >= 3:
            depth_float = depth[..., :3].astype(np.float32)

            # Preserve depth values if RGB channels are effectively identical.
            if (
                np.allclose(depth_float[..., 0], depth_float[..., 1])
                and np.allclose(depth_float[..., 0], depth_float[..., 2])
            ):
                depth = depth_float[..., 0]

            else:
                # Fallback for accidentally loading a color image as depth.
                depth = (
                    0.299 * depth_float[..., 0]
                    + 0.587 * depth_float[..., 1]
                    + 0.114 * depth_float[..., 2]
                )

        else:
            raise ValueError(
                f"Unsupported depth shape: {depth.shape}"
            )

    if depth.ndim != 2:
        raise ValueError(
            f"Depth must be two-dimensional. Got shape {depth.shape}."
        )

    depth = np.asarray(depth, dtype=np.float32)

    depth = np.nan_to_num(
        depth,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    depth[depth < 0] = 0.0

    return np.ascontiguousarray(depth)


def depth_statistics(depth: np.ndarray) -> dict:
    valid = np.isfinite(depth) & (depth > 0)

    total_count = int(depth.size)
    valid_count = int(valid.sum())

    if valid_count == 0:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "zero_percent": 100.0,
            "valid_pixels": 0,
            "total_pixels": total_count,
        }

    values = depth[valid].astype(np.float64)

    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
        "zero_percent": float(
            100.0 * (1.0 - valid_count / max(total_count, 1))
        ),
        "valid_pixels": valid_count,
        "total_pixels": total_count,
    }


def suggested_depth_scalar(depth: np.ndarray) -> float:
    valid = depth[depth > 0]

    if valid.size == 0:
        return 1.0

    return 0.001 if float(valid.max()) > 100.0 else 1.0


# ---------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------

def resize_for_preview(
    image: np.ndarray,
    max_side: Optional[int],
) -> np.ndarray:
    if max_side is None:
        return image

    height, width = image.shape[:2]
    current_max = max(height, width)

    if current_max <= max_side:
        return image

    scale = max_side / float(current_max)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def png_data_url(
    rgb_image: np.ndarray,
    max_side: Optional[int] = PREVIEW_MAX_SIDE,
) -> str:
    rgb_image = resize_for_preview(rgb_image, max_side)

    if rgb_image.ndim == 2:
        rgb_image = np.repeat(
            rgb_image[..., None],
            3,
            axis=-1,
        )

    rgb_image = np.clip(
        rgb_image,
        0,
        255,
    ).astype(np.uint8)

    buffer = io.BytesIO()

    Image.fromarray(
        np.ascontiguousarray(rgb_image),
        mode="RGB",
    ).save(
        buffer,
        format="PNG",
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"


def robust_range(
    depth: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[float, float]:
    values = depth[valid_mask]

    if values.size == 0:
        return 0.0, 1.0

    values = values.astype(np.float64)

    low = float(np.percentile(values, 2.0))
    high = float(np.percentile(values, 98.0))

    if not math.isfinite(low):
        low = 0.0

    if not math.isfinite(high) or high <= low:
        high = low + 1e-6

    return low, high


def magma_depth_image(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    normalized = (
        depth.astype(np.float32) - low
    ) / max(high - low, 1e-8)

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    color_bgr = cv2.applyColorMap(
        (normalized * 255.0).astype(np.uint8),
        cv2.COLORMAP_MAGMA,
    )

    color_rgb = cv2.cvtColor(
        color_bgr,
        cv2.COLOR_BGR2RGB,
    )

    color_rgb[~valid_mask] = np.array(
        [15, 18, 24],
        dtype=np.uint8,
    )

    return color_rgb


def signed_feature_image(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    normalized = np.clip(
        (values + 1.0) * 0.5,
        0.0,
        1.0,
    )

    colormap = getattr(
        cv2,
        "COLORMAP_TWILIGHT",
        cv2.COLORMAP_TURBO,
    )

    color_bgr = cv2.applyColorMap(
        (normalized * 255.0).astype(np.uint8),
        colormap,
    )

    return cv2.cvtColor(
        color_bgr,
        cv2.COLOR_BGR2RGB,
    )


def common_depth_display_range(
    *depth_maps: np.ndarray,
) -> Tuple[float, float]:
    values = []

    for depth in depth_maps:
        valid = depth[depth > 0]

        if valid.size > 0:
            values.append(valid)

    if not values:
        return 0.0, 1.0

    values = np.concatenate(values)

    low = float(np.percentile(values, 2.0))
    high = float(np.percentile(values, 98.0))

    return low, max(high, low + 1e-6)


# ---------------------------------------------------------------------
# Demo/upload preview helpers
# ---------------------------------------------------------------------

def source_payload(
    source: SourceRef,
    purpose: str,
) -> dict:
    path, label = resolve_source(source, purpose)

    if purpose == "rgb":
        rgb = load_rgb(path)

        return {
            "source": source_to_dict(source),
            "label": label,
            "shape": {
                "height": int(rgb.shape[0]),
                "width": int(rgb.shape[1]),
            },
            "preview_data_url": png_data_url(rgb),
        }

    depth = load_depth(path)

    valid = depth > 0
    low, high = robust_range(depth, valid)

    preview = magma_depth_image(
        depth,
        valid,
        low,
        high,
    )

    return {
        "source": source_to_dict(source),
        "label": label,
        "shape": {
            "height": int(depth.shape[0]),
            "width": int(depth.shape[1]),
        },
        "preview_data_url": png_data_url(preview),
        "raw_stats": depth_statistics(depth),
        "auto_scalar": suggested_depth_scalar(depth),
    }


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def checkpoint_args_to_namespace(
    checkpoint: dict,
) -> SimpleNamespace:
    raw_args = checkpoint.get("logs", {}).get("args", None)

    if raw_args is None:
        raw_args = checkpoint.get("args", None)

    if raw_args is None:
        raise RuntimeError(
            "Checkpoint has no arguments under logs.args or args."
        )

    if isinstance(raw_args, SimpleNamespace):
        raw_args = vars(raw_args)

    if not isinstance(raw_args, dict):
        raw_args = vars(raw_args)

    args = SimpleNamespace(**raw_args)

    defaults = {
        "with_intr": False,
        "with_depth": True,
        "with_depth_scales": False,
        "with_dino_head": False,
        "return_rdps": False,
        "decode_floor": 1e-4,
        "decode_floor_exp": 3.0,
        "decode_dist_scale_inv_log10": False,
        "add_depth_scales_to_cls_token": False,
        "freeze_color_encoder": False,
        "pretrained_imagenet": False,
        "pretrained_path": "",
        "train_size": 448,
        "max_depth": 15.0,
        "position_offset": 0.0,
        "zero_eps": 1e-6,
        "with_positional_encoding": True,
        "encoder_hidden_dims": 768,
    }

    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Prevent accidental reloading/downloading external backbone weights.
    # The complete model weights should come from the selected checkpoint.
    args.pretrained_path = ""
    args.pretrained_imagenet = False

    args.device = "cuda:0"
    args.multi_gpu = False
    args.gpu_ids = "0"

    args.temperature = float(args.temperature)
    args.scale = float(args.scale)
    args.max_depth = float(args.max_depth)

    return args


def normalize_state_dict(state_dict: dict) -> dict:
    if not state_dict:
        raise RuntimeError("Checkpoint state_dict is empty.")

    keys = list(state_dict.keys())

    if keys and all(key.startswith("module.") for key in keys):
        return {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }

    return state_dict


def load_model():
    global MODEL
    global MODEL_ARGS
    global DPE

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {CHECKPOINT_PATH}"
        )

    print(f"[webapp] Loading checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    MODEL_ARGS = checkpoint_args_to_namespace(checkpoint)

    if getattr(MODEL_ARGS, "with_intr", False):
        raise RuntimeError(
            "This application supports scalar depth only. "
            "The selected checkpoint has with_intr=True."
        )

    if not getattr(MODEL_ARGS, "with_depth", True):
        raise RuntimeError(
            "The selected checkpoint does not use depth input."
        )

    if not getattr(MODEL_ARGS, "with_positional_encoding", False):
        raise RuntimeError(
            "This application expects a checkpoint using "
            "DepthPositionalEncoding."
        )

    model = build_model(MODEL_ARGS)

    state_dict = checkpoint.get(
        "state_dict",
        checkpoint.get("model", None),
    )

    if state_dict is None:
        raise RuntimeError(
            "Checkpoint has neither state_dict nor model keys."
        )

    state_dict = normalize_state_dict(state_dict)

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )

        print("[webapp] Checkpoint loaded strictly.")

    except RuntimeError as strict_error:
        print("[webapp] Strict checkpoint loading failed.")
        print(strict_error)

        model_state = model.state_dict()

        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state
            and model_state[key].shape == value.shape
        }

        model.load_state_dict(
            compatible,
            strict=False,
        )

        match_ratio = len(compatible) / max(len(model_state), 1)

        print(
            "[webapp] Compatible tensors: "
            f"{len(compatible)}/{len(model_state)} "
            f"({match_ratio * 100.0:.2f}%)."
        )

        if match_ratio < 0.90:
            raise RuntimeError(
                "Too few checkpoint tensors matched this code revision. "
                "Use the exact repository revision that trained the checkpoint."
            )

    model = model.to(DEVICE)
    model.eval()

    DPE = vd_transforms.DepthPositionalEncoding(
        depth_channels=int(MODEL_ARGS.depth_channels),
        temperature=float(MODEL_ARGS.temperature),
        position_offset=float(
            getattr(MODEL_ARGS, "position_offset", 0.0)
        ),
        scale=float(MODEL_ARGS.scale),
        zero_eps=float(
            getattr(MODEL_ARGS, "zero_eps", 1e-6)
        ),
        flatten_4th_dim=False,
        div_factors=(
            MODEL_ARGS.pe_div_factors
            if getattr(MODEL_ARGS, "set_pe_div_factors", False)
            else None
        ),
        with_dino_head=False,
        encoder_hidden_dims=int(
            getattr(MODEL_ARGS, "encoder_hidden_dims", 768)
        ),
        enable_dino_head_epoch=0,
        get_depth_scales=bool(
            getattr(MODEL_ARGS, "with_depth_scales", False)
        ),
        max_depth=float(MODEL_ARGS.max_depth),
    )

    MODEL = model

    print("[webapp] Model loaded on cuda:0.")


@app.on_event("startup")
def startup_event():
    load_model()


# ---------------------------------------------------------------------
# Model input processing helpers
# ---------------------------------------------------------------------

def model_patch_size() -> int:
    assert MODEL is not None

    encoder = getattr(MODEL, "encoder", None)
    patch_size = getattr(encoder, "patch_size", 14)

    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]

    return int(patch_size)


def round_to_patch(
    value: int,
    patch_size: int,
) -> int:
    return int(round(value / patch_size) * patch_size)


def ceil_to_patch(
    value: int,
    patch_size: int,
) -> int:
    return int(math.ceil(value / patch_size) * patch_size)


def get_processing_shape(
    source_height: int,
    source_width: int,
    process_min_size: int,
) -> Tuple[int, int]:
    patch_size = model_patch_size()

    if process_min_size < PROCESS_MIN_SIDE_MIN:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Processing size must be at least "
                f"{PROCESS_MIN_SIDE_MIN}."
            ),
        )

    if process_min_size > PROCESS_MIN_SIDE_MAX:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Processing size must be at most "
                f"{PROCESS_MIN_SIDE_MAX}."
            ),
        )

    if process_min_size % patch_size != 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Processing size must be divisible by "
                f"patch size {patch_size}."
            ),
        )

    if source_height <= source_width:
        process_height = process_min_size

        process_width = ceil_to_patch(
            int(
                round(
                    source_width
                    * process_min_size
                    / source_height
                )
            ),
            patch_size,
        )

    else:
        process_width = process_min_size

        process_height = ceil_to_patch(
            int(
                round(
                    source_height
                    * process_min_size
                    / source_width
                )
            ),
            patch_size,
        )

    return process_height, process_width


def normalize_rgb_for_model(
    rgb: np.ndarray,
) -> torch.Tensor:
    tensor = torch.from_numpy(
        np.ascontiguousarray(rgb)
    ).float()

    tensor = tensor.permute(2, 0, 1) / 255.0

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32,
    )[:, None, None]

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32,
    )[:, None, None]

    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0).to(
        DEVICE,
        non_blocking=True,
    )


def encode_depth_for_model(
    depth_m: np.ndarray,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Uses the exact project DepthPositionalEncoding implementation.
    """

    depth_tensor = torch.from_numpy(
        np.ascontiguousarray(depth_m)
    ).float()

    sample = {
        "depth": depth_tensor.clone(),
    }

    sample = DPE(sample)

    encoded_depth = sample["depth"].unsqueeze(0).to(
        DEVICE,
        non_blocking=True,
    )

    depth_scales = sample.get("depth_scales", None)

    if depth_scales is not None:
        depth_scales = depth_scales.unsqueeze(0).to(
            DEVICE,
            non_blocking=True,
        )

    return encoded_depth, depth_scales


def extract_final_depth_prediction(
    output,
) -> torch.Tensor:
    """
    Handles:
      - Tensor
      - List[Tensor] for FPN
      - {"maps": Tensor}
      - {"maps": List[Tensor]}
    """

    if isinstance(output, dict):
        output = output.get("maps", output)

    if isinstance(output, (list, tuple)):
        tensors = [
            item
            for item in output
            if torch.is_tensor(item)
        ]

        if not tensors:
            raise RuntimeError(
                "No tensor output found in model prediction."
            )

        output = tensors[-1]

    if not torch.is_tensor(output):
        raise RuntimeError(
            f"Unexpected model output type: {type(output)}"
        )

    if output.ndim == 3:
        output = output.unsqueeze(1)

    if output.ndim != 4:
        raise RuntimeError(
            f"Expected [B,C,H,W] model output, got {tuple(output.shape)}."
        )

    # This application is intended for scalar depth models.
    if output.shape[1] != 1:
        output = output[:, :1]

    return output


# ---------------------------------------------------------------------
# Drawing and corruption
# ---------------------------------------------------------------------

def apply_draw_strokes(
    depth_m: np.ndarray,
    strokes: List[DrawStroke],
):
    """
    Applies browser-normalized line/point strokes at native image resolution.
    """

    height, width = depth_m.shape
    min_side = min(height, width)

    for stroke in strokes:
        if not stroke.points:
            continue

        depth_value_m = safe_float(
            stroke.depth_m,
            default=0.0,
        )

        if depth_value_m is None or depth_value_m < 0:
            continue

        radius = max(
            1,
            int(
                round(
                    clamp01(stroke.radius_norm)
                    * min_side
                )
            ),
        )

        points = []

        for x_norm, y_norm in stroke.points:
            x = int(
                round(
                    clamp01(x_norm)
                    * max(width - 1, 0)
                )
            )

            y = int(
                round(
                    clamp01(y_norm)
                    * max(height - 1, 0)
                )
            )

            points.append((x, y))

        if len(points) == 1:
            cv2.circle(
                depth_m,
                points[0],
                radius,
                float(depth_value_m),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            continue

        for point_a, point_b in zip(
            points[:-1],
            points[1:],
        ):
            cv2.line(
                depth_m,
                point_a,
                point_b,
                float(depth_value_m),
                thickness=max(1, radius * 2),
                lineType=cv2.LINE_AA,
            )


def coherent_noise_field(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Fallback coherent noise if utils.perlin_noise is unavailable.
    """

    field = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    total_weight = 0.0

    for scale_fraction, weight in [
        (0.025, 0.60),
        (0.060, 0.28),
        (0.140, 0.12),
    ]:
        coarse_height = max(
            2,
            int(round(height * scale_fraction)),
        )

        coarse_width = max(
            2,
            int(round(width * scale_fraction)),
        )

        coarse = rng.random(
            (coarse_height, coarse_width),
            dtype=np.float32,
        )

        upsampled = cv2.resize(
            coarse,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )

        field += weight * upsampled
        total_weight += weight

    field /= max(total_weight, 1e-8)

    field -= field.min()
    field /= max(field.max(), 1e-8)

    return field


def make_perlin_mask(
    valid_mask: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
    allow_project_perlin: bool,
) -> np.ndarray:
    height, width = valid_mask.shape
    threshold = clamp01(threshold)

    if threshold <= 0:
        return np.zeros_like(
            valid_mask,
            dtype=bool,
        )

    if threshold >= 1:
        return valid_mask.copy()

    if allow_project_perlin and HAS_PROJECT_PERLIN:
        try:
            mask = random_binary_perlin_noise(
                shape=(height, width),
                t=threshold,
            )

            if torch.is_tensor(mask):
                mask = mask.detach().cpu().numpy()

            mask = np.asarray(mask).astype(bool)

            if mask.shape == valid_mask.shape:
                return mask & valid_mask

        except Exception:
            pass

    field = coherent_noise_field(
        height,
        width,
        rng,
    )

    values = field[valid_mask]

    if values.size == 0:
        return np.zeros_like(
            valid_mask,
            dtype=bool,
        )

    cutoff = np.quantile(values, threshold)

    return (field <= cutoff) & valid_mask


def make_patch_mask(
    valid_mask: np.ndarray,
    threshold: float,
    patch_min: int,
    patch_max: int,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = valid_mask.shape
    threshold = clamp01(threshold)

    if threshold <= 0:
        return np.zeros_like(
            valid_mask,
            dtype=bool,
        )

    if threshold >= 1:
        return valid_mask.copy()

    patch_min = max(1, int(patch_min))
    patch_max = max(patch_min, int(patch_max))

    target_pixels = int(
        round(valid_mask.sum() * threshold)
    )

    mask = np.zeros_like(
        valid_mask,
        dtype=bool,
    )

    attempts = 0
    max_attempts = 3000

    while (
        int((mask & valid_mask).sum()) < target_pixels
        and attempts < max_attempts
    ):
        attempts += 1

        size = int(
            rng.integers(
                patch_min,
                patch_max + 1,
            )
        )

        x = int(rng.integers(0, max(width, 1)))
        y = int(rng.integers(0, max(height, 1)))

        x2 = min(width, x + size)
        y2 = min(height, y + size)

        mask[y:y2, x:x2] = True

    mask &= valid_mask

    remaining = target_pixels - int(mask.sum())

    if remaining > 0:
        available = np.flatnonzero(
            valid_mask & ~mask
        )

        if available.size > 0:
            chosen = rng.choice(
                available,
                size=min(remaining, available.size),
                replace=False,
            )

            mask.flat[chosen] = True

    return mask


def corrupt_depth(
    depth_m: np.ndarray,
    request: GenerateMissingDepthRequest,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        generated_sparse_depth_m
        random_corruption_mask
    """

    sparse_depth_m = depth_m.copy()

    valid = sparse_depth_m > 0
    threshold = clamp01(request.threshold)

    rng = np.random.default_rng(
        request.random_seed
    )

    removal_mask = np.zeros_like(
        valid,
        dtype=bool,
    )

    if request.use_perlin:
        removal_mask |= make_perlin_mask(
            valid,
            threshold,
            rng,
            allow_project_perlin=request.random_seed is None,
        )

    if request.use_pixel_noise:
        pixel_mask = rng.random(
            sparse_depth_m.shape
        ) < threshold

        removal_mask |= pixel_mask & valid

    if request.use_patch_cuts:
        removal_mask |= make_patch_mask(
            valid,
            threshold,
            request.patch_min,
            request.patch_max,
            rng,
        )

    sparse_depth_m[removal_mask] = 0.0

    return sparse_depth_m, removal_mask


# ---------------------------------------------------------------------
# Feature capture and PCA
# ---------------------------------------------------------------------

class EncoderFeatureCapture:
    """
    Captures color and depth encoder outputs during the same model forward pass.
    """

    def __init__(
        self,
        model: torch.nn.Module,
    ):
        self.model = model
        self.rgb_output = None
        self.depth_output = None
        self.handles = []

    def __enter__(self):
        encoder = getattr(
            self.model,
            "encoder",
            None,
        )

        color_encoder = getattr(
            encoder,
            "color_encoder",
            None,
        )

        depth_encoder = getattr(
            encoder,
            "depth_encoder",
            None,
        )

        if color_encoder is not None:
            self.handles.append(
                color_encoder.register_forward_hook(
                    self._color_hook
                )
            )

        if depth_encoder is not None:
            self.handles.append(
                depth_encoder.register_forward_hook(
                    self._depth_hook
                )
            )

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        exc_traceback,
    ):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()

    def _color_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.rgb_output = output

    def _depth_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.depth_output = output


def pick_spatial_feature(output):
    """
    Finds a spatial/tokens tensor from the wrapper output.

    Handles:
      Tensor
      List[Tensor]
      Tuple[List[Tensor], rdps]
      {"return": List[Tensor], "fusion": ...}
    """

    if output is None:
        return None

    if torch.is_tensor(output):
        if output.ndim in (3, 4):
            return output

        return None

    if isinstance(output, tuple):
        if not output:
            return None

        return pick_spatial_feature(output[0])

    if isinstance(output, dict):
        if "return" in output:
            return pick_spatial_feature(output["return"])

        if "fusion" in output:
            return pick_spatial_feature(output["fusion"])

        for value in reversed(list(output.values())):
            feature = pick_spatial_feature(value)

            if feature is not None:
                return feature

        return None

    if isinstance(output, (list, tuple)):
        for value in reversed(output):
            feature = pick_spatial_feature(value)

            if feature is not None:
                return feature

    return None


def infer_token_grid(
    token_count: int,
    input_height: int,
    input_width: int,
) -> Optional[Tuple[int, int]]:
    target_aspect = input_width / max(input_height, 1)

    best = None
    best_error = float("inf")

    for height in range(
        1,
        int(math.sqrt(token_count)) + 1,
    ):
        if token_count % height != 0:
            continue

        width = token_count // height
        aspect = width / max(height, 1)

        error = abs(aspect - target_aspect)

        if error < best_error:
            best_error = error
            best = (height, width)

    return best


def pca_feature_image(
    feature_output,
    process_height: int,
    process_width: int,
    original_height: int,
    original_width: int,
) -> Optional[np.ndarray]:
    """
    Returns PCA RGB visualization upsampled to original RGB resolution.
    """

    feature = pick_spatial_feature(feature_output)

    if feature is None:
        return None

    feature = (
        feature.detach()
        .to(device="cpu", dtype=torch.float32)
        .clone()
    )

    if feature.ndim == 4:
        # [B,C,H,W] -> [H,W,C]
        feature_map = feature[0].permute(
            1,
            2,
            0,
        ).numpy()

    elif feature.ndim == 3:
        # [B,tokens,C]
        tokens = feature[0]

        patch_size = model_patch_size()

        expected_height = process_height // patch_size
        expected_width = process_width // patch_size

        expected_tokens = (
            expected_height
            * expected_width
        )

        if tokens.shape[0] >= expected_tokens:
            # Removes CLS/register prefix tokens.
            tokens = tokens[-expected_tokens:]

            feature_map = tokens.reshape(
                expected_height,
                expected_width,
                tokens.shape[-1],
            ).numpy()

        else:
            grid = infer_token_grid(
                int(tokens.shape[0]),
                process_height,
                process_width,
            )

            if grid is None:
                return None

            grid_height, grid_width = grid

            feature_map = tokens.reshape(
                grid_height,
                grid_width,
                tokens.shape[-1],
            ).numpy()

    else:
        return None

    feature_height, feature_width, channels = feature_map.shape

    if feature_height * feature_width < 3 or channels < 3:
        return None

    flat = torch.from_numpy(
        feature_map.reshape(-1, channels)
    ).float()

    finite = torch.isfinite(flat).all(dim=1)

    if int(finite.sum()) < 3:
        return None

    valid_features = flat[finite]
    centered = valid_features - valid_features.mean(
        dim=0,
        keepdim=True,
    )

    sample_count = min(
        4096,
        centered.shape[0],
    )

    indices = torch.linspace(
        0,
        centered.shape[0] - 1,
        sample_count,
    ).long()

    try:
        _, _, vectors = torch.pca_lowrank(
            centered[indices],
            q=3,
            center=False,
            niter=4,
        )

        projected = centered @ vectors[:, :3]

    except Exception:
        return None

    rgb_flat = torch.zeros(
        (feature_height * feature_width, 3),
        dtype=torch.float32,
    )

    rgb_flat[finite] = projected

    for channel_index in range(3):
        values = rgb_flat[finite, channel_index]

        low = torch.quantile(
            values,
            0.01,
        )

        high = torch.quantile(
            values,
            0.99,
        )

        if float(high - low) < 1e-8:
            rgb_flat[:, channel_index] = 0.5

        else:
            rgb_flat[:, channel_index] = torch.clamp(
                (
                    rgb_flat[:, channel_index]
                    - low
                ) / (high - low),
                0.0,
                1.0,
            )

    rgb = (
        rgb_flat.reshape(
            feature_height,
            feature_width,
            3,
        ).numpy()
        * 255.0
    ).astype(np.uint8)

    # Important: PCA output is returned at original RGB resolution.
    return cv2.resize(
        rgb,
        (original_width, original_height),
        interpolation=cv2.INTER_CUBIC,
    )


# ---------------------------------------------------------------------
# Original depth SDP/DPE visualization
# ---------------------------------------------------------------------

def original_depth_sdp_cosine_panels(
    original_depth_m: np.ndarray,
    scale_to_model: float,
) -> dict:
    """
    Computes five cosine channels from ORIGINAL full depth.

    It intentionally does not use:
      - manually cleared depth,
      - manually corrupted depth,
      - Perlin/pixel/patch removals,
      - generated sparse depth.

    It does apply the same global metric scaling used before model inference,
    so the shown channel frequencies remain physically consistent with the
    model input coordinate range.
    """

    original_depth_m = original_depth_m.astype(
        np.float32,
        copy=True,
    )

    has_valid_depth = bool(
        np.any(original_depth_m > 0)
    )

    # Model-range scaling, but no spatial resizing:
    # resulting maps remain at original RGB resolution.
    depth = original_depth_m * scale_to_model

    depth[depth == 0] = float(DPE.zero_eps)

    normalization_mode = "raw metric depth"
    physical_normalization_max_m = None

    if DPE.get_depth_scales:
        normalization_mode = (
            "per-image original depth maximum"
        )

        max_model_depth = float(
            max(
                float(depth.max()),
                float(DPE.zero_eps),
            )
        )

        depth = depth / max_model_depth

        if has_valid_depth:
            physical_normalization_max_m = (
                max_model_depth
                / max(scale_to_model, 1e-12)
            )

    elif DPE.norm_with_max_scale:
        normalization_mode = (
            "checkpoint maximum depth"
        )

        depth = depth / float(DPE.max_depth)

        physical_normalization_max_m = (
            float(DPE.max_depth)
            / max(scale_to_model, 1e-12)
        )

    phase_depth = depth.copy()

    if DPE.scale > 0:
        phase_depth *= float(DPE.scale)

    dim_t = DPE.dim_t.detach().cpu().numpy()

    pair_count = len(dim_t) // 2

    if pair_count <= 0:
        return {
            "source": "Original full depth.",
            "normalization_mode": normalization_mode,
            "normalization_max_m": physical_normalization_max_m,
            "channels": [],
        }

    pair_indices = np.rint(
        np.linspace(
            0,
            pair_count - 1,
            5,
        )
    ).astype(int)

    channels = []

    for percent, pair_index in zip(
        [0, 25, 50, 75, 100],
        pair_indices,
    ):
        cosine_channel = int(
            pair_index * 2 + 1
        )

        dim_value = float(
            dim_t[cosine_channel]
        )

        cosine_map = np.cos(
            phase_depth / dim_value
        )

        if physical_normalization_max_m is not None:
            if DPE.scale > 0:
                unique_2pi_distance_m = (
                    physical_normalization_max_m
                    * 2.0
                    * math.pi
                    * dim_value
                    / float(DPE.scale)
                )
            else:
                unique_2pi_distance_m = (
                    physical_normalization_max_m
                    * 2.0
                    * math.pi
                    * dim_value
                )

        else:
            if DPE.scale > 0:
                unique_2pi_distance_m = (
                    2.0
                    * math.pi
                    * dim_value
                    / (
                        float(DPE.scale)
                        * max(scale_to_model, 1e-12)
                    )
                )
            else:
                unique_2pi_distance_m = (
                    2.0
                    * math.pi
                    * dim_value
                    / max(scale_to_model, 1e-12)
                )

        channels.append(
            {
                "temperature_percent": int(percent),
                "frequency_pair_index": int(pair_index),
                "cosine_channel": cosine_channel,
                "dim_t": dim_value,
                "unique_2pi_distance_m": (
                    float(unique_2pi_distance_m)
                    if has_valid_depth
                    else None
                ),
                # Original/native RGB resolution; no resizing.
                "image": png_data_url(
                    signed_feature_image(cosine_map),
                    max_side=None,
                ),
            }
        )

    return {
        "source": (
            "Original full depth before clear, drawing, and missing-depth generation."
            if has_valid_depth
            else (
                "No original depth was supplied. "
                "This is a zero depth image in RGB-only mode."
            )
        ),
        "normalization_mode": normalization_mode,
        "normalization_max_m": physical_normalization_max_m,
        "channels": channels,
    }


# ---------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------

def rmse(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> Optional[float]:
    valid = (
        mask
        & np.isfinite(prediction)
        & np.isfinite(target)
    )

    if not valid.any():
        return None

    error = (
        prediction[valid].astype(np.float64)
        - target[valid].astype(np.float64)
    )

    return float(
        np.sqrt(
            np.mean(
                np.square(error)
            )
        )
    )


# ---------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if CHECKPOINT_PATH.exists():
        checkpoint_display = CHECKPOINT_PATH.relative_to(
            ROOT
        ).as_posix()
    else:
        checkpoint_display = str(CHECKPOINT_PATH)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "checkpoint": checkpoint_display,
        },
    )


@app.get("/api/health")
def health():
    return {
        "ready": MODEL is not None,
        "device": "cuda:0",
        "checkpoint": str(CHECKPOINT_PATH),
    }


@app.get("/api/config")
def config():
    patch_size = model_patch_size()

    checkpoint_train_size = int(
        getattr(
            MODEL_ARGS,
            "train_size",
            448,
        )
    )

    default_process_size = round_to_patch(
        checkpoint_train_size,
        patch_size,
    )

    default_process_size = max(
        PROCESS_MIN_SIDE_MIN,
        default_process_size,
    )

    default_process_size = min(
        PROCESS_MIN_SIDE_MAX,
        default_process_size,
    )

    return {
        "patch_size": patch_size,
        "process_min_size_min": PROCESS_MIN_SIDE_MIN,
        "process_min_size_max": PROCESS_MIN_SIDE_MAX,
        "process_min_size_default": default_process_size,
        "checkpoint_max_depth_m": float(
            getattr(
                MODEL_ARGS,
                "max_depth",
                15.0,
            )
        ),
    }


@app.get("/api/demo-files")
def demo_files():
    return scan_demo_files()


@app.get("/api/demo-preview")
def demo_preview(
    kind: str,
    path: str,
):
    if kind not in {"rgb", "depth"}:
        raise HTTPException(
            status_code=400,
            detail="kind must be rgb or depth.",
        )

    source = SourceRef(
        kind="demo",
        id=path,
    )

    try:
        return source_payload(source, kind)

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/api/upload/{kind}")
async def upload_file(
    kind: str,
    file: UploadFile = File(...),
):
    if kind not in {"rgb", "depth"}:
        raise HTTPException(
            status_code=400,
            detail="kind must be rgb or depth.",
        )

    cleanup_uploads()

    filename = file.filename or (
        f"upload_{uuid.uuid4().hex}"
    )

    extension = Path(filename).suffix.lower()

    allowed_extensions = (
        RASTER_EXTENSIONS
        if kind == "rgb"
        else ALL_DEPTH_EXTENSIONS
    )

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported {kind} extension {extension}. "
                f"Allowed: {sorted(allowed_extensions)}"
            ),
        )

    upload_id = uuid.uuid4().hex
    destination = UPLOAD_DIR / f"{upload_id}{extension}"

    total_bytes = 0

    try:
        with destination.open("wb") as output_file:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file exceeds 80 MB.",
                    )

                output_file.write(chunk)

        # Validate data before registering upload.
        if kind == "rgb":
            _ = load_rgb(destination)
        else:
            _ = load_depth(destination)

    except HTTPException:
        destination.unlink(missing_ok=True)
        raise

    except Exception as exc:
        destination.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read uploaded {kind} file: {exc}"
            ),
        ) from exc

    finally:
        await file.close()

    with UPLOAD_LOCK:
        UPLOADS[upload_id] = UploadRecord(
            path=destination,
            purpose=kind,
            label=filename,
            created_at=time.time(),
        )

    source = SourceRef(
        kind="upload",
        id=upload_id,
    )

    return source_payload(source, kind)


# ---------------------------------------------------------------------
# Stage 1: Generate missing depth
# ---------------------------------------------------------------------

@app.post("/api/generate-missing-depth")
def generate_missing_depth(
    request: GenerateMissingDepthRequest,
):
    """
    First stage:

      - load RGB,
      - load optional original depth,
      - resize original depth to RGB resolution if needed,
      - optionally clear to zero,
      - add manual strokes,
      - generate Perlin/pixel/patch missing depth,
      - freeze all arrays in a short-lived stage cache.
    """

    rgb_path, _ = resolve_source(
        request.rgb,
        "rgb",
    )

    rgb = load_rgb(rgb_path)

    rgb_height, rgb_width = rgb.shape[:2]

    has_reference_depth = request.depth is not None

    if request.depth is not None:
        depth_path, _ = resolve_source(
            request.depth,
            "depth",
        )

        raw_depth = load_depth(depth_path)

        if raw_depth.shape != (
            rgb_height,
            rgb_width,
        ):
            raw_depth = cv2.resize(
                raw_depth,
                (rgb_width, rgb_height),
                interpolation=cv2.INTER_NEAREST,
            )

    else:
        # RGB-only mode:
        # zero depth exactly at RGB native resolution.
        raw_depth = np.zeros(
            (rgb_height, rgb_width),
            dtype=np.float32,
        )

    depth_scalar = safe_float(
        request.depth_scalar,
        default=None,
    )

    if depth_scalar is None or depth_scalar <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "Depth scalar must be a finite value greater than zero."
            ),
        )

    # Original depth/reference map in meters.
    reference_depth_m = (
        raw_depth.astype(np.float32)
        * float(depth_scalar)
    )

    reference_depth_m = np.nan_to_num(
        reference_depth_m,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    reference_depth_m[reference_depth_m < 0] = 0.0

    # Editable depth starts as reference depth.
    editable_depth_m = reference_depth_m.copy()

    # Clear applies to input only; it does not modify original reference depth.
    if request.clear_depth:
        editable_depth_m.fill(0.0)

    # Manual metric depth is inserted after clear.
    apply_draw_strokes(
        editable_depth_m,
        request.strokes,
    )

    editable_depth_m = np.nan_to_num(
        editable_depth_m,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    editable_depth_m[editable_depth_m < 0] = 0.0

    generated_depth_m, corruption_mask = corrupt_depth(
        editable_depth_m,
        request,
    )

    stage = GeneratedDepthStage(
        created_at=time.time(),
        rgb=rgb.copy(),
        reference_depth_m=reference_depth_m.copy(),
        editable_depth_m=editable_depth_m.copy(),
        generated_depth_m=generated_depth_m.copy(),
        corruption_mask=corruption_mask.copy(),
        has_reference_depth=has_reference_depth,
    )

    stage_id = store_stage(stage)

    reference_valid = reference_depth_m > 0
    generated_valid = generated_depth_m > 0

    observed_reference = (
        reference_valid
        & generated_valid
    )

    removed_reference = (
        reference_valid
        & ~generated_valid
    )

    depth_low, depth_high = common_depth_display_range(
        reference_depth_m,
        editable_depth_m,
        generated_depth_m,
    )

    return {
        "stage_id": stage_id,
        "native_shape": {
            "height": rgb_height,
            "width": rgb_width,
        },
        "has_reference_depth": has_reference_depth,
        "stats": {
            "reference_depth_m": depth_statistics(
                reference_depth_m
            ),
            "editable_depth_m": depth_statistics(
                editable_depth_m
            ),
            "generated_depth_m": depth_statistics(
                generated_depth_m
            ),
        },
        "coverage": {
            "reference_valid_pixels": int(
                reference_valid.sum()
            ),
            "observed_reference_pixels": int(
                observed_reference.sum()
            ),
            "removed_reference_pixels": int(
                removed_reference.sum()
            ),
            "removed_reference_percent": (
                float(
                    100.0
                    * removed_reference.sum()
                    / reference_valid.sum()
                )
                if reference_valid.any()
                else None
            ),
            "random_corruption_pixels": int(
                corruption_mask.sum()
            ),
        },
        "images": {
            "rgb": png_data_url(rgb),
            "original_depth_magma": png_data_url(
                magma_depth_image(
                    reference_depth_m,
                    reference_valid,
                    depth_low,
                    depth_high,
                )
            ),
            "editable_depth_magma": png_data_url(
                magma_depth_image(
                    editable_depth_m,
                    editable_depth_m > 0,
                    depth_low,
                    depth_high,
                )
            ),
            "generated_missing_depth_magma": png_data_url(
                magma_depth_image(
                    generated_depth_m,
                    generated_valid,
                    depth_low,
                    depth_high,
                )
            ),
        },
    }


# ---------------------------------------------------------------------
# Stage 2: Process frozen generated depth
# ---------------------------------------------------------------------

@app.post("/api/process")
def process_generated_depth(
    request: ProcessStageRequest,
):
    """
    Second stage:

      - retrieve frozen generated sparse depth,
      - resize RGB and sparse depth to selected processing min size,
      - apply DPE,
      - run model,
      - restore prediction to original RGB resolution,
      - compute metrics,
      - generate ORIGINAL-depth DPE panels,
      - upsample PCA maps to original RGB resolution.
    """

    if MODEL is None or MODEL_ARGS is None:
        raise HTTPException(
            status_code=503,
            detail="Model has not finished loading.",
        )

    stage = get_stage(request.stage_id)

    original_height, original_width = stage.rgb.shape[:2]

    process_height, process_width = get_processing_shape(
        original_height,
        original_width,
        request.process_min_size,
    )

    # Depth scale is determined from editable depth before random removal,
    # so removal severity does not alter the selected metric range.
    editable_valid = stage.editable_depth_m > 0

    editable_max_depth_m = (
        float(
            stage.editable_depth_m[
                editable_valid
            ].max()
        )
        if editable_valid.any()
        else 0.0
    )

    checkpoint_max_depth_m = float(
        getattr(
            MODEL_ARGS,
            "max_depth",
            15.0,
        )
    )

    if editable_max_depth_m > checkpoint_max_depth_m:
        scale_to_model = (
            checkpoint_max_depth_m
            / editable_max_depth_m
        )
    else:
        scale_to_model = 1.0

    restore_scale = 1.0 / max(
        scale_to_model,
        1e-12,
    )

    # Resize sparse depth and RGB together to processing shape.
    generated_depth_model_m = cv2.resize(
        stage.generated_depth_m * scale_to_model,
        (process_width, process_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)

    rgb_model = cv2.resize(
        stage.rgb,
        (process_width, process_height),
        interpolation=cv2.INTER_CUBIC,
    )

    rgb_tensor = normalize_rgb_for_model(
        rgb_model
    )

    try:
        with MODEL_LOCK:
            encoded_depth, depth_scales = encode_depth_for_model(
                generated_depth_model_m
            )

            with torch.inference_mode():
                with EncoderFeatureCapture(
                    MODEL
                ) as feature_capture:
                    output = MODEL(
                        rgb_tensor,
                        encoded_depth,
                        depth_scales,
                    )

                prediction = extract_final_depth_prediction(
                    output
                )

                # Return prediction to original RGB/native resolution.
                prediction = F.interpolate(
                    prediction,
                    size=(
                        original_height,
                        original_width,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )

            completed_depth_m = (
                prediction[0, 0]
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            completed_depth_m = np.maximum(
                np.nan_to_num(
                    completed_depth_m,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                0.0,
            )

            # Undo scaling when source depth was above checkpoint max range.
            completed_depth_m *= restore_scale

            # PCA maps are upsampled to original RGB resolution.
            rgb_pca = pca_feature_image(
                feature_capture.rgb_output,
                process_height,
                process_width,
                original_height,
                original_width,
            )

            depth_pca = pca_feature_image(
                feature_capture.depth_output,
                process_height,
                process_width,
                original_height,
                original_width,
            )

            # SDP/DPE must use original depth, not sparse/masked depth.
            sdp = original_depth_sdp_cosine_panels(
                stage.reference_depth_m,
                scale_to_model,
            )

    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()

        raise HTTPException(
            status_code=507,
            detail=(
                "CUDA out of memory. "
                "Reduce processing minimum size and try again."
            ),
        ) from exc

    reference_valid = stage.reference_depth_m > 0

    observed_mask = (
        reference_valid
        & (stage.generated_depth_m > 0)
    )

    removed_mask = (
        reference_valid
        & ~observed_mask
    )

    global_rmse = rmse(
        completed_depth_m,
        stage.reference_depth_m,
        reference_valid,
    )

    removed_rmse = rmse(
        completed_depth_m,
        stage.reference_depth_m,
        removed_mask,
    )

    observed_rmse = rmse(
        completed_depth_m,
        stage.reference_depth_m,
        observed_mask,
    )

    depth_low, depth_high = common_depth_display_range(
        stage.reference_depth_m,
        stage.editable_depth_m,
        stage.generated_depth_m,
        completed_depth_m,
    )

    absolute_error = np.abs(
        completed_depth_m
        - stage.reference_depth_m
    )

    error_low, error_high = robust_range(
        absolute_error,
        reference_valid,
    )

    return {
        "model": {
            "device": "cuda:0",
            "process_height": process_height,
            "process_width": process_width,
            "process_min_size": request.process_min_size,
            "patch_size": model_patch_size(),
            "checkpoint_max_depth_m": checkpoint_max_depth_m,
            "editable_max_depth_m": editable_max_depth_m,
            "scale_to_model": float(scale_to_model),
            "restore_scale": float(restore_scale),
        },
        "metrics": {
            "global_rmse_m": global_rmse,
            "removed_area_rmse_m": removed_rmse,
            "observed_area_rmse_m": observed_rmse,
        },
        "coverage": {
            "reference_valid_pixels": int(
                reference_valid.sum()
            ),
            "observed_reference_pixels": int(
                observed_mask.sum()
            ),
            "removed_reference_pixels": int(
                removed_mask.sum()
            ),
            "removed_reference_percent": (
                float(
                    100.0
                    * removed_mask.sum()
                    / reference_valid.sum()
                )
                if reference_valid.any()
                else None
            ),
        },
        "stats": {
            "reference_depth_m": depth_statistics(
                stage.reference_depth_m
            ),
            "generated_depth_m": depth_statistics(
                stage.generated_depth_m
            ),
            "completed_depth_m": depth_statistics(
                completed_depth_m
            ),
        },
        "images": {
            "rgb": png_data_url(stage.rgb),
            "original_depth_magma": png_data_url(
                magma_depth_image(
                    stage.reference_depth_m,
                    reference_valid,
                    depth_low,
                    depth_high,
                )
            ),
            "generated_depth_magma": png_data_url(
                magma_depth_image(
                    stage.generated_depth_m,
                    stage.generated_depth_m > 0,
                    depth_low,
                    depth_high,
                )
            ),
            "completed_depth_magma": png_data_url(
                magma_depth_image(
                    completed_depth_m,
                    completed_depth_m > 0,
                    depth_low,
                    depth_high,
                )
            ),
            "absolute_error_magma": png_data_url(
                magma_depth_image(
                    absolute_error,
                    reference_valid,
                    error_low,
                    error_high,
                )
            ),
        },
        "sdp": sdp,
        "pca": {
            "rgb": (
                png_data_url(
                    rgb_pca,
                    max_side=None,
                )
                if rgb_pca is not None
                else None
            ),
            "depth": (
                png_data_url(
                    depth_pca,
                    max_side=None,
                )
                if depth_pca is not None
                else None
            ),
        },
    }


# ---------------------------------------------------------------------
# Standalone launch
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        workers=1,
        reload=False,
    )