#!/usr/bin/env python3
"""
web_dashboard.py — Multi-view inference dashboard.
"""

import io
import time
import base64
import threading
import collections
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

from flask import Flask, render_template_string
from flask_socketio import SocketIO
from PIL import Image as PILImage, ImageDraw

from common.helper_fnc import tcp_pose_from_relative


# ─────────────────────────────────────────────────────────────────────────────
#  Camera geometry
# ─────────────────────────────────────────────────────────────────────────────

FLANGE_T_CAM_POS       = np.array([-0.006056, 0.0907, 0.080564], dtype=np.float64)
FLANGE_T_CAM_QUAT_XYZW = np.array([0.086947, -0.002139, -0.010563, 0.99615], dtype=np.float64)
CAM_FX, CAM_FY         = 651.265475, 651.265475
CAM_CX, CAM_CY         = 627.629144, 360.407377


def _hmat(pos, quat):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3]  = pos
    return T

def _hmat_inv(T):
    Ri = T[:3, :3].T
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = Ri
    Ti[:3, 3]  = -Ri @ T[:3, 3]
    return Ti

FLANGE_T_CAM = _hmat(FLANGE_T_CAM_POS, FLANGE_T_CAM_QUAT_XYZW)


# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DashboardConfig:
    obs_interval_ms:          float = 100.0
    min_inference_ms:         float = 200.0
    sync_tolerance_ms:        float = 15.0
    pred_display_indices:     List[int] = field(default_factory=lambda: [0, 15])
    coord_arrow_length_px:    int   = 40
    coord_arrow_width_px:     int   = 3
    schnapp_cut_threshold_cm: float = 1.0
    schnapp_cut_hold_sec:     float = 1.0
    robot_publish_enabled:    bool  = True
    auto_stop_on_cut:         bool  = True
    schnapp_avg_window:       int   = 1
    schnapp_bar_max_cm:       float = 5.0
    schnapp_bar_max_deg:      float = 10.0
    dist_2_schnapp_cm:        float = 15.0
    deg_2_schnapp_deg:        float = 90.0
    freeze_depth:             bool  = False
    zero_depth:               bool  = False
    cut_stop_delay_ms:        float = 100.0  
    pose_limiting_index:      int   = 0
    depth_far_away_threshold_cm: float = 7.0
    min_depth_filter_cm: float = 16.0

    


# ─────────────────────────────────────────────────────────────────────────────
#  State
# ─────────────────────────────────────────────────────────────────────────────

class DashboardState:
    def __init__(self):
        self._lock  = threading.RLock()
        self.config = DashboardConfig()

        self.obs_frames: List[dict] = []
        self.current_prediction: Optional[dict] = None
        self.traj_history    = collections.deque(maxlen=300)
        self.schnapp_history = collections.deque(maxlen=100)

        self.last_enc_ms:     float = 0.0
        self.last_dif_ms:     float = 0.0
        self.last_pub_ms:     float = 0.0
        self.last_obs_gap_ms: float = 0.0

        self.frozen_depth: Optional[np.ndarray] = None
        self.stopped_by_cut: bool = False
        self._cut_first_seen: Optional[float] = None   

        self.far_away: bool = False              
        self.min_depth_cm: float = 0.0  

    def update_obs(self, frames: List[dict]):
        with self._lock:
            self.obs_frames = frames

    def update_prediction(self, pred: dict):
        with self._lock:
            self.current_prediction = pred

    def add_traj_point(self, stamp, pos, quat, speed):
        with self._lock:
            self.traj_history.append({'stamp': stamp, 'pos': pos.copy(),
                                      'quat': quat.copy(), 'speed': speed})
            cutoff = stamp - 3.0
            while self.traj_history and self.traj_history[0]['stamp'] < cutoff:
                self.traj_history.popleft()

    def add_schnapp(self, stamp, dist_pct, deg_pct):
        with self._lock:
            self.schnapp_history.append({'stamp': stamp,
                                         'dist_pct': dist_pct, 'deg_pct': deg_pct})
            cutoff = stamp - 3.0
            while self.schnapp_history and self.schnapp_history[0]['stamp'] < cutoff:
                self.schnapp_history.popleft()
        
    def get_schnapp_avg(self) -> Tuple[float, float]:
        with self._lock:
            n = self.config.schnapp_avg_window
            if not self.schnapp_history:
                return 0.0, 0.0
            
            if n <= 1:
                return (self.schnapp_history[-1]['dist_pct'],
                        self.schnapp_history[-1]['deg_pct'])
            
            recent = list(self.schnapp_history)[-n:]
            return (sum(s['dist_pct'] for s in recent) / len(recent),
                    sum(s['deg_pct']  for s in recent) / len(recent))

    def is_cut_now(self) -> bool:
        with self._lock:
            if not self.schnapp_history:
                return False
            now       = self.schnapp_history[-1]['stamp']
            hold      = self.config.schnapp_cut_hold_sec
            scale     = self.config.dist_2_schnapp_cm
            n         = self.config.schnapp_avg_window
            n = n if n > 0 else 1
            thresh_cm = self.config.schnapp_cut_threshold_cm
            window = [s for s in self.schnapp_history if (now - s['stamp']) <= hold]
            if len(window) < n:
                return False
            for i in range(len(window) - n + 1):
                chunk = window[i:i + n]
                avg_cm = sum(s['dist_pct'] for s in chunk) / len(chunk) * scale
                if avg_cm >= thresh_cm:
                    return False
            return True

    def get_snapshot(self) -> dict:
        with self._lock:
            avg_d, avg_r = self.get_schnapp_avg()
        
            # Compute distance-to-schnapp as a fraction [0.0, 1.0]
            # where 0.0 = sehr nah (schneiden), 1.0 = weit weg
            scale_cm = self.config.dist_2_schnapp_cm
            dist_fraction = 1.0 if scale_cm == 0 else min(1.0, avg_d * scale_cm / scale_cm)
            # Alternative: if avg_d is already in cm, use directly:
            dist_fraction = min(1.0, max(0.0, avg_d / scale_cm)) if scale_cm > 0 else 1.0

            # Remaining delay until publish is actually blocked
            cut_pending_ms = 0.0
            if (self._cut_first_seen is not None
                    and self.config.robot_publish_enabled
                    and not self.stopped_by_cut):
                elapsed_ms = (time.time() - self._cut_first_seen) * 1e3
                cut_pending_ms = max(0.0, self.config.cut_stop_delay_ms - elapsed_ms)

            return {
                'obs_frames': list(self.obs_frames),
                'prediction': self.current_prediction,
                'traj':       list(self.traj_history),
                'schnapp':    list(self.schnapp_history),
                'schnapp_avg_dist_pct': avg_d,
                'schnapp_avg_deg_pct':  avg_r,
                'schnapp_dist_fraction': dist_fraction,
                'cut_now': self.is_cut_now(),
                'far_away': self.far_away,            
                'min_depth_cm': self.min_depth_cm,    
                'config': {
                    'obs_interval_ms':          self.config.obs_interval_ms,
                    'min_inference_ms':         self.config.min_inference_ms,
                    'sync_tolerance_ms':        self.config.sync_tolerance_ms,
                    'pred_display_indices':     self.config.pred_display_indices,
                    'coord_arrow_length_px':    self.config.coord_arrow_length_px,
                    'coord_arrow_width_px':     self.config.coord_arrow_width_px,
                    'schnapp_cut_threshold_cm': self.config.schnapp_cut_threshold_cm,
                    'robot_publish_enabled':    self.config.robot_publish_enabled,
                    'auto_stop_on_cut':         self.config.auto_stop_on_cut,
                    'stopped_by_cut':           self.stopped_by_cut,
                    'schnapp_avg_window':       self.config.schnapp_avg_window,
                    'schnapp_bar_max_cm':       self.config.schnapp_bar_max_cm,
                    'schnapp_bar_max_deg':      self.config.schnapp_bar_max_deg,
                    'dist_2_schnapp_cm':        self.config.dist_2_schnapp_cm,
                    'deg_2_schnapp_deg':        self.config.deg_2_schnapp_deg,
                    'freeze_depth':             self.config.freeze_depth,
                    'zero_depth':               self.config.zero_depth,
                    'cut_stop_delay_ms':        self.config.cut_stop_delay_ms,   
                    'cut_pending_ms':           cut_pending_ms,                  
                    'pose_limiting_index':      self.config.pose_limiting_index, 
                    'depth_far_away_threshold_cm': self.config.depth_far_away_threshold_cm, 
                    'min_depth_filter_cm': self.config.min_depth_filter_cm,  
                },
                'timing': {
                    'enc_ms':     self.last_enc_ms,
                    'dif_ms':     self.last_dif_ms,
                    'pub_ms':     self.last_pub_ms,
                    'obs_gap_ms': self.last_obs_gap_ms,
                },
            }
        

# ─────────────────────────────────────────────────────────────────────────────
#  Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_magma(depth_m: np.ndarray, max_depth: float = 0.24) -> np.ndarray:
    d = depth_m.copy()
    valid = d > 0
    d_norm = np.zeros_like(d)
    if valid.any():
        d_norm[valid] = np.clip(d[valid] / max_depth, 0, 1)
    colored = (cm.magma(d_norm)[:, :, :3] * 255).astype(np.uint8)
    colored[~valid] = [30, 30, 30]
    return colored


def _to_b64(arr: np.ndarray, w: int = 240, h: int = 180) -> str:
    pil = PILImage.fromarray(arr.astype(np.uint8)).resize((w, h))
    buf = io.BytesIO()
    pil.save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _draw_coord_frame(draw, origin, axes_px, colors, width=2):
    if origin is None:
        return
    for end, color in zip(axes_px, colors):
        if end is None:
            continue
        draw.line([origin, end], fill=color, width=width)


def render_prediction_overlay(
    image_np, pred_abs_poses, display_indices,
    ref_flange_pos, ref_flange_quat,
    arrow_length_m=0.015, arrow_width_px=3,
    crop_region=(0, 0, 1280, 720),
    ref_tcp_pos=None, ref_tcp_quat=None,
) -> np.ndarray:
    img_h, img_w = image_np.shape[:2]
    cl, ct, _, _ = crop_region

    base_T_fl  = _hmat(ref_flange_pos, ref_flange_quat)
    base_T_cam = base_T_fl @ FLANGE_T_CAM
    cam_T_base = _hmat_inv(base_T_cam)

    def proj(p_base):
        p = (cam_T_base @ np.append(p_base, 1.0))[:3]
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        if z <= 1e-4:
            return None
        u = CAM_FX * x / z + CAM_CX - cl
        v = CAM_FY * y / z + CAM_CY - ct
        if u < -50 or u > img_w + 50 or v < -50 or v > img_h + 50:
            return None
        return int(round(u)), int(round(v))

    pil   = PILImage.fromarray(image_np.copy())
    draw  = ImageDraw.Draw(pil)
    ax_cl = [(255, 50, 50), (50, 255, 50), (50, 50, 255)]

    if ref_tcp_pos is not None:
        tcp_px = proj(ref_tcp_pos.astype(np.float64))
        if tcp_px:
            r = 8
            draw.ellipse([tcp_px[0]-r, tcp_px[1]-r,
                          tcp_px[0]+r, tcp_px[1]+r],
                         fill=(255, 0, 0), outline=(180, 0, 0), width=2)
            try:
                draw.text((tcp_px[0]+10, tcp_px[1]-6), 'TCP', fill=(255, 0, 0))
            except Exception:
                pass

    for step_idx in display_indices:
        if step_idx >= len(pred_abs_poses):
            continue
        p_pos, p_quat = pred_abs_poses[step_idx]
        origin_px = proj(p_pos)
        if origin_px is None:
            continue
        rot = R.from_quat(p_quat).as_matrix()
        axes = [proj(p_pos + rot[:, ax] * arrow_length_m) for ax in range(3)]
        _draw_coord_frame(draw, origin_px, axes, ax_cl, width=arrow_width_px)
        try:
            draw.text((origin_px[0]+4, origin_px[1]-8),
                      f's{step_idx+1}', fill=(255, 220, 0))
        except Exception:
            pass

    return np.array(pil)


# ─────────────────────────────────────────────────────────────────────────────
#  HTML Template
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""
<!DOCTYPE html><html>
<head>
<title>Diffusion Policy — Multi-View Dashboard</title>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
:root{--bg:#fff;--panel:#f4f4f4;--inp:#fff;--fg:#111;--fg2:#555;--bdr:#ddd;--acc:#0077cc}
[data-theme=dark]{--bg:#12121f;--panel:#1b1b2e;--inp:#0f2a44;--fg:#eee;--fg2:#999;--bdr:#333;--acc:#00d4ff}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:var(--bg);color:var(--fg);padding:10px}
h1{color:var(--acc);margin-bottom:8px;font-size:1.3em}
h2{color:var(--fg2);font-size:.95em;margin:6px 0 3px}

/* ── two-column page layout ──────────────────────────────────── */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.panel{background:var(--panel);border-radius:8px;padding:10px;border:1px solid var(--bdr)}

/* ── observation grid: RGB row + Depth row ───────────────────── *
 *  Columns = BASE-t0 | BASE-t1 | … | UP-t0 | UP-t1 | …          *
 *  --obs-cols is set dynamically by JS                            */
.obs-grid{
  display:grid;
  grid-template-columns:repeat(var(--obs-cols,4), 1fr);
  gap:4px;
}
.obs-cell{
  display:flex;
  flex-direction:column;
  align-items:center;
  gap:2px;
}
.obs-cell img{
  width:100%;
  border-radius:3px;
  border:1px solid var(--bdr);
  display:block;
  /* crossfade instead of blink */
  transition:opacity 0.08s ease;
}
.obs-cell img.loading{opacity:0}
.obs-cell-lbl{font-size:.62em;color:var(--acc);font-weight:600;white-space:nowrap}
.obs-row-lbl{
  font-size:.68em;color:var(--fg2);font-weight:bold;
  grid-column:1/-1;               /* span full row */
  margin:2px 0 0;
  border-top:1px solid var(--bdr);
  padding-top:3px;
}
.obs-cell-info{font-size:.60em;color:var(--fg2);text-align:center;white-space:nowrap}

/* ── rest of UI ──────────────────────────────────────────────── */
.pred-img{width:100%;border-radius:4px;border:2px solid var(--bdr)}
.cut{background:#f00;color:#fff;font-size:1.8em;font-weight:bold;padding:15px;
     text-align:center;border-radius:6px;display:none;
     animation:pulse .5s infinite alternate}
@keyframes pulse{from{opacity:.7}to{opacity:1}}
.bar{height:22px;background:#ccc;border-radius:3px;margin:3px 0;
     overflow:hidden;position:relative}
[data-theme=dark] .bar{background:#333}
.bar-fill{height:100%;transition:width .25s ease-out, background .25s;border-radius:3px}
.bar-lbl{position:absolute;top:3px;left:6px;font-size:.72em;font-weight:bold;color:#222}
.settings{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.settings label{font-size:.82em;color:var(--fg2)}
.settings input{width:100%;padding:3px;background:var(--inp);
                border:1px solid var(--bdr);color:var(--fg);border-radius:2px}
.btn{padding:6px 14px;border:none;border-radius:3px;cursor:pointer;
     font-weight:bold;margin:2px}
.btn-stop{background:#f44;color:#fff}
.btn-apply{background:#48f;color:#fff}
.btn-theme{background:#666;color:#fff}
.tag{display:inline-block;padding:3px 8px;border-radius:3px;
     font-size:.82em;font-weight:bold}
.on{background:#4f4;color:#111}.off{background:#f44;color:#fff}
.frozen{background:#f80;color:#fff}
.timing-bar{font-size:.8em;color:var(--fg2);margin-top:3px}
.traj{width:100%;height:260px}
.inter-timing{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
.inter-chip{background:var(--inp);border:1px solid var(--bdr);border-radius:4px;
            padding:3px 7px;font-size:.75em;color:var(--fg2)}
.inter-chip span{color:var(--acc);font-weight:bold}
.cut-pending{
  background:#f80;color:#fff;font-size:1.1em;font-weight:bold;
  padding:8px 12px;border-radius:6px;display:none;
  border:2px solid #c60;margin-bottom:4px;
}
</style>
</head>
<body>
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
  <h1>🤖 Diffusion Policy — Multi-View Dashboard</h1>
  <button class="btn btn-theme" onclick="toggleTheme()">🌙 Dark</button>
</div>
<div id="cut" class="cut">✂️ Jetzt Schneiden ✂️</div>
<div id="cut-pending" class="cut-pending">
  ⏳ CUT detected — stopping in <span id="cut-pending-ms">--</span> ms
</div>

<div class="grid2">
<!-- ═══ LEFT ══════════════════════════════════════════════════ -->
<div>

  <!--
    Single observation panel.
    Structure (rebuilt once, updated in-place every tick):

      [row label: RGB]
      BASE t0 RGB | BASE t1 RGB | … | UP t0 RGB | UP t1 RGB | …

      [row label: Depth]
      BASE t0 D   | BASE t1 D   | … | UP t0 D   | UP t1 D   | …

    Column headers + per-cell info live inside each .obs-cell.
  -->
  <div class="panel">
    <h2>📷 Observation Horizon</h2>
    <div id="obs-grid" class="obs-grid"></div>
  </div>

  <div class="panel" style="margin-top:8px">
    <h2>⏱ Inter-Camera Timing (base ↔ up, per obs step)</h2>
    <div id="inter-timing" class="inter-timing"></div>
    <div class="timing-bar" id="inter-timing-info"></div>
  </div>

  


  <div class="panel" style="margin-top:8px">
    <h2>📏 Sensor Status & Distance to Schnapp</h2>
    
    <!-- ── Far Away Detection ────────────────────────────────── -->
    <div style="margin-bottom:8px">
        <div>Min Depth: <b><span id="d-min-depth">--</span> cm</b></div>
        <div>Threshold: <b><span id="d-depth-threshold">30</span> cm</b></div>
        Far Away: <span id="far-away-tag" class="tag off">NO</span>
    </div>
    
    <!-- ── Position & Rotation Bars ──────────────────────── -->
    <div style="border-top:1px solid var(--bdr);padding-top:8px">
        <div>Position: <b><span id="d-dist">--</span> cm</b> 
        <span id="dist-source" style="font-size:0.8em;color:var(--fg2)">(computed)</span>
        </div>
        <div class="bar">
        <div id="b-dist" class="bar-fill" style="width:0%;background:#4f4"></div>
        <span class="bar-lbl" id="l-dist">0%</span>
        </div>
        <div>Rotation: <b><span id="d-rot">--</span>°</b></div>
        <div class="bar">
        <div id="b-rot" class="bar-fill" style="width:0%;background:#48f"></div>
        <span class="bar-lbl" id="l-rot">0%</span>
        </div>
    </div>
    </div>

  <div class="panel" style="margin-top:8px">
    <h2>🛤 Trajectory (3 s)</h2>
    <div id="traj" class="traj"></div>
  </div>

</div>
<!-- ═══ RIGHT ═════════════════════════════════════════════════ -->
<div>

  <div class="panel">
    <h2>🔮 Prediction Overlay (BASE view)</h2>
    <img id="pred-img" class="pred-img" src="" alt="waiting...">
    <div class="timing-bar" id="pred-info"></div>
  </div>

  <div class="panel" style="margin-top:8px">
    <h2>🕹 Controls</h2>
    Robot: <span id="robot-tag" class="tag on">ON</span>
    <button class="btn btn-stop" onclick="toggleRobot()">Toggle</button>
    <br><br>
    Auto-Stop on CUT: <span id="autostop-tag" class="tag on">ON</span>
    <button class="btn btn-apply" onclick="toggleAutoStop()">Toggle</button>
    <br><br>
    Freeze Depth: <span id="freeze-tag" class="tag off">OFF</span>
    <button class="btn btn-apply" onclick="toggleFreezeDepth()">Toggle</button>
    &nbsp;
    Zero Depth: <span id="zero-tag" class="tag off">OFF</span>
    <button class="btn btn-apply" onclick="toggleZeroDepth()">Toggle</button>
    <div class="timing-bar" id="timing-display"></div>
  </div>

  <div class="panel" style="margin-top:8px">
    <h2>⚙️ Settings</h2>
    <div class="settings">
      <label>obs_interval_ms<input id="s-obs"   type="number" step="1"   value="100"></label>
      <label>min_inference_ms<input id="s-inf"   type="number" step="1"   value="200"></label>
      <label>sync_tolerance_ms<input id="s-sync" type="number" step="1"   value="15"></label>
      <label>cut_threshold_cm<input id="s-cut"   type="number" step="0.1" value="1.0"></label>
      <label>avg_window<input id="s-avg"          type="number" step="1"   value="1"></label>
      <label>bar_max_cm<input id="s-bmcm"         type="number" step="0.5" value="5"></label>
      <label>bar_max_deg<input id="s-bmdeg"        type="number" step="1"   value="10"></label>
      <label>arrow_len_px<input id="s-alen"        type="number" step="1"   value="40"></label>
      <label>arrow_w_px<input id="s-awid"          type="number" step="1"   value="3"></label>
      <label>pred_steps (comma)<input id="s-pidx"  type="text"              value="1,16"></label>
      <label>cut_stop_delay_ms<input id="s-cdelay" type="number" step="50" value="100"></label>
      <label>pose_limiting_index<input id="s-pose-lim" type="number" step="1" value="0"></label>
      <label>depth_far_away_cm<input id="s-far-depth" type="number" step="1" value="7"></label>
      <label>depth_min_cm<input id="s-min-depth" type="number" step="1" value="16"></label>
    </div>
    <button class="btn btn-apply" style="margin-top:6px" onclick="applySettings()">Apply</button>
  </div>

</div>
</div><!-- .grid2 -->

<script>
const socket = io();
let dark = false;

// ─── EMA state for schnapp bars (avoids per-tick jitter) ──────────────────
const EMA_ALPHA = 1.0;   // lower = smoother, higher = more responsive
let _emaDist = null, _emaDeg = null;

socket.on('update', d => {
  renderObs(d.obs_base, d.obs_up);
  renderInterTiming(d.inter_timing);
  renderSchnapp(d);
  renderTraj(d.traj);
  renderPred(d.pred_img, d.pred_info);
  renderTiming(d.timing);
  renderConfig(d.config);
  document.getElementById('cut').style.display = d.cut_now ? 'block' : 'none';
});

// ─────────────────────────────────────────────────────────────────────────────
//  swapImg — pre-decode in an off-screen Image so the visible element never
//  shows a blank frame.  The data-b64 guard skips unchanged images entirely.
// ─────────────────────────────────────────────────────────────────────────────
function swapImg(img, b64) {
  if (!img || !b64) return;
  if (img.dataset.b64 === b64) return;          // truly unchanged → skip
  img.dataset.b64 = b64;
  const src = 'data:image/png;base64,' + b64;
  const tmp = new window.Image();
  tmp.onload = () => {
    img.classList.remove('loading');
    img.src = tmp.src;                          // decoded → no blank flash
  };
  tmp.onerror = () => img.classList.remove('loading');
  img.classList.add('loading');
  tmp.src = src;
}

// ─────────────────────────────────────────────────────────────────────────────
//  renderObs
//
//  Layout (single .obs-grid):
//
//    [ROW LABEL "RGB"]
//    BASE-t0 RGB | BASE-t1 RGB | … | UP-t0 RGB | UP-t1 RGB | …
//
//    [ROW LABEL "Depth"]
//    BASE-t0 D   | BASE-t1 D   | … | UP-t0 D   | UP-t1 D   | …
//
//  DOM is rebuilt only when the column-count changes.
// ─────────────────────────────────────────────────────────────────────────────
function renderObs(baseFrames, upFrames) {
  const nb   = (baseFrames || []).length;
  const nu   = (upFrames   || []).length;
  const cols = nb + nu;
  if (!cols) return;

  const grid = document.getElementById('obs-grid');

  // Rebuild skeleton only when number of columns changes
  if (parseInt(grid.dataset.cols || '0') !== cols) {
    grid.dataset.cols = cols;
    grid.style.setProperty('--obs-cols', cols);

    // Build column-header cells (appear once at the top of each row section)
    const makeCell = (cam, idx, rowType) =>
      `<div class="obs-cell" data-cam="${cam}" data-idx="${idx}" data-type="${rowType}">
         <div class="obs-cell-lbl">${cam}&nbsp;t${idx}</div>
         <img alt="${cam}-${rowType}">
         <div class="obs-cell-info"></div>
       </div>`;

    const rgbLabel   = `<div class="obs-row-lbl">▸ RGB</div>`;
    const depLabel   = `<div class="obs-row-lbl">▸ Depth</div>`;

    const rgbCells   = [
      ...Array.from({length: nb}, (_, i) => makeCell('BASE', i, 'rgb')),
      ...Array.from({length: nu}, (_, i) => makeCell('UP',   i, 'rgb')),
    ].join('');
    const depCells   = [
      ...Array.from({length: nb}, (_, i) => makeCell('BASE', i, 'dep')),
      ...Array.from({length: nu}, (_, i) => makeCell('UP',   i, 'dep')),
    ].join('');

    grid.innerHTML = rgbLabel + rgbCells + depLabel + depCells;
  }

  // Helper: update one cell
  function fillCell(cam, idx, rowType, frame) {
    const cell = grid.querySelector(
      `.obs-cell[data-cam="${cam}"][data-idx="${idx}"][data-type="${rowType}"]`);
    if (!cell || !frame) return;
    swapImg(cell.querySelector('img'),
            rowType === 'rgb' ? frame.rgb_b64 : frame.dep_b64);
    const info = cell.querySelector('.obs-cell-info');
    if (rowType === 'rgb') {
      info.textContent =
        `t-${(frame.time_past_ms||0).toFixed(0)}ms `
        + `Δ${(frame.disp_mm||0).toFixed(1)}mm ${(frame.rot_deg||0).toFixed(1)}°`;
    } else {
      info.textContent = `depth ${(frame.mean_depth_mm||0).toFixed(0)} mm`;
    }
  }

  (baseFrames||[]).forEach((f, i) => { fillCell('BASE', i, 'rgb', f); fillCell('BASE', i, 'dep', f); });
  (upFrames  ||[]).forEach((f, i) => { fillCell('UP',   i, 'rgb', f); fillCell('UP',   i, 'dep', f); });
}

// ─────────────────────────────────────────────────────────────────────────────
function renderInterTiming(chips) {
  const el = document.getElementById('inter-timing');
  if (!chips || !chips.length) { el.innerHTML = '--'; return; }
  el.innerHTML = chips.map((c, i) =>
    `<div class="inter-chip">obs${i} base↔up: <span>${c.delta_ms.toFixed(1)}ms</span>
     ${c.has_real_up ? '✓' : '<i>(no up)</i>'}</div>`
  ).join('');
  const deltas = chips.map(c => c.delta_ms).filter(v => !isNaN(v));
  document.getElementById('inter-timing-info').textContent = deltas.length
    ? `avg Δ${(deltas.reduce((a,b)=>a+b,0)/deltas.length).toFixed(1)}ms `
      + `| max Δ${Math.max(...deltas).toFixed(1)}ms`
    : '';
}

// ─────────────────────────────────────────────────────────────────────────────
//  renderSchnapp — EMA-smoothed so bars don't jump on every tick
// ─────────────────────────────────────────────────────────────────────────────
function renderSchnapp(d) {
  const cfg    = d.config;
  const far    = d.far_away || false;  // ← NEW
  const minD   = d.min_depth_cm || 0.0;
  
  // Display min depth
  document.getElementById('d-min-depth').textContent = minD.toFixed(2);
  document.getElementById('d-depth-threshold').textContent = 
    cfg.depth_far_away_threshold_cm.toFixed(1);
  
  // Update far-away tag
  const farTag = document.getElementById('far-away-tag');
  if (far) {
    farTag.textContent = 'YES';
    farTag.className = 'tag frozen';
  } else {
    farTag.textContent = 'NO';
    farTag.className = 'tag off';
  }
  
  // ── Schnapp distances (geforcet zu 1.0 wenn far_away) ────
  let rawDc = d.schnapp_avg_dist_pct * cfg.dist_2_schnapp_cm;
  let rawDg = d.schnapp_avg_deg_pct  * cfg.deg_2_schnapp_deg;
  
  // Override mit 1.0 wenn far_away (normalisiert)
  if (far) {
    rawDc = cfg.dist_2_schnapp_cm;  // max distance
    rawDg = cfg.deg_2_schnapp_deg;  // max angle
    document.getElementById('dist-source').textContent = '(far away override)';
  } else {
    document.getElementById('dist-source').textContent = '(computed)';
  }

  // Initialise EMA on first call
  if (_emaDist === null) { _emaDist = rawDc; _emaDeg = rawDg; }
  _emaDist = _emaDist + EMA_ALPHA * (rawDc - _emaDist);
  _emaDeg  = _emaDeg  + EMA_ALPHA * (rawDg - _emaDeg);

  const dc = _emaDist, dg = _emaDeg;

  document.getElementById('d-dist').textContent = dc.toFixed(2);
  document.getElementById('d-rot').textContent  = dg.toFixed(1);

  const cd = Math.max(0, Math.min(1, 1 - dc / cfg.schnapp_bar_max_cm)) * 100;
  const cr = Math.max(0, Math.min(1, 1 - dg / cfg.schnapp_bar_max_deg)) * 100;

  const bd = document.getElementById('b-dist');
  bd.style.width      = cd.toFixed(1) + '%';
  bd.style.background = cd > 80 ? '#4f4' : cd > 50 ? '#fa0' : '#f64';
  document.getElementById('l-dist').textContent = cd.toFixed(0) + '% close';

  const br = document.getElementById('b-rot');
  br.style.width      = cr.toFixed(1) + '%';
  br.style.background = cr > 80 ? '#4f4' : cr > 50 ? '#48f' : '#f64';
  document.getElementById('l-rot').textContent  = cr.toFixed(0) + '% close';
}

function renderTraj(traj) {
  if (!traj || traj.length < 2) return;
  const x  = traj.map(p => p.pos[0]*1000);
  const y  = traj.map(p => p.pos[1]*1000);
  const z  = traj.map(p => p.pos[2]*1000);
  const sp = traj.map(p => p.speed*1000);
  function rng(v){const mn=Math.min(...v),mx=Math.max(...v),c=(mn+mx)/2,s=mx-mn;
    return s<30?[c-15,c+15]:[mn,mx];}
  const bg = dark?'#1b1b2e':'#f4f4f4', fg=dark?'#888':'#333';
  Plotly.react('traj',[{type:'scatter3d',mode:'lines+markers',x,y,z,
    marker:{size:3,color:sp,colorscale:'Hot',cmin:0,cmax:Math.max(...sp)||1},
    line:{color:sp,colorscale:'Hot',width:4}}],
    {margin:{l:0,r:0,t:20,b:0},paper_bgcolor:bg,
     scene:{xaxis:{title:'X mm',color:fg,range:rng(x)},
            yaxis:{title:'Y mm',color:fg,range:rng(y)},
            zaxis:{title:'Z mm',color:fg,range:rng(z)},bgcolor:bg},
     font:{color:fg}},{responsive:true});
}

function renderPred(b64, info) {
  if (b64) swapImg(document.getElementById('pred-img'), b64);
  if (info) document.getElementById('pred-info').textContent = info;
}

function renderTiming(t) {
  if (!t) return;
  document.getElementById('timing-display').textContent =
    `enc=${t.enc_ms.toFixed(0)}ms dif=${t.dif_ms.toFixed(0)}ms `
    + `pub=${t.pub_ms.toFixed(1)}ms obs_gap=${t.obs_gap_ms.toFixed(0)}ms`;
}

function renderConfig(c) {
  if (!c) return;

  // Robot tag: show PENDING state during delay window
  let robotLabel = 'OFF';
  let robotCls   = 'off';
  if (c.robot_publish_enabled) {
    if (c.stopped_by_cut) {
      robotLabel = 'ON(CUT)'; robotCls = 'off';
    } else if (c.cut_pending_ms > 0) {
      robotLabel = 'PENDING'; robotCls = 'frozen';
    } else {
      robotLabel = 'ON'; robotCls = 'on';
    }
  }
  const rt = document.getElementById('robot-tag');
  rt.textContent = robotLabel;
  rt.className   = 'tag ' + robotCls;

  setTag('autostop-tag', c.auto_stop_on_cut);
  setTag('freeze-tag',   c.freeze_depth, 'FROZEN', 'OFF', c.freeze_depth ? 'frozen' : 'off');
  setTag('zero-tag',     c.zero_depth,   'ZERO',   'OFF', c.zero_depth   ? 'frozen' : 'off');

  // Pending countdown banner
  const pend = document.getElementById('cut-pending');
  if (c.cut_pending_ms > 0 && c.robot_publish_enabled) {
    pend.style.display = 'block';
    document.getElementById('cut-pending-ms').textContent =
      c.cut_pending_ms.toFixed(0);
  } else {
    pend.style.display = 'none';
  }
  
  document.getElementById('d-depth-threshold').textContent = c.depth_far_away_threshold_cm.toFixed(1);
}

function setTag(id, on, labelOn='ON', labelOff='OFF', cls=null) {
  const el = document.getElementById(id);
  el.textContent = on ? labelOn : labelOff;
  el.className   = 'tag ' + (cls || (on ? 'on' : 'off'));
}

function toggleTheme() {
  dark = !dark;
  document.body.setAttribute('data-theme', dark ? 'dark' : '');
  document.querySelector('.btn-theme').textContent = dark ? '☀️ Light' : '🌙 Dark';
}
function toggleRobot()       { socket.emit('toggle_robot'); }
function toggleAutoStop()    { socket.emit('toggle_auto_stop_on_cut'); }
function toggleFreezeDepth() { socket.emit('toggle_freeze_depth'); }
function toggleZeroDepth()   { socket.emit('toggle_zero_depth'); }

function applySettings() {
  const idx = document.getElementById('s-pidx').value
    .split(',').map(s => parseInt(s.trim())-1).filter(n => !isNaN(n));
  socket.emit('update_settings', {
    obs_interval_ms:          parseFloat(document.getElementById('s-obs').value),
    min_inference_ms:         parseFloat(document.getElementById('s-inf').value),
    sync_tolerance_ms:        parseFloat(document.getElementById('s-sync').value),
    schnapp_cut_threshold_cm: parseFloat(document.getElementById('s-cut').value),
    schnapp_avg_window:       parseInt(document.getElementById('s-avg').value),
    schnapp_bar_max_cm:       parseFloat(document.getElementById('s-bmcm').value),
    schnapp_bar_max_deg:      parseFloat(document.getElementById('s-bmdeg').value),
    coord_arrow_length_px:    parseInt(document.getElementById('s-alen').value),
    coord_arrow_width_px:     parseInt(document.getElementById('s-awid').value),
    pred_display_indices:     idx,
    cut_stop_delay_ms:        parseFloat(document.getElementById('s-cdelay').value), 
    pose_limiting_index:      parseInt(document.getElementById('s-pose-lim').value),
    depth_far_away_threshold_cm: parseFloat(document.getElementById('s-far-depth').value),
    min_depth_filter_cm: parseFloat(document.getElementById('s-min-depth').value),
  });
}
</script>
</body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Flask / SocketIO server
# ─────────────────────────────────────────────────────────────────────────────
def create_dashboard_server(state: DashboardState, port: int = 5000):
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dp_mv_dash'
    socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

    @app.route('/')
    def index():
        return render_template_string(HTML)

    @socketio.on('toggle_robot')
    def _toggle_robot():
        state.config.robot_publish_enabled = not state.config.robot_publish_enabled
        state.stopped_by_cut = False

    @socketio.on('toggle_auto_stop_on_cut')
    def _toggle_auto():
        state.config.auto_stop_on_cut = not state.config.auto_stop_on_cut

    @socketio.on('toggle_freeze_depth')
    def _toggle_freeze():
        state.config.freeze_depth = not state.config.freeze_depth
        if not state.config.freeze_depth:
            state.frozen_depth = None

    @socketio.on('toggle_zero_depth')
    def _toggle_zero():
        state.config.zero_depth = not state.config.zero_depth

    @socketio.on('update_settings')
    def _settings(data):
        for k, v in data.items():
            if hasattr(state.config, k):
                setattr(state.config, k, v)

    return app, socketio


# ─────────────────────────────────────────────────────────────────────────────
#  Emitter
# ─────────────────────────────────────────────────────────────────────────────

class DashboardEmitter:
    def __init__(self, state: DashboardState, socketio: SocketIO,
                 emit_hz: float = 10.0, depth_max: float = 0.24):
        self.state    = state
        self.socketio = socketio
        self.period   = 1.0 / emit_hz
        self.dmax     = depth_max
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            t0 = time.perf_counter()
            try:
                self._emit()
            except Exception as e:
                print(f'[Dashboard emitter] {e}')
            time.sleep(max(0, self.period - (time.perf_counter() - t0)))

    def _render_obs_frames(self, frames: List[dict], view_key: str) -> List[dict]:
        out = []
        for f in frames:
            rgb = f.get(f'{view_key}_rgb')
            dep = f.get(f'{view_key}_depth')

            rgb_b64 = _to_b64(rgb.astype(np.uint8)) if rgb is not None else ''

            mean_depth_mm = 0.0
            if dep is not None:
                colored = depth_to_magma(dep, self.dmax)
                dep_b64 = _to_b64(colored)
                valid = dep > 0
                if valid.any():
                    mean_depth_mm = float(dep[valid].mean()) * 1e3
            else:
                dep_b64 = rgb_b64

            out.append({
                'rgb_b64':       rgb_b64,
                'dep_b64':       dep_b64,
                'time_past_ms':  f.get(f'{view_key}_time_past_ms', 0),
                'disp_mm':       f.get('disp_mm', 0),
                'rot_deg':       f.get('rot_deg', 0),
                'mean_depth_mm': mean_depth_mm,
            })
        return out


    def _emit(self):
        snap   = self.state.get_snapshot()
        frames = snap['obs_frames']

        obs_base = self._render_obs_frames(frames, 'base')
        obs_up   = self._render_obs_frames(frames, 'up')

        inter = [{'delta_ms':    f.get('base_up_delta_ms', 0.0),
                'has_real_up': f.get('has_real_up', False)}
                for f in frames]

        pred_b64, pred_info = '', ''
        pred = snap['prediction']
        if pred and pred.get('current_image') is not None:
            pa   = pred.get('pred_abs_poses', [])
            fl_p = pred.get('ref_flange_pos')
            fl_q = pred.get('ref_flange_quat')
            cfg  = snap['config']
            if fl_p is not None and len(pa) > 0:
                ov = render_prediction_overlay(
                    pred['current_image'].astype(np.uint8), pa,
                    cfg['pred_display_indices'], fl_p, fl_q,
                    arrow_length_m=cfg['coord_arrow_length_px'] * 0.0004,
                    arrow_width_px=cfg['coord_arrow_width_px'],
                    crop_region=pred.get('crop_region', (0, 0, 1280, 720)),
                    ref_tcp_pos=pred.get('ref_pos'),
                    ref_tcp_quat=pred.get('ref_quat'),
                )
                buf = io.BytesIO()
                PILImage.fromarray(ov).save(buf, format='PNG', optimize=True)
                pred_b64 = base64.b64encode(buf.getvalue()).decode()

            dc = snap['schnapp_avg_dist_pct'] * cfg['dist_2_schnapp_cm']
            dr = snap['schnapp_avg_deg_pct']  * cfg['deg_2_schnapp_deg']
            pred_info = (f"schnapp dist={dc:.2f}cm rot={dr:.1f}° | "
                        f"pred_len={pred.get('pred_length_mm', 0):.1f}mm "
                        f"rot={pred.get('pred_rot_deg', 0):.1f}°")

        traj = snap['traj']
        step = max(1, len(traj) // 100)
        traj_ds = [{'pos': t['pos'].tolist(), 'speed': t['speed']}
                for t in traj[::step]]

        self.socketio.emit('update', {
            'obs_base':             obs_base,
            'obs_up':               obs_up,
            'inter_timing':         inter,
            'traj':                 traj_ds,
            'pred_img':             pred_b64,
            'pred_info':            pred_info,
            'schnapp_avg_dist_pct': snap['schnapp_avg_dist_pct'],
            'schnapp_avg_deg_pct':  snap['schnapp_avg_deg_pct'],
            'cut_now':              snap['cut_now'],
            'far_away':             snap['far_away'],             
            'min_depth_cm':         snap['min_depth_cm'],         
            'timing':               snap['timing'],
            'config':               snap['config'],
        })

# ─────────────────────────────────────────────────────────────────────────────
#  DashboardIntegration mixin
# ─────────────────────────────────────────────────────────────────────────────

class DashboardIntegration:
    def init_dashboard(self, port: int = 5000):
        self.dashboard_state = DashboardState()
        if hasattr(self, 'config'):
            if hasattr(self.config, 'dist_2_schnapp_treshold'):
                self.dashboard_state.config.dist_2_schnapp_cm = \
                    self.config.dist_2_schnapp_treshold * 100.0
            if hasattr(self.config, 'deg_2_schnapp_treshold'):
                self.dashboard_state.config.deg_2_schnapp_deg = \
                    float(self.config.deg_2_schnapp_treshold)

        app, socketio = create_dashboard_server(self.dashboard_state, port)
        self.dashboard_socketio = socketio
        depth_max = getattr(getattr(self, 'config', None), 'depth_cut_treshold', 0.24)
        self.dashboard_emitter = DashboardEmitter(
            self.dashboard_state, socketio, emit_hz=10.0, depth_max=depth_max)

        self._dash_thread = threading.Thread(
            target=lambda: socketio.run(app, host='0.0.0.0', port=port,
                                        allow_unsafe_werkzeug=True),
            daemon=True)
        self._dash_thread.start()
        self.dashboard_emitter.start()

        self._prev_tcp_pos: Optional[np.ndarray] = None
        self._prev_tcp_stamp: Optional[float]    = None
        try:
            self.get_logger().info(f'Dashboard → http://0.0.0.0:{port}')
        except Exception:
            print(f'Dashboard → http://0.0.0.0:{port}')

    def apply_depth_override(self, depth):
        if not hasattr(self, 'dashboard_state'):
            return depth
        cfg = self.dashboard_state.config
        if cfg.zero_depth:
            return np.zeros_like(depth) if depth is not None else None
        if cfg.freeze_depth:
            if self.dashboard_state.frozen_depth is None and depth is not None:
                self.dashboard_state.frozen_depth = depth.copy()
            return self.dashboard_state.frozen_depth
        return depth

    def update_dashboard_obs(self, frames):
        ref       = frames[-1]
        ref_pos   = ref.tcp_pos
        ref_stamp = ref.base_img_stamp_ms

        obs_data = []
        for f in frames:
            disp_mm = float(np.linalg.norm(f.tcp_pos - ref_pos)) * 1e3
            q1 = f.tcp_quat   / (np.linalg.norm(f.tcp_quat)   + 1e-12)
            q2 = ref.tcp_quat / (np.linalg.norm(ref.tcp_quat) + 1e-12)
            rot_deg = float(2.0 * np.arccos(
                np.clip(abs(float(np.dot(q1, q2))), -1.0, 1.0)) * 180.0 / np.pi)
            obs_data.append({
                'base_rgb':          f.base_image,
                'base_depth':        f.base_depth,
                'base_time_past_ms': float(ref_stamp - f.base_img_stamp_ms),
                'up_rgb':            f.up_image,
                'up_depth':          f.up_depth,
                'up_time_past_ms':   float(ref_stamp - f.up_img_stamp_ms),
                'base_up_delta_ms':  abs(f.base_img_stamp_ms - f.up_img_stamp_ms),
                'has_real_up':       f.has_real_up,
                'disp_mm':           disp_mm,
                'rot_deg':           rot_deg,
            })
        self.dashboard_state.update_obs(obs_data)

        now   = ref.stamp_sec
        speed = 0.0
        if self._prev_tcp_pos is not None and self._prev_tcp_stamp is not None:
            dt = now - self._prev_tcp_stamp
            if dt > 1e-6:
                speed = float(np.linalg.norm(ref_pos - self._prev_tcp_pos)) / dt
        self._prev_tcp_pos   = ref_pos.copy()
        self._prev_tcp_stamp = now
        self.dashboard_state.add_traj_point(now, ref_pos, ref.tcp_quat, speed)


    def update_dashboard_prediction(self, rel_actions, ref_pos, ref_quat,
                                    current_image, enc_ms, dif_ms, pub_ms,
                                    obs_gap_ms, far_away: bool = False,           
                                    min_depth_cm: float = 0.0):                    
        pred_abs = []
        for i in range(len(rel_actions)):
            pa, qa = tcp_pose_from_relative(
                np.concatenate([rel_actions[i, :3], rel_actions[i, 3:6]]),
                ref_pos=ref_pos, ref_quat_xyzw=ref_quat)
            pred_abs.append((pa.astype(np.float64), qa.astype(np.float64)))

        schnapp_dist = float(rel_actions[0, 6])
        schnapp_deg  = float(rel_actions[0, 7])

        pred_len_mm = pred_rot_deg = 0.0
        if pred_abs:
            pred_len_mm = float(np.linalg.norm(pred_abs[-1][0] - ref_pos)) * 1e3
            dot = float(np.clip(abs(np.dot(pred_abs[-1][1], ref_quat)), -1.0, 1.0))
            pred_rot_deg = float(2.0 * np.arccos(dot) * 180.0 / np.pi)

        ref_fl_pos, ref_fl_quat = self._tcp_to_flange(ref_pos, ref_quat)
        self.dashboard_state.update_prediction({
            'ref_pos':         ref_pos,
            'ref_quat':        ref_quat,
            'ref_flange_pos':  ref_fl_pos.astype(np.float64),
            'ref_flange_quat': ref_fl_quat.astype(np.float64),
            'current_image':   current_image,
            'pred_abs_poses':  pred_abs,
            'pred_length_mm':  pred_len_mm,
            'pred_rot_deg':    pred_rot_deg,
            'crop_region': tuple(
                self.config.image_static_crop
                if hasattr(self, 'config') else (0, 0, 1280, 720)),
        })

        now = time.time()
        self.dashboard_state.add_schnapp(now, schnapp_dist, schnapp_deg)
        self.dashboard_state.last_enc_ms     = enc_ms
        self.dashboard_state.last_dif_ms     = dif_ms
        self.dashboard_state.last_pub_ms     = pub_ms
        self.dashboard_state.last_obs_gap_ms = obs_gap_ms
        
        self.dashboard_state.far_away = far_away
        self.dashboard_state.min_depth_cm = min_depth_cm


    def get_dashboard_robot_enabled(self) -> bool:
        if not hasattr(self, 'dashboard_state'):
            return True

        state = self.dashboard_state
        cfg   = state.config

        if cfg.auto_stop_on_cut:
            cut = state.is_cut_now()

            if cut and cfg.robot_publish_enabled:
                now = time.time()
                # Start the delay timer on first detection
                if state._cut_first_seen is None:
                    state._cut_first_seen = now

                elapsed_ms = (now - state._cut_first_seen) * 1e3
                if elapsed_ms < cfg.cut_stop_delay_ms:
                    # Still within grace window — keep publishing
                    return True

                # Delay expired → block publishing
                cfg.robot_publish_enabled = False
                state.stopped_by_cut      = True
                state._cut_first_seen     = None
                return False

            if not cut:
                # Cut cleared — reset delay timer unconditionally
                state._cut_first_seen = None
                if state.stopped_by_cut:
                    cfg.robot_publish_enabled = True
                    state.stopped_by_cut      = False

        return cfg.robot_publish_enabled
    
    def get_dashboard_settings(self) -> DashboardConfig:
        if not hasattr(self, 'dashboard_state'):
            return DashboardConfig()
        return self.dashboard_state.config


# ─────────────────────────────────────────────────────────────────────────────
#  Standalone demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    state = DashboardState()
    app, sio = create_dashboard_server(state, args.port)
    emitter  = DashboardEmitter(state, sio, emit_hz=5.0)

    def _dummy():
        t = 0
        while True:
            time.sleep(0.2)
            t += 0.2
            H, W = 150, 200
            frames = []
            for i in range(2):
                frames.append({
                    'base_rgb':          np.random.randint(0, 255, (H, W, 3), np.uint8),
                    'base_depth':        np.random.rand(H, W).astype(np.float32) * 0.2,
                    'base_time_past_ms': (1-i)*100.0,
                    'up_rgb':            np.random.randint(50, 200, (H, W, 3), np.uint8),
                    'up_depth':          np.random.rand(H, W).astype(np.float32) * 0.18,
                    'up_time_past_ms':   (1-i)*100.0 + np.random.rand()*20,
                    'base_up_delta_ms':  np.random.rand() * 30,
                    'has_real_up':       True,
                    'disp_mm':           np.random.rand() * 5,
                    'rot_deg':           np.random.rand() * 2,
                })
            state.update_obs(frames)
            pos = np.array([0.5+0.01*np.sin(t), 0.1+0.01*np.cos(t), 0.3])
            state.add_traj_point(time.time(), pos, np.array([0,0,0,1.]),
                                 np.random.rand()*0.02)
            state.add_schnapp(time.time(),
                              max(0, 0.5-t*0.004), max(0, 0.3-t*0.003))
            state.last_enc_ms = 15.0
            state.last_dif_ms = 45.0

    threading.Thread(target=_dummy, daemon=True).start()
    emitter.start()
    print(f'Dashboard at http://localhost:{args.port}')
    sio.run(app, host='0.0.0.0', port=args.port, allow_unsafe_werkzeug=True)