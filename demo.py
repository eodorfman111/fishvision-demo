# FishVision Demo
# Underwater marine-species detection — portfolio showcase

from __future__ import annotations

import base64
import io
import os
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# cv2 · torch · ultralytics · reportlab · huggingface_hub imported lazily
# (inside functions) so the showcase page loads on any Python/system config.

# ── CONFIG ────────────────────────────────────────────────────────────────────
APP_NAME       = "FishVision"
MODEL_PATH     = Path(__file__).parent / "models" / "best_v1.04.pt"
STATIC_DIR     = Path(__file__).parent / "static"
PREVIEW_VIDEO  = STATIC_DIR / "preview.mp4"
FISH_COLOR_BGR = (150, 204, 0)
FISH_COLOR_HEX = "#00CC96"
MAX_UPLOAD_MB  = 200
TARGET_CLASS   = "fish"

CONTACT_EMAIL = "leodorfman1@gmail.com"
GITHUB_URL    = "https://github.com/eodorfman111"
LINKEDIN_URL  = "https://www.linkedin.com/in/leo-dorfman"

# ── DETECTION GALLERY metadata ────────────────────────────────────────────────
GALLERY = [
    {
        "file":    "detect_crab.png",
        "species": "CRAB · STARFISH",
        "caption": "Benthic invertebrate detection",
    },
    {
        "file":    "detect_reef.png",
        "species": "SEABREAM · MANGROVE SNAPPER · SEA URCHIN",
        "caption": "Mixed reef species, single frame",
    },
    {
        "file":    "detect_eel.png",
        "species": "EEL · STARFISH",
        "caption": "Elongated & cryptic species",
    },
    {
        "file":    "detect_cuttlefish.png",
        "species": "CUTTLEFISH · SEABREAM",
        "caption": "Cephalopod detection alongside schooling fish",
    },
]

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&family=Share+Tech+Mono&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #020d1a !important;
    color: #c8e6f5 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, header[data-testid="stHeader"], .stApp > header { display: none !important; }

[data-testid="stSidebar"] {
    background: #02111f !important;
    border-right: 1px solid #0a3a5c;
}
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: #00e5ff !important;
    letter-spacing: 2px;
}
h1 { font-size: 1.8rem !important; }

.stButton > button {
    background: linear-gradient(135deg, #00b4d8, #0077b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px;
    padding: 0.65rem 2rem !important;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 24px rgba(0,180,216,0.5);
}

.glass-card {
    background: rgba(0,180,216,0.07);
    border: 1px solid rgba(0,180,216,0.22);
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}
.kpi-title { font-size: 0.68rem; color: #5ebbdc; letter-spacing: 2px; text-transform: uppercase; }
.kpi-value { font-size: 2.2rem; font-weight: 700; color: #00e5ff; font-family: 'Orbitron', sans-serif; line-height: 1.1; }
.kpi-sub   { font-size: 0.68rem; color: #5ebbdc; margin-top: 4px; }

.tag {
    display: inline-block;
    background: rgba(0,229,255,0.10);
    border: 1px solid rgba(0,229,255,0.28);
    border-radius: 4px;
    padding: 3px 12px;
    font-size: 0.72rem;
    color: #00e5ff;
    margin: 3px;
    font-family: 'Share Tech Mono', monospace;
}
.demo-badge {
    display: inline-block;
    background: rgba(0,229,255,0.12);
    border: 1px solid rgba(0,229,255,0.4);
    border-radius: 20px;
    padding: 3px 16px;
    font-size: 0.68rem;
    color: #00e5ff;
    letter-spacing: 3px;
    font-family: 'Orbitron', sans-serif;
    margin-bottom: 0.5rem;
}
.cta-card {
    background: linear-gradient(135deg, rgba(0,77,115,0.5), rgba(0,180,216,0.12));
    border: 1px solid rgba(0,229,255,0.35);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-top: 0.5rem;
}
.section-label {
    font-family: Orbitron, sans-serif;
    color: #00e5ff;
    font-size: 0.82rem;
    letter-spacing: 2px;
    margin: 0.4rem 0 1rem;
    padding-top: 0.2rem;
}
.sidebar-bio {
    font-size: 0.72rem;
    color: #5ebbdc;
    line-height: 1.95;
}
.sidebar-bio b { color: #00e5ff; }
.sidebar-bio a { color: #00b4d8; text-decoration: none; }

[data-testid="stFileUploader"] {
    border: 1px dashed rgba(0,180,216,0.35) !important;
    border-radius: 10px !important;
    background: rgba(0,180,216,0.04) !important;
}
footer { visibility: hidden; }
</style>
"""

# ── HELPERS ───────────────────────────────────────────────────────────────────
def hms(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(2,13,26,0)",
        plot_bgcolor="rgba(2,13,26,0.6)",
        font=dict(color="#c8e6f5", family="Share Tech Mono"),
        xaxis=dict(gridcolor="#0a2a40", zerolinecolor="#0a2a40"),
        yaxis=dict(gridcolor="#0a2a40", zerolinecolor="#0a2a40"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ── ASSET LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_gallery_images() -> dict[str, str]:
    """Load gallery PNGs + hero video as base64 strings (cached once per process)."""
    result: dict[str, str] = {}
    # Hero video
    if PREVIEW_VIDEO.exists():
        with open(PREVIEW_VIDEO, "rb") as f:
            result["video"] = base64.b64encode(f.read()).decode()
    # Detection images
    for item in GALLERY:
        path = STATIC_DIR / item["file"]
        if path.exists():
            key = item["file"].replace(".png", "")
            with open(path, "rb") as f:
                result[key] = base64.b64encode(f.read()).decode()
    return result


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    import torch
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH), task="detect")
    try:
        model.model.half = False
    except Exception:
        pass
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model


# ── INFERENCE ─────────────────────────────────────────────────────────────────
def run_inference(
    video_path: str,
    model: Any,
    sample_every: int,
    conf: float,
    topk: int,
    progress_bar,
    status_text,
) -> tuple[pd.DataFrame, list[dict], dict]:
    import cv2
    import torch

    cap         = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_fr    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration    = total_fr / fps
    iou         = 0.45
    rows: list[dict] = []
    topk_candidates: list[dict] = []
    frame_idx   = 0
    processed   = 0
    sample_step = max(1, int(fps * sample_every))

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        ts  = frame_idx / fps
        pct = frame_idx / max(total_fr, 1)
        progress_bar.progress(min(pct, 1.0))
        status_text.markdown(
            f"<span style='color:#00e5ff;font-size:0.8rem'>"
            f"Analyzing {hms(ts)} / {hms(duration)}</span>",
            unsafe_allow_html=True,
        )

        results    = model.predict(frame, conf=conf, iou=iou, verbose=False)
        boxes      = results[0].boxes if results else None
        fish_count = 0
        annotated  = frame.copy()

        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id   = int(box.cls[0])
                cls_name = model.names.get(cls_id, "").lower()
                if cls_name == TARGET_CLASS or cls_name in TARGET_CLASS:
                    fish_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = float(box.conf[0])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), FISH_COLOR_BGR, 2)
                    cv2.putText(annotated, f"fish {score:.2f}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, FISH_COLOR_BGR, 2)

        rows.append({
            "frame_index":   frame_idx,
            "timestamp_s":   ts,
            "timestamp_hms": hms(ts),
            "time_min":      round(ts / 60, 3),
            "fish_count":    fish_count,
        })

        if fish_count > 0:
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            topk_candidates.append({
                "frame_index":   frame_idx,
                "timestamp_s":   ts,
                "timestamp_hms": hms(ts),
                "fish_count":    fish_count,
                "jpeg":          buf.tobytes(),
            })

        frame_idx += sample_step
        processed += 1

    cap.release()
    progress_bar.progress(1.0)
    status_text.markdown(
        "<span style='color:#00CC96;font-size:0.8rem'>✓ Analysis complete</span>",
        unsafe_allow_html=True,
    )

    df          = pd.DataFrame(rows)
    topk_frames = sorted(topk_candidates, key=lambda x: x["fish_count"], reverse=True)[:topk]
    stats = {
        "duration_s":       duration,
        "frames_sampled":   processed,
        "peak_count":       int(df["fish_count"].max()) if not df.empty else 0,
        "peak_ts":          df.loc[df["fish_count"].idxmax(), "timestamp_hms"]
                            if not df.empty and df["fish_count"].max() > 0 else "—",
        "total_detections": int(df["fish_count"].sum()),
        "detection_frames": int((df["fish_count"] > 0).sum()),
        "device":           "cuda" if torch.cuda.is_available() else "cpu",
    }
    return df, topk_frames, stats


def build_charts(df: pd.DataFrame):
    df_det   = df[df["fish_count"] > 0]
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=df["time_min"], y=df["fish_count"],
        mode="lines", fill="tozeroy",
        line=dict(color=FISH_COLOR_HEX, width=2),
        fillcolor="rgba(0,204,150,0.15)",
        name="Fish count",
    ))
    fig_line.update_layout(title="Fish Count Over Time",
                           xaxis_title="Time (min)", yaxis_title="Count")
    apply_theme(fig_line)

    fig_hist = go.Figure(go.Histogram(
        x=df_det["fish_count"] if not df_det.empty else [],
        nbinsx=20, marker_color=FISH_COLOR_HEX, opacity=0.85,
    ))
    fig_hist.update_layout(title="Detection Count Distribution",
                           xaxis_title="Fish count", yaxis_title="Frames")
    apply_theme(fig_hist)
    return fig_line, fig_hist


def make_pdf(stats: dict, topk_frames: list[dict], video_name: str, conf: float) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as rlcanvas

    buf  = io.BytesIO()
    c    = rlcanvas.Canvas(buf, pagesize=A4)
    W, H = A4

    c.setFillColorRGB(0.008, 0.051, 0.102); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColorRGB(0, 0.447, 0.722);     c.rect(0, H - 90, W, 90, fill=1, stroke=0)
    c.setFillColorRGB(0, 0.706, 0.933);     c.rect(0, H - 90, int(W * 0.42), 90, fill=1, stroke=0)

    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 24)
    c.drawString(30, H - 46, "FISHVISION")
    c.setFillColorRGB(0.85, 0.95, 1); c.setFont("Helvetica", 10)
    c.drawString(30, H - 62, "Underwater Fish Detection Report")
    c.setFillColorRGB(0.70, 0.88, 0.97); c.setFont("Helvetica", 8)
    c.drawString(30, H - 76,
        f"YOLO26 Custom Model  ·  Confidence: {conf:.0%}  ·  Device: {stats['device'].upper()}")
    c.setFillColorRGB(0.85, 0.95, 1); c.setFont("Helvetica", 9)
    c.drawRightString(W - 30, H - 44, f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
    vname = video_name[:50] + ("..." if len(video_name) > 50 else "")
    c.drawRightString(W - 30, H - 58, f"Source: {vname}")

    y = H - 112
    c.setFillColorRGB(0, 0.898, 1); c.setFont("Helvetica-Bold", 10)
    c.drawString(30, y, "EXECUTIVE SUMMARY")
    c.setStrokeColorRGB(0, 0.898, 1); c.setLineWidth(0.4)
    c.line(30, y - 5, W - 30, y - 5); y -= 18

    kpis = [
        ("Video Duration",   hms(stats["duration_s"])),
        ("Peak Fish Count",  str(stats["peak_count"])),
        ("Peak Timestamp",   stats["peak_ts"]),
        ("Frames Sampled",   str(stats["frames_sampled"])),
        ("Detection Frames", str(stats["detection_frames"])),
        ("Total Detections", str(stats["total_detections"])),
    ]
    col_w = (W - 60) / 3
    row_h = 42
    for i, (label, val) in enumerate(kpis):
        cx = 30 + (i % 3) * col_w
        cy = y - (i // 3) * row_h
        c.setFillColorRGB(0.004, 0.18, 0.30)
        c.roundRect(cx, cy - 34, col_w - 8, 38, 5, fill=1, stroke=0)
        c.setFillColorRGB(0.47, 0.73, 0.87); c.setFont("Helvetica", 7)
        c.drawString(cx + 8, cy - 12, label.upper())
        c.setFillColorRGB(0, 0.898, 1); c.setFont("Helvetica-Bold", 13)
        c.drawString(cx + 8, cy - 28, val)

    y -= ((len(kpis) + 2) // 3) * row_h + 18

    if topk_frames:
        c.setFillColorRGB(0, 0.898, 1); c.setFont("Helvetica-Bold", 10)
        c.drawString(30, y, f"TOP DETECTION FRAMES  ({len(topk_frames)} shown)")
        c.setStrokeColorRGB(0, 0.898, 1); c.setLineWidth(0.4)
        c.line(30, y - 5, W - 30, y - 5); y -= 16

        cols  = 3
        img_w = (W - 60 - (cols - 1) * 8) / cols
        img_h = img_w * 0.62
        for i, frame in enumerate(topk_frames[:6]):
            col_ = i % cols; row_ = i // cols
            fx   = 30 + col_ * (img_w + 8)
            fy   = y - (row_ + 1) * (img_h + 24)
            if fy < 50:
                break
            try:
                pil_img = ImageReader(io.BytesIO(frame["jpeg"]))
                c.drawImage(pil_img, fx, fy, width=img_w, height=img_h,
                            preserveAspectRatio=True)
                c.setFillColorRGB(0.004, 0.18, 0.30)
                c.rect(fx, fy - 16, img_w, 16, fill=1, stroke=0)
                c.setFillColorRGB(0.784, 0.902, 0.957); c.setFont("Helvetica", 7)
                c.drawString(fx + 4, fy - 10,
                    f"  {frame['timestamp_hms']}  ·  {frame['fish_count']} fish")
            except Exception:
                pass

    c.setFillColorRGB(0, 0.29, 0.51); c.rect(0, 0, W, 38, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 9)
    c.drawString(30, 24, "FISHVISION  ·  Prepared by Leo Dorfman")
    c.setFillColorRGB(0.75, 0.9, 1); c.setFont("Helvetica", 8)
    c.drawString(30, 12,
        "leodorfman1@gmail.com  |  linkedin.com/in/leo-dorfman  |  github.com/eodorfman111")
    c.setFillColorRGB(0.6, 0.85, 1); c.setFont("Helvetica", 8)
    c.drawRightString(W - 30, 18, "Confidential")
    c.save()
    return buf.getvalue()


# ── UI COMPONENTS ─────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("""
<div class='sidebar-bio'>
<b style='font-family:Orbitron,sans-serif;font-size:0.88rem;letter-spacing:2px'>FishVision</b><br>
<span style='font-size:0.69rem;color:#3d7fa0'>Marine species detection pipeline</span>
<br><br>
<b>Built by</b><br>
Leo Dorfman<br>
CS @ University of Florida<br><br>
<a href='mailto:leodorfman1@gmail.com'>✉ leodorfman1@gmail.com</a><br>
<a href='https://www.linkedin.com/in/leo-dorfman' target='_blank'>⬡ LinkedIn</a><br>
<a href='https://github.com/eodorfman111' target='_blank'>⌥ GitHub</a><br><br>
<b>Tech Stack</b><br>
· YOLO26 (ultralytics)<br>
· OpenCV · PyTorch<br>
· Streamlit · Plotly<br>
· Python · ReportLab<br><br>
<b>Model Stats</b><br>
· v1.04 · custom-trained<br>
· 10+ marine species<br>
· 95%+ confidence scores<br>
· Real underwater footage
</div>
""", unsafe_allow_html=True)


def render_hero() -> None:
    """Full-width looping detection video, base64-embedded for reliable cloud delivery."""
    assets = _load_gallery_images()
    if "video" not in assets:
        st.warning("Hero video not found — place preview.mp4 in the static/ folder.")
        return
    b64 = assets["video"]
    st.markdown(
        "<div style='"
        "position:relative;border-radius:14px;overflow:hidden;"
        "box-shadow:0 8px 48px rgba(0,229,255,0.14);margin-bottom:0.3rem'>"
        f"<video autoplay loop muted playsinline "
        f"style='width:100%;display:block;max-height:480px;"
        f"object-fit:cover;object-position:center bottom'>"
        f"<source src='data:video/mp4;base64,{b64}' type='video/mp4'>"
        f"</video>"
        "<div style='"
        "position:absolute;bottom:0;left:0;right:0;"
        "background:linear-gradient(to top,rgba(2,13,26,0.93) 0%,"
        "rgba(2,13,26,0.45) 55%,transparent 100%);"
        "padding:1.8rem 2rem'>"
        "<div style='"
        "font-family:Orbitron,sans-serif;color:#00e5ff;"
        "font-size:1.05rem;letter-spacing:3px;margin-bottom:0.5rem'>"
        "REAL-TIME MARINE SPECIES DETECTION"
        "</div>"
        "<div style='color:#c8e6f5;font-size:0.8rem;letter-spacing:1px'>"
        "YOLO26 &nbsp;·&nbsp; custom-trained &nbsp;·&nbsp; "
        "multi-species &nbsp;·&nbsp; real underwater footage"
        "</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_detection_gallery() -> None:
    """2×2 grid of detection screenshots with overlay species labels."""
    st.markdown(
        "<div class='section-label'>DETECTION SHOWCASE</div>",
        unsafe_allow_html=True,
    )
    images = _load_gallery_images()
    col_a, col_b = st.columns(2, gap="medium")
    cols = [col_a, col_b]

    for i, item in enumerate(GALLERY):
        key = item["file"].replace(".png", "")
        with cols[i % 2]:
            if key in images:
                b64 = images[key]
                st.markdown(
                    "<div style='"
                    "position:relative;border-radius:10px;overflow:hidden;"
                    "margin-bottom:1.1rem;"
                    "box-shadow:0 6px 28px rgba(0,0,0,0.55)'>"
                    f"<img src='data:image/png;base64,{b64}' "
                    "style='width:100%;display:block'>"
                    "<div style='"
                    "position:absolute;bottom:0;left:0;right:0;"
                    "background:linear-gradient(to top,"
                    "rgba(2,13,26,0.97) 0%,rgba(2,13,26,0.35) 65%,transparent 100%);"
                    "padding:0.75rem 1rem'>"
                    "<div style='"
                    "color:#00e5ff;font-size:0.68rem;"
                    "font-family:Orbitron,sans-serif;"
                    f"letter-spacing:1.5px;margin-bottom:3px'>{item['species']}</div>"
                    "<div style='color:#7ecce8;font-size:0.66rem'>"
                    f"{item['caption']}</div>"
                    "</div></div>",
                    unsafe_allow_html=True,
                )


def render_pipeline_overview() -> None:
    st.markdown(
        "<div class='section-label'>PIPELINE CAPABILITIES</div>",
        unsafe_allow_html=True,
    )

    _CARDS = [
        ("📹", "Multi-Video Batch Processing",
         "Ingest entire folders of clips as a <b style='color:#c8e6f5'>continuous timeline</b>"
         " — no manual stitching."
         "<br>Configurable frame sampling · 4K-capable · .mp4 .mov .mkv .avi"),
        ("🤖", "YOLO26 Custom Detector",
         "Purpose-trained on real underwater footage with"
         " <b style='color:#c8e6f5'>IoU deduplication</b> and nested-box filtering"
         " for pixel-accurate counts — no double-counting."),
        ("🧠", "Behavioral Pattern Analysis",
         "Detects <b style='color:#c8e6f5'>population-level events</b> from count dynamics:"
         "<br>· Sudden exodus / fleeing behaviour"
         "<br>· Novel species entry (e.g. cuttlefish incursion)"
         "<br>· Burst activity windows and quiet periods"),
        ("🔴", "Live Preview Mode",
         "Streams annotated frames in real time <i>during</i> processing"
         " — no waiting for the full run to see what the model found."),
        ("📊", "Analytics & AI Ecological Summary",
         "Time-series plots · peak identification · detection heatmaps."
         "<br><b style='color:#c8e6f5'>GPT-4o-mini</b> generates a plain-English"
         " ecological summary from the numeric output."),
        ("📄", "Multi-Format Export",
         "PDF report with peak annotated frames"
         "<br>SOP-compliant Excel (all samples + detections-only sheets)"
         "<br>ZIP of annotated frame gallery · CSV raw data"),
    ]

    col_a, col_b = st.columns(2, gap="medium")
    cols = [col_a, col_b]
    for i, (icon, title, body) in enumerate(_CARDS):
        with cols[i % 2]:
            st.markdown(
                "<div class='glass-card'>"
                f"<div style='color:#00e5ff;font-size:0.8rem;margin-bottom:6px'>"
                f"{icon}&nbsp; {title}</div>"
                f"<div style='color:#5ebbdc;font-size:0.74rem;line-height:1.75'>{body}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    # Behavioral callout
    st.markdown(
        "<div style='"
        "background:rgba(0,229,255,0.06);"
        "border:1px solid rgba(0,229,255,0.22);"
        "border-left:4px solid #00e5ff;"
        "border-radius:10px;"
        "padding:1.5rem 1.8rem;margin-top:0.4rem'>"
        "<div style='"
        "font-family:Orbitron,sans-serif;color:#00e5ff;"
        "font-size:0.78rem;letter-spacing:2px;margin-bottom:1rem'>"
        "WHAT THE SYSTEM HAS FLAGGED IN REAL DEPLOYMENTS"
        "</div>"
        "<div style='color:#c8e6f5;font-size:0.83rem;line-height:2.3'>"
        "🐟 &nbsp;Sudden 80%+ drop in fish count within 3 seconds"
        "&nbsp;→&nbsp;<b style='color:#00e5ff'>predator-induced fleeing event</b><br>"
        "🦑 &nbsp;Novel silhouette enters frame mid-session"
        "&nbsp;→&nbsp;<b style='color:#00e5ff'>cuttlefish / octopus incursion detected</b><br>"
        "📈 &nbsp;Multi-clip timeline reveals"
        " <b style='color:#00e5ff'>peak activity windows</b> tied to tidal cycle<br>"
        "🔕 &nbsp;Zero-detection stretches automatically flagged as"
        " <b style='color:#00e5ff'>low-activity or obstructed-camera periods</b>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_try_it() -> None:
    """Upload section — model loads only when a file is actually provided."""
    st.markdown(
        "<div class='section-label'>TRY IT ON YOUR OWN VIDEO</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#5ebbdc;font-size:0.78rem;margin-bottom:1.2rem;line-height:1.7'>"
        "Upload any underwater video and the model runs full detection, "
        "generates time-series analytics, and exports a PDF report."
        "<br><span style='color:#3d7fa0'>Model weights download on first run (~15 s).</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        sample_every = st.slider("Sample every N seconds", 1, 10, 3,
                                 help="Lower = more frames analyzed (slower).")
    with c2:
        conf = st.slider("Confidence threshold", 0.10, 0.95, 0.35, 0.05,
                         help="Minimum score for a detection to be counted.")
    with c3:
        topk = st.slider("Top frames to show", 3, 20, 5,
                         help="Number of best-detection frames in the gallery.")

    uploaded = st.file_uploader(
        f"Drag & drop an underwater video (max {MAX_UPLOAD_MB} MB)",
        type=["mp4", "mov", "mkv", "avi"],
        label_visibility="collapsed",
    )

    if not uploaded:
        return

    with tempfile.NamedTemporaryFile(delete=False,
                                     suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.success(f"✓ Loaded: **{uploaded.name}** ({uploaded.size / 1e6:.1f} MB)")

    # Model weights — download only now, not at page load
    if not MODEL_PATH.exists():
        with st.spinner("Downloading model weights…"):
            from huggingface_hub import hf_hub_download
            MODEL_PATH.parent.mkdir(exist_ok=True)
            hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
            hf_hub_download(
                repo_id="leodorf/fishvision-detector",
                filename="best_v1.04.pt",
                local_dir=str(MODEL_PATH.parent),
                token=hf_token,
            )

    if st.button("▶️ Run Fish Detection", type="primary"):
        with st.spinner("Loading model…"):
            model = load_model()
        try:
            import torch
            dev_label = "🟢 GPU" if torch.cuda.is_available() else "🔵 CPU"
        except Exception:
            dev_label = "🔵 CPU"
        st.caption(f"Inference device: {dev_label}")

        progress_bar = st.progress(0.0)
        status_text  = st.empty()
        df, topk_frames, stats = run_inference(
            video_path=tmp_path, model=model,
            sample_every=sample_every, conf=conf, topk=topk,
            progress_bar=progress_bar, status_text=status_text,
        )

        # KPI cards
        st.markdown("### 📡 Results")
        st.markdown(
            "<div style='display:flex;gap:14px;flex-wrap:wrap;margin-bottom:1rem'>"
            f"<div class='glass-card' style='flex:1;min-width:140px'>"
            f"<div class='kpi-title'>Peak fish count</div>"
            f"<div class='kpi-value'>{stats['peak_count']}</div>"
            f"<div class='kpi-sub'>@ {stats['peak_ts']}</div></div>"
            f"<div class='glass-card' style='flex:1;min-width:140px'>"
            f"<div class='kpi-title'>Detection frames</div>"
            f"<div class='kpi-value'>{stats['detection_frames']}</div>"
            f"<div class='kpi-sub'>of {stats['frames_sampled']} sampled</div></div>"
            f"<div class='glass-card' style='flex:1;min-width:140px'>"
            f"<div class='kpi-title'>Total detections</div>"
            f"<div class='kpi-value'>{stats['total_detections']}</div>"
            f"<div class='kpi-sub'>sum across all frames</div></div>"
            f"<div class='glass-card' style='flex:1;min-width:140px'>"
            f"<div class='kpi-title'>Video duration</div>"
            f"<div class='kpi-value'>{hms(stats['duration_s'])}</div>"
            f"<div class='kpi-sub'>sampled every {sample_every}s</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        fig_line, fig_hist = build_charts(df)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_line, use_container_width=True)
        with col2:
            st.plotly_chart(fig_hist, use_container_width=True)

        if topk_frames:
            st.markdown(f"### 🏆 Top {len(topk_frames)} Detection Frames")
            fcols = st.columns(min(len(topk_frames), 3))
            for idx, frame in enumerate(topk_frames):
                with fcols[idx % 3]:
                    st.image(frame["jpeg"],
                             caption=f"{frame['timestamp_hms']} · {frame['fish_count']} fish",
                             use_container_width=True)
        else:
            st.info("No fish detected. Try lowering the confidence threshold.")

        with st.expander("📊 Raw Detection Data"):
            df_show = df[df["fish_count"] > 0][
                ["timestamp_hms", "time_min", "fish_count"]
            ].reset_index(drop=True)
            st.dataframe(df_show, use_container_width=True, height=280)
            st.download_button(
                "⬇️ Download CSV",
                data=df_show.to_csv(index=False).encode(),
                file_name=f"detections_{Path(uploaded.name).stem}.csv",
                mime="text/csv",
            )

        st.markdown("### 📄 Export Report")
        pdf_bytes = make_pdf(stats, topk_frames, uploaded.name, conf)
        st.download_button(
            "⬇️ Download PDF Report",
            data=pdf_bytes,
            file_name=f"fishvision_report_{Path(uploaded.name).stem}.pdf",
            mime="application/pdf",
        )

        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def render_contact_cta() -> None:
    st.markdown(
        "<div class='cta-card'>"
        "<div style='"
        "font-family:Orbitron,sans-serif;color:#00e5ff;"
        "font-size:0.9rem;letter-spacing:2px;margin-bottom:0.4rem'>"
        "BUILT BY LEO DORFMAN"
        "</div>"
        "<div style='color:#5ebbdc;font-size:0.8rem;margin-bottom:1.4rem'>"
        "CS @ University of Florida &nbsp;·&nbsp; Computer Vision &amp; ML Engineering"
        "</div>"
        "<div style='display:flex;justify-content:center;gap:1.5rem;flex-wrap:wrap'>"
        "<a href='mailto:leodorfman1@gmail.com' style='"
        "color:#00e5ff;text-decoration:none;font-size:0.82rem;"
        "border:1px solid rgba(0,229,255,0.35);border-radius:6px;padding:7px 18px'>"
        "✉ leodorfman1@gmail.com</a>"
        "<a href='https://www.linkedin.com/in/leo-dorfman' target='_blank' style='"
        "color:#00e5ff;text-decoration:none;font-size:0.82rem;"
        "border:1px solid rgba(0,229,255,0.35);border-radius:6px;padding:7px 18px'>"
        "⬡ LinkedIn</a>"
        "<a href='https://github.com/eodorfman111' target='_blank' style='"
        "color:#00e5ff;text-decoration:none;font-size:0.82rem;"
        "border:1px solid rgba(0,229,255,0.35);border-radius:6px;padding:7px 18px'>"
        "⌥ GitHub</a>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="FishVision Demo",
        page_icon="🐟",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()

    # ── Page header (compact — keep video above the fold)
    st.markdown(
        "<div style='text-align:center;padding:0.3rem 0 0.4rem'>"
        "<div class='demo-badge'>LIVE DEMO</div>"
        "<h1 style='font-size:1.5rem!important;margin:0.2rem 0 0.1rem'>🐟 FishVision</h1>"
        "<p style='color:#5ebbdc;font-size:0.75rem;letter-spacing:3px;margin:0 0 0.4rem'>"
        "UNDERWATER MARINE SPECIES DETECTION &nbsp;·&nbsp; YOLO26 &nbsp;·&nbsp; COMPUTER VISION"
        "</p>"
        "<div>"
        "<span class='tag'>Computer Vision</span>"
        "<span class='tag'>YOLO26</span>"
        "<span class='tag'>Marine Research</span>"
        "<span class='tag'>Multi-Species</span>"
        "<span class='tag'>Underwater Video</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── Showcase sections (no model needed)
    render_hero()
    st.markdown("---")
    render_detection_gallery()
    st.markdown("---")
    render_pipeline_overview()
    st.markdown("---")
    render_try_it()
    st.markdown("---")
    render_contact_cta()


if __name__ == "__main__":
    main()
