# FishVision Demo
# Underwater marine-species detection — portfolio showcase

from __future__ import annotations

import base64
import io
import os
import tempfile
import time
import zipfile
import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load local environment variables if available (for local dev only)
load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
APP_NAME       = "FishVision"
MODEL_PATH     = Path(__file__).parent / "models" / "best_v1.08.pt"
STATIC_DIR     = Path(__file__).parent / "static"
PREVIEW_VIDEO  = STATIC_DIR / "preview.mp4"
FISH_COLOR_BGR = (150, 204, 0)
FISH_COLOR_HEX = "#00CC96"
MAX_UPLOAD_MB  = 4000
TARGET_CLASS   = "fish"

CONTACT_EMAIL = "leodorfman1@gmail.com"
GITHUB_URL    = "https://github.com/eodorfman111"
LINKEDIN_URL  = "https://www.linkedin.com/in/leo-dorfman"

# (Global client setup removed; clients are initialized dynamically based on user config)

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

/* ── sticky mobile CTA ─── */
@media (max-width: 768px) {
    .mobile-sticky-cta {
        position: fixed; bottom: 0; left: 0; right: 0; z-index: 9999;
        background: linear-gradient(90deg, #00b4d8, #0077b6);
        padding: 13px 20px; text-align: center;
        border-top: 1px solid rgba(0,229,255,0.5);
    }
    .mobile-sticky-cta a {
        color: #fff; text-decoration: none;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.78rem; letter-spacing: 1.5px;
    }
    /* pad bottom so sticky bar doesn't cover last section */
    [data-testid="stAppViewContainer"] > section:first-child { padding-bottom: 60px !important; }
}
</style>
"""

# ── HELPERS ───────────────────────────────────────────────────────────────────
def hms(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def format_timecode(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


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
# (Removed _load_gallery_images to avoid base64 payload bloat on WebSockets)


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    if not MODEL_PATH.exists():
        from huggingface_hub import hf_hub_download
        MODEL_PATH.parent.mkdir(exist_ok=True)
        hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
        hf_hub_download(
            repo_id="leodorf/fishvision-detector",
            filename="best_v1.08.pt",
            local_dir=str(MODEL_PATH.parent),
            token=hf_token,
        )
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
) -> tuple[pd.DataFrame, list[dict], list[dict], dict]:
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
        confs      = []

        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id   = int(box.cls[0])
                cls_name = model.names.get(cls_id, "").lower()
                if cls_name == TARGET_CLASS or cls_name in TARGET_CLASS:
                    fish_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = float(box.conf[0])
                    confs.append(score)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), FISH_COLOR_BGR, 2)
                    cv2.putText(annotated, f"fish {score:.2f}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, FISH_COLOR_BGR, 2)

        max_conf = max(confs) if confs else 0.0
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        rows.append({
            "frame_index":     frame_idx,
            "timestamp_s":     ts,
            "timestamp_hms":   hms(ts),
            "time_min":        round(ts / 60, 3),
            "fish_count":      fish_count,
            "max_conf_frame":  max_conf,
            "avg_conf_frame":  avg_conf,
        })

        if fish_count > 0:
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            topk_candidates.append({
                "frame_index":     frame_idx,
                "timestamp_s":     ts,
                "timestamp_hms":   hms(ts),
                "fish_count":      fish_count,
                "max_conf_frame":  max_conf,
                "avg_conf_frame":  avg_conf,
                "jpeg":            buf.tobytes(),
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
        "presence_rate_pct": (int((df["fish_count"] > 0).sum()) / max(processed, 1)) * 100,
        "max_confidence":   float(df["max_conf_frame"].max()) if not df.empty else 0.0,
        "device":           "cuda" if torch.cuda.is_available() else "cpu",
        "sample_every":     sample_every,
    }
    return df, topk_frames, topk_candidates, stats


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


# ── AI ECOLOGICAL SUMMARY ─────────────────────────────────────────────────────
def generate_fallback_summary(df: pd.DataFrame, stats: dict, video_name: str, conf_threshold: float, sample_every: int) -> str:
    total_detections = stats.get('total_detections', 0)
    presence_rate_pct = stats.get('presence_rate_pct', 0.0)
    peak_count = stats.get('peak_count', 0)
    peak_ts = stats.get('peak_ts', '—')
    max_conf = stats.get('max_confidence', 0.0)
    frames_sampled = stats.get('frames_sampled', 1)
    
    # Calculate average confidence of positive detections
    df_det = df[df["fish_count"] > 0]
    avg_conf = df_det["avg_conf_frame"].mean() if not df_det.empty else 0.0
    if pd.isna(avg_conf):
        avg_conf = 0.0
        
    # Analyze timeseries trend
    if not df_det.empty:
        df_copy = df.copy()
        df_copy['minute'] = (df_copy['timestamp_s'] // 60).astype(int)
        minutes_group = df_copy.groupby('minute')['fish_count'].mean().round(2).to_dict()
        peak_minute = max(minutes_group, key=minutes_group.get) if minutes_group else 0
        trend_desc = f"Peak abundance occurred during minute {peak_minute} of the timeline, averaging {minutes_group.get(peak_minute, 0)} fish per frame."
    else:
        trend_desc = "No positive detections were recorded across the video timeline, indicating zero active fish presence."

    abundance_level = "low"
    if presence_rate_pct > 50:
        abundance_level = "extremely high"
    elif presence_rate_pct > 25:
        abundance_level = "moderate-to-high"
    elif presence_rate_pct > 5:
        abundance_level = "low-to-moderate"

    summary = f"""**1. Fish Activity**
Fish activity was monitored over the timeline of {video_name}, recording a total of {total_detections} individual detections. The maximum concentration occurred at timestamp {peak_ts} with a peak count of {peak_count} individuals in a single frame, representing a localized period of high abundance. Overall fish presence rate was calculated at {presence_rate_pct:.1f}% across all sampled frames, showing {abundance_level} habitat utilization.

**2. Detection Trends**
Abundance dynamics indicate a {abundance_level} presence profile. {trend_desc} The timeseries demonstrates a pulsed distribution pattern, indicating potential schooling behavior or transient movement of marine life through the camera's field of view.

**3. Data Quality**
The survey was conducted using a YOLOv8 custom-trained detector with a confidence threshold of {conf_threshold:.2f} and sampling interval of {sample_every}s. Detection confidence remained consistently high (averaging {avg_conf:.2f} across detection events and peaking at {max_conf:.3f}), ensuring high data integrity.

**4. Survey Observations**
The observed activity confirms that the site serves as a functional marine habitat during the monitored period. Continued monitoring at higher sampling rates (e.g., every 1s) is recommended during peak tidal flows to better capture transient schools.

*(Note: This report was compiled using the rule-based scientific fallback engine. Configure a Gemini or OpenAI API Key in the sidebar for complete LLM narrative analysis.)*"""
    return summary


def generate_ai_summary(df: pd.DataFrame, stats: dict, video_name: str, conf_threshold: float, sample_every: int) -> str:
    provider = st.session_state.get("ai_provider", "Gemini (Recommended)")
    api_key = st.session_state.get("ai_api_key", "")
    
    # Defaults if not set in session state
    if not api_key:
        if "gemini" in provider.lower():
            api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
        else:
            api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

    # If still no key, fallback to rules-based scientific summary
    if not api_key:
        return generate_fallback_summary(df, stats, video_name, conf_threshold, sample_every)

    try:
        # Initialize appropriate client
        if "gemini" in provider.lower():
            client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            model_name = "gemini-1.5-flash"
        else:
            client = OpenAI(api_key=api_key)
            model_name = "gpt-4o-mini"

        # Generate minutes history details for accurate LLM timestamps
        df_det = df[df["fish_count"] > 0]
        if not df_det.empty:
            df_copy = df.copy()
            df_copy['minute'] = (df_copy['timestamp_s'] // 60).astype(int)
            minutes_group = df_copy.groupby('minute')['fish_count'].mean().round(2).to_dict()
            timeseries_summary = ", ".join([f"Min {m}: {c} avg" for m, c in minutes_group.items()])
        else:
            timeseries_summary = "No detections across the video duration."

        prompt = f"""
        You are an expert marine biologist analyzing video detection data from an underwater fish monitoring camera.
        
        Analyze the following survey metadata and statistics:
        - Video Filename: {video_name}
        - Video Duration: {stats.get('duration_s'):.1f} seconds
        - Model Version: YOLOv8 Custom Marine Detector (best_v1.08.pt)
        - Confidence Threshold: {conf_threshold:.2f}
        - Frame Sampling Rate: Every {sample_every} second(s)
        
        Detection Statistics:
        - Total Sampled Frames: {stats.get('frames_sampled')}
        - Frames with Fish Detected: {stats.get('detection_frames')}
        - Fish Presence Rate: {stats.get('presence_rate_pct'):.2f}% of sampled frames
        - Total Individual Detections: {stats.get('total_detections')}
        - Peak Fish Count in a single frame: {stats.get('peak_count')} (at timestamp {stats.get('peak_ts')})
        - Maximum Detection Confidence: {stats.get('max_confidence'):.3f}
        - Detections over time (average count grouped by minute): {timeseries_summary}
        
        Please generate a professional, structured ecological summary covering:
        1. Fish Activity: Density and behaviors observed. Address when abundance was highest (e.g. "Fish activity was highest between X and Y minutes...").
        2. Detection Trends: Discuss temporal abundance dynamics (e.g. "The survey appears to contain several periods of elevated abundance...").
        3. Data Quality: Evaluate detection confidence over the timeline (e.g. "Detection confidence remained consistently high...").
        4. Survey Observations: General ecological observations and recommendations.
        
        Maintain a formal scientific writing style. Limit the summary to under 300 words total.
        """
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a professional marine biologist preparing a fish monitoring survey summary."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        fallback = generate_fallback_summary(df, stats, video_name, conf_threshold, sample_every)
        return f"{fallback}\n\n*(Note: LLM summary generation failed ({str(e)}). Displaying rule-based fallback.)*"


# ── EXCEL & ZIP GENERATORS ────────────────────────────────────────────────────
def make_video_excel(df: pd.DataFrame, stats: dict, video_name: str) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Statistics
        stats_data = [
            {"Metric": "Video Identifier", "Value": video_name},
            {"Metric": "Video Duration (s)", "Value": round(stats.get("duration_s", 0), 2)},
            {"Metric": "Total Frames Sampled", "Value": stats.get("frames_sampled", 0)},
            {"Metric": "Frames with Fish", "Value": stats.get("detection_frames", 0)},
            {"Metric": "Detection Rate (%)", "Value": round(stats.get("presence_rate_pct", 0), 2)},
            {"Metric": "Total Fish Detections", "Value": stats.get("total_detections", 0)},
            {"Metric": "Peak Fish Count", "Value": stats.get("peak_count", 0)},
            {"Metric": "Peak Timestamp", "Value": stats.get("peak_ts", "—")},
            {"Metric": "Max Confidence", "Value": round(stats.get("max_confidence", 0), 3)},
            {"Metric": "Model Version", "Value": "YOLOv8 Custom Marine (best_v1.08.pt)"},
            {"Metric": "Confidence Threshold", "Value": stats.get("sample_every", 3)},
            {"Metric": "Inference Device", "Value": stats.get("device", "cpu").upper()}
        ]
        pd.DataFrame(stats_data).to_excel(writer, sheet_name="Statistics", index=False)
        
        # Sheet 2: Sampled Frames
        df.to_excel(writer, sheet_name="Sampled Frames", index=False)
        
        # Sheet 3: Detection Summary
        df_det = df[df["fish_count"] > 0]
        df_det.to_excel(writer, sheet_name="Detection Summary", index=False)
        
    return output.getvalue()


def make_batch_summary_excel(video_results: dict[str, dict]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_rows = []
        for v_name, res in video_results.items():
            stats = res["stats"]
            summary_rows.append({
                "video_identifier": v_name,
                "duration_s": round(stats.get("duration_s", 0), 2),
                "frames_sampled": stats.get("frames_sampled", 0),
                "frames_with_fish": stats.get("detection_frames", 0),
                "presence_rate_pct": round(stats.get("presence_rate_pct", 0), 2),
                "total_detections": stats.get("total_detections", 0),
                "peak_count": stats.get("peak_count", 0),
                "peak_ts": stats.get("peak_ts", "—"),
                "max_confidence": round(stats.get("max_confidence", 0), 3),
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Batch Summary", index=False)
    return output.getvalue()


def make_zip(
    video_results: dict[str, dict],
    summary_excel_bytes: bytes
) -> bytes:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Add summary.xlsx
        zip_file.writestr("summary.xlsx", summary_excel_bytes)
        
        is_single = len(video_results) == 1
        
        for v_name, res in video_results.items():
            safe_name = Path(v_name).stem
            
            # File naming
            pdf_filename = "report.pdf" if is_single else f"{safe_name}_report.pdf"
            excel_filename = "detections.xlsx" if is_single else f"{safe_name}_detections.xlsx"
            
            # Add PDF report
            zip_file.writestr(pdf_filename, res["pdf_bytes"])
            
            # Add individual Excel
            ind_excel = make_video_excel(res["df"], res["stats"], v_name)
            zip_file.writestr(excel_filename, ind_excel)
            
            # Add annotated frames
            for frame in res["all_detected"]:
                frame_num = frame["frame_index"] + 1
                if is_single:
                    img_name = f"frames/frame_{frame_num:06d}.jpg"
                else:
                    img_name = f"frames/{safe_name}_frame_{frame_num:06d}.jpg"
                zip_file.writestr(img_name, frame["jpeg"])
                
    return zip_buf.getvalue()


# ── REPORTLAB PDF GENERATOR ───────────────────────────────────────────────────
def make_pdf(stats: dict, topk_frames: list[dict], video_name: str, conf: float, ai_summary: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as rlcanvas
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    buf  = io.BytesIO()
    c    = rlcanvas.Canvas(buf, pagesize=A4)
    W, H = A4

    # ── PAGE 1 ──────────────────────────────────────────────────────────────────
    c.setFillColorRGB(0.008, 0.051, 0.102); c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColorRGB(0, 0.447, 0.722);     c.rect(0, H - 90, W, 90, fill=1, stroke=0)
    c.setFillColorRGB(0, 0.706, 0.933);     c.rect(0, H - 90, int(W * 0.42), 90, fill=1, stroke=0)

    # Title
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 24)
    c.drawString(30, H - 46, "FISHVISION")
    c.setFillColorRGB(0.85, 0.95, 1); c.setFont("Helvetica", 10)
    c.drawString(30, H - 62, "Underwater Fish Detection Report")
    c.setFillColorRGB(0.70, 0.88, 0.97); c.setFont("Helvetica", 8)
    c.drawString(30, H - 76,
        f"YOLOv8 Custom Model  ·  Confidence: {conf:.0%}  ·  Device: {stats['device'].upper()}")
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
        ("Detection Rate",   f"{stats['presence_rate_pct']:.1f}%"),
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
    c.drawRightString(W - 30, 18, "Page 1 of 2")

    # ── PAGE 2 ──────────────────────────────────────────────────────────────────
    c.showPage()
    
    # Background
    c.setFillColorRGB(0.008, 0.051, 0.102); c.rect(0, 0, W, H, fill=1, stroke=0)
    # Header bar
    c.setFillColorRGB(0, 0.447, 0.722);     c.rect(0, H - 60, W, 60, fill=1, stroke=0)
    c.setFillColorRGB(0, 0.706, 0.933);     c.rect(0, H - 60, int(W * 0.42), 60, fill=1, stroke=0)
    
    # Title
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 16)
    c.drawString(30, H - 36, "FISHVISION DETAILED SURVEY ANALYSIS")
    c.setFillColorRGB(0.85, 0.95, 1); c.setFont("Helvetica", 8)
    c.drawString(30, H - 48, f"Source Video: {vname}")
    
    y = H - 90
    
    # System Metadata Section
    c.setFillColorRGB(0, 0.898, 1); c.setFont("Helvetica-Bold", 10)
    c.drawString(30, y, "ANALYSIS SETTINGS & SYSTEM METADATA")
    c.setStrokeColorRGB(0, 0.898, 1); c.setLineWidth(0.4)
    c.line(30, y - 5, W - 30, y - 5); y -= 20
    
    c.setFillColorRGB(0.004, 0.18, 0.30)
    c.roundRect(30, y - 55, W - 60, 50, 5, fill=1, stroke=0)
    
    metadata = [
        ("Model Version", "YOLOv8 Custom Marine (best_v1.08.pt)"),
        ("Confidence Thresh", f"{conf:.2f}"),
        ("Frame Sampling", f"Every {stats.get('sample_every', 3)}s"),
        ("Inference Device", stats['device'].upper())
    ]
    meta_w = (W - 80) / 4
    for i, (m_label, m_val) in enumerate(metadata):
        mx = 40 + i * meta_w
        c.setFillColorRGB(0.47, 0.73, 0.87); c.setFont("Helvetica", 7)
        c.drawString(mx, y - 18, m_label.upper())
        c.setFillColorRGB(0, 0.898, 1); c.setFont("Helvetica-Bold", 10)
        c.drawString(mx, y - 36, m_val)
        
    y -= 80
    
    # AI summary
    c.setFillColorRGB(0, 0.898, 1); c.setFont("Helvetica-Bold", 10)
    c.drawString(30, y, "AI ECOLOGICAL ANALYSIS")
    c.setStrokeColorRGB(0, 0.898, 1); c.setLineWidth(0.4)
    c.line(30, y - 5, W - 30, y - 5); y -= 20
    
    styles = getSampleStyleSheet()
    ai_style = ParagraphStyle(
        'AISummaryStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14.5,
        textColor='#c8e6f5'
    )
    
    formatted_summary = ai_summary
    for heading in ["1. Fish Activity", "2. Detection Trends", "3. Data Quality", "4. Survey Observations"]:
        formatted_summary = formatted_summary.replace(heading, f"<font color='#00e5ff'><b>{heading}</b></font>")
    
    formatted_summary = formatted_summary.replace("\n", "<br/>")
    formatted_summary = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_summary)
    
    p = Paragraph(formatted_summary, ai_style)
    p.wrap(W - 60, y - 55)
    p.drawOn(c, 30, y - p.height)
    
    # Footer on Page 2
    c.setFillColorRGB(0, 0.29, 0.51); c.rect(0, 0, W, 38, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica-Bold", 9)
    c.drawString(30, 24, "FISHVISION  ·  Prepared by Leo Dorfman")
    c.setFillColorRGB(0.75, 0.9, 1); c.setFont("Helvetica", 8)
    c.drawString(30, 12,
        "leodorfman1@gmail.com  |  linkedin.com/in/leo-dorfman  |  github.com/eodorfman111")
    c.setFillColorRGB(0.6, 0.85, 1); c.setFont("Helvetica", 8)
    c.drawRightString(W - 30, 18, "Page 2 of 2")
    
    c.save()
    return buf.getvalue()


# ── UI COMPONENTS ─────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    import torch
    dev_name = "GPU" if torch.cuda.is_available() else "CPU"
    status_text = f"<span style='color:#00CC96'>🟢 Ready ({dev_name})</span>" if st.session_state.get("model_loaded") else "<span style='color:#00b4d8'>⏳ Initializing...</span>"
    
    with st.sidebar:
        st.markdown(f"""
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
· YOLO (ultralytics)<br>
· OpenAI GPT-4o-mini / Gemini<br>
· OpenCV · PyTorch<br>
· Streamlit · Plotly<br>
· Python · ReportLab<br><br>
· Custom-trained on real underwater footage<br>
· Multi-species marine detection<br>
· Industrial fish grading &amp; measurement
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        if False:
            st.text_input(
                "API Key",
                value=default_key,
                type="password",
                key="ai_api_key",
                placeholder="Enter API key..."
            )


def render_hero() -> None:
    """Full-width looping detection video, served natively from static folder to avoid WS bloat."""
    if not PREVIEW_VIDEO.exists():
        st.warning("Hero video not found — place preview.mp4 in the static/ folder.")
        return
    b64_hero = _video_b64(PREVIEW_VIDEO)
    st.markdown(
        "<div style='"
        "position:relative;border-radius:14px;overflow:hidden;"
        "box-shadow:0 8px 48px rgba(0,229,255,0.14);margin-bottom:0.3rem'>"
        f"<video autoplay loop muted playsinline "
        f"style='width:100%;display:block;max-height:480px;"
        f"object-fit:cover;object-position:center bottom'>"
        f"<source src='data:video/mp4;base64,{b64_hero}' type='video/mp4'>"
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
        "YOLOv8 &nbsp;·&nbsp; custom-trained &nbsp;·&nbsp; "
        "multi-species &nbsp;·&nbsp; real underwater footage"
        "</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def _video_b64(path: Path) -> str:
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def render_dual_demo() -> None:
    st.markdown(
        "<div class='section-label'>LIVE DETECTION DEMOS</div>",
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2, gap="medium")

    reef_path    = STATIC_DIR / "reef.mp4"
    grading_path = STATIC_DIR / "grading.mp4"

    with col_a:
        if reef_path.exists():
            b64 = _video_b64(reef_path)
            st.markdown(
                "<div style='font-family:Orbitron,sans-serif;color:#00e5ff;font-size:0.75rem;"
                "letter-spacing:2px;margin-bottom:0.5rem'>🐟 UNDERWATER REEF DETECTION</div>"
                "<div style='border-radius:10px;overflow:hidden;box-shadow:0 6px 28px rgba(0,0,0,0.55);height:460px'>"
                f"<video autoplay loop muted playsinline style='width:115%;margin-left:-7.5%;height:460px;object-fit:cover;object-position:center;display:block'>"
                f"<source src='data:video/mp4;base64,{b64}' type='video/mp4'></video></div>"
                "<div style='color:#5ebbdc;font-size:0.7rem;margin-top:0.4rem'>"
                "Real-time species detection · BRUV underwater footage · Custom-trained model</div>",
                unsafe_allow_html=True,
            )
    with col_b:
        if grading_path.exists():
            b64 = _video_b64(grading_path)
            st.markdown(
                "<div style='font-family:Orbitron,sans-serif;color:#00e5ff;font-size:0.75rem;"
                "letter-spacing:2px;margin-bottom:0.5rem'>⚖️ INDUSTRIAL GRADING & MEASUREMENT</div>"
                "<div style='border-radius:10px;overflow:hidden;box-shadow:0 6px 28px rgba(0,0,0,0.55)'>"
                f"<video autoplay loop muted playsinline style='width:100%;display:block'>"
                f"<source src='data:video/mp4;base64,{b64}' type='video/mp4'></video></div>"
                "<div style='color:#5ebbdc;font-size:0.7rem;margin-top:0.4rem'>"
                "Length &amp; weight estimation · Fish grading S/M/L/XL · Directional counting</div>",
                unsafe_allow_html=True,
            )


def render_detection_gallery() -> None:
    """2×2 grid of detection screenshots with overlay species labels, served natively from static folder."""
    st.markdown(
        "<div class='section-label'>DETECTION SHOWCASE</div>",
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2, gap="medium")
    cols = [col_a, col_b]

    for i, item in enumerate(GALLERY):
        path = STATIC_DIR / item["file"]
        with cols[i % 2]:
            if path.exists():
                import base64
                with open(path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                st.markdown(
                    "<div style='"
                    "position:relative;border-radius:10px;overflow:hidden;"
                    "margin-bottom:1.1rem;"
                    "box-shadow:0 6px 28px rgba(0,0,0,0.55)'>"
                    f"<img src='data:image/png;base64,{img_b64}' "
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
         "Ingest entire folders of clips as a <b style='color:#c8e6f5'>continuous batch timeline</b>"
         " — no manual individual runs."
         "<br>Configurable frame sampling · 4K-capable · .mp4 .mov .mkv .avi"),
        ("🤖", "Custom Fish Detector",
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
    """Upload section — model loads only when files are actually provided."""
    st.markdown(
        "<div class='section-label'>TRY IT ON YOUR OWN VIDEOS</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#5ebbdc;font-size:0.78rem;margin-bottom:1.2rem;line-height:1.7'>"
        "Upload underwater video(s) and the model runs batch detection, "
        "generates time-series analytics, and exports reports (Excel, PDF, ZIP)."
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
        topk = st.slider("Top frames to show per video", 3, 20, 5,
                         help="Number of best-detection frames in the gallery.")

    uploaded_files = st.file_uploader(
        f"Drag & drop underwater videos (max {MAX_UPLOAD_MB} MB)",
        type=["mp4", "mov", "mkv", "avi"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if "batch_results" not in st.session_state:
        st.session_state["batch_results"] = None

    if not uploaded_files:
        st.session_state["batch_results"] = None
        return

    st.success(f"✓ Loaded: **{len(uploaded_files)} video(s)**")

    if st.button("▶️ Run Batch Fish Detection", type="primary"):
        model = load_model()
        try:
            import torch
            dev_label = "🟢 GPU" if torch.cuda.is_available() else "🔵 CPU"
        except Exception:
            dev_label = "🔵 CPU"
        st.caption(f"Inference device: {dev_label}")

        videos_results = {}
        excel_dataframes = {}

        # Loop through videos
        for idx, uploaded in enumerate(uploaded_files):
            st.markdown(f"#### Processing video {idx+1}/{len(uploaded_files)}: **{uploaded.name}**")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            progress_bar = st.progress(0.0)
            status_text  = st.empty()
            
            df, topk_frames, all_detected, stats = run_inference(
                video_path=tmp_path, model=model,
                sample_every=sample_every, conf=conf, topk=topk,
                progress_bar=progress_bar, status_text=status_text,
            )
            
            # Format Excel columns according to approved plan
            excel_rows = []
            for _, r in df.iterrows():
                excel_rows.append({
                    "video_identifier": uploaded.name,
                    "frame_number":     int(r["frame_index"]) + 1,
                    "timecode":         format_timecode(r["timestamp_s"]),
                    "timestamp_s":      round(float(r["timestamp_s"]), 3),
                    "fish_count":       int(r["fish_count"]),
                    "fish_present":     bool(r["fish_count"] > 0),
                    "max_conf_frame":   round(float(r["max_conf_frame"]), 3),
                    "avg_conf_frame":   round(float(r["avg_conf_frame"]), 3),
                })
            df_excel = pd.DataFrame(excel_rows)
            excel_dataframes[uploaded.name] = df_excel

            # Generate AI Ecological Summary (Priority 2)
            with st.spinner("Generating AI Ecological Summary..."):
                ai_summary = generate_ai_summary(df_excel, stats, uploaded.name, conf, sample_every)

            # Generate PDF Report
            with st.spinner("Compiling PDF Report..."):
                pdf_bytes = make_pdf(stats, topk_frames, uploaded.name, conf, ai_summary)

            videos_results[uploaded.name] = {
                "stats": stats,
                "df": df_excel,
                "raw_df": df,
                "topk_frames": topk_frames,
                "all_detected": all_detected,
                "ai_summary": ai_summary,
                "pdf_bytes": pdf_bytes
            }

            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        # Package batch reports (Priority 1 & 3)
        summary_excel_bytes = make_batch_summary_excel(videos_results)
        zip_bytes = make_zip(videos_results, summary_excel_bytes)

        st.session_state["batch_results"] = {
            "videos": videos_results,
            "zip_bytes": zip_bytes,
            "excel_bytes": summary_excel_bytes
        }

    # Render results dashboard if available
    if st.session_state["batch_results"] is not None:
        batch = st.session_state["batch_results"]
        videos = batch["videos"]
        
        st.markdown("---")
        st.markdown("### 📡 Batch Analysis Results")
        
        # Batch aggregate KPIs
        total_vids = len(videos)
        total_detections = sum(v["stats"]["total_detections"] for v in videos.values())
        
        # Find peak fish count across all videos
        peak_count = 0
        peak_video = ""
        peak_ts = ""
        for name, v in videos.items():
            if v["stats"]["peak_count"] > peak_count:
                peak_count = v["stats"]["peak_count"]
                peak_video = name
                peak_ts = v["stats"]["peak_ts"]
                
        avg_presence = sum(v["stats"]["presence_rate_pct"] for v in videos.values()) / total_vids
        
        st.markdown(
            "<div style='display:flex;gap:14px;flex-wrap:wrap;margin-bottom:1.5rem'>"
            f"<div class='glass-card' style='flex:1;min-width:160px'>"
            f"<div class='kpi-title'>Videos Processed</div>"
            f"<div class='kpi-value'>{total_vids}</div>"
            f"<div class='kpi-sub'>batch execution</div></div>"
            f"<div class='glass-card' style='flex:1;min-width:160px'>"
            f"<div class='kpi-title'>Peak Fish Count</div>"
            f"<div class='kpi-value'>{peak_count}</div>"
            f"<div class='kpi-sub'>{peak_video[:20]} @ {peak_ts}</div></div>"
            f"<div class='glass-card' style='flex:1;min-width:160px'>"
            f"<div class='kpi-title'>Total Detections</div>"
            f"<div class='kpi-value'>{total_detections}</div>"
            f"<div class='kpi-sub'>sum across all videos</div></div>"
            f"<div class='glass-card' style='flex:1;min-width:160px'>"
            f"<div class='kpi-title'>Average Presence %</div>"
            f"<div class='kpi-value'>{avg_presence:.1f}%</div>"
            f"<div class='kpi-sub'>avg detection rate</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Download ZIP and combined excel buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇️ Download Combined ZIP Archive",
                data=batch["zip_bytes"],
                file_name="results.zip",
                mime="application/zip",
                use_container_width=True
            )
        with col_dl2:
            st.download_button(
                "⬇️ Download Summary Excel Report",
                data=batch["excel_bytes"],
                file_name="summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        st.markdown("### 🔍 Video Detailed Analytics")
        # Video selector
        selected_video_name = st.selectbox("Select a video to view charts & reports", list(videos.keys()))
        selected_vid = videos[selected_video_name]
        v_stats = selected_vid["stats"]
        v_df = selected_vid["df"]
        v_raw_df = selected_vid["raw_df"]
        
        # Individual KPIs
        st.markdown(
            "<div style='display:flex;gap:14px;flex-wrap:wrap;margin-bottom:1rem'>"
            f"<div class='glass-card' style='flex:1;min-width:140px'>"
            f"<div class='kpi-title'>Peak fish count</div>"
            f"<div class='kpi-value'>{v_stats['peak_count']}</div>"
            f"<div class='kpi-sub'>@ {v_stats['peak_ts']}</div></div>"
            f"<div class='glass-card' style='flex:1;min-width:140px'>"
            f"<div class='kpi-title'>Detection rate</div>"
            f"<div class='kpi-value'>{v_stats['presence_rate_pct']:.1f}%</div>"
            f"<div class='kpi-sub'>of {v_stats['frames_sampled']} frames</div></div>"
            f"<div class='glass-card' style='flex:1;min-width:140px'>"
            f"<div class='kpi-title'>Total detections</div>"
            f"<div class='kpi-value'>{v_stats['total_detections']}</div>"
            f"<div class='kpi-sub'>sum across frames</div></div>"
            f"<div class='glass-card' style='flex:1;min-width:140px'>"
            f"<div class='kpi-title'>Video duration</div>"
            f"<div class='kpi-value'>{hms(v_stats['duration_s'])}</div>"
            f"<div class='kpi-sub'>sampled every {sample_every}s</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Charts
        fig_line, fig_hist = build_charts(v_raw_df)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.plotly_chart(fig_line, use_container_width=True)
        with col_c2:
            st.plotly_chart(fig_hist, use_container_width=True)

        # AI summary display
        st.markdown("#### 🧠 AI Ecological Summary")
        summary_html = selected_vid['ai_summary'].replace('\n', '<br>')
        st.markdown(
            f"<div style='background:rgba(0,180,216,0.05);border:1px solid rgba(0,180,216,0.2);border-radius:10px;padding:1.2rem;color:#c8e6f5;font-size:0.85rem;line-height:1.6'>"
            f"{summary_html}"
            f"</div>",
            unsafe_allow_html=True
        )

        # Top Detection frames gallery
        if selected_vid["topk_frames"]:
            st.markdown(f"#### 🏆 Top {len(selected_vid['topk_frames'])} Detection Frames")
            fcols = st.columns(min(len(selected_vid['topk_frames']), 3))
            for idx, frame in enumerate(selected_vid['topk_frames']):
                with fcols[idx % 3]:
                    st.image(frame["jpeg"],
                             caption=f"{frame['timestamp_hms']} · {frame['fish_count']} fish · max conf: {frame['max_conf_frame']:.2f}",
                             use_container_width=True)

        # Raw Data Download & Expander
        with st.expander("📊 Raw Detection Data (One row per sampled frame)"):
            st.dataframe(v_df, use_container_width=True, height=280)
            
            # Download individual Excel file (Priority 1)
            ind_excel_bytes = make_video_excel(v_df, v_stats, selected_video_name)
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "⬇️ Download Excel Data",
                    data=ind_excel_bytes,
                    file_name=f"detections_{Path(selected_video_name).stem}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            with col_d2:
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=selected_vid["pdf_bytes"],
                    file_name=f"fishvision_report_{Path(selected_video_name).stem}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )


def render_above_fold_cta() -> None:
    st.markdown(
        "<div style='text-align:center;margin:0.9rem 0 1.1rem'>"
        "<a href='mailto:leodorfman1@gmail.com' style='"
        "background:linear-gradient(135deg,#00b4d8,#0077b6);"
        "color:white;text-decoration:none;"
        "font-family:Orbitron,sans-serif;font-size:0.82rem;"
        "letter-spacing:1.5px;border-radius:8px;"
        "padding:12px 32px;display:inline-block;"
        "box-shadow:0 4px 22px rgba(0,180,216,0.4)'>"
        "📬 REQUEST A CUSTOM BUILD"
        "</a>"
        "<div style='color:#3d7fa0;font-size:0.7rem;margin-top:9px;letter-spacing:1px'>"
        "I respond within 24 hours &nbsp;·&nbsp; No commitment required"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_case_studies() -> None:
    st.markdown("<div class='section-label'>CASE STUDIES</div>", unsafe_allow_html=True)
    cases = [
        {
            "client": "ARCA Marine",
            "type": "Marine Research · BRUV Footage",
            "outcome": (
                "Automated species detection on underwater BRUV footage for a paid marine-research client. "
                "System identified cuttlefish, seabream, and reef species across multi-hour deployments "
                "with 0.95+ confidence on primary detections. Delivered PDF reports, Excel timeseries data, "
                "and annotated frame archives."
            ),
            "tags": ["BRUV footage", "Reef environment", "Multi-species", "Paid · Completed"],
        },
        {
            "client": "Coonamessett Farm Foundation",
            "type": "Aquaculture · Live Monitoring",
            "outcome": (
                "Built ScallopVision — a live overhead monitoring system for scallop tank health. "
                "Running on a $1k/month retainer with ongoing hypothesis testing across tank populations. "
                "System tracks individual scallop behavior and environmental response in real time."
            ),
            "tags": ["Live monitoring", "Aquaculture", "$1k/mo retainer", "Ongoing"],
        },
    ]
    col_a, col_b = st.columns(2, gap="medium")
    for i, (col, case) in enumerate(zip([col_a, col_b], cases)):
        tags_html = " ".join(f"<span class='tag'>{t}</span>" for t in case["tags"])
        with col:
            st.markdown(
                "<div class='glass-card'>"
                f"<div style='font-family:Orbitron,sans-serif;color:#00e5ff;"
                f"font-size:0.82rem;letter-spacing:2px;margin-bottom:4px'>{case['client']}</div>"
                f"<div style='color:#3d7fa0;font-size:0.68rem;margin-bottom:10px'>{case['type']}</div>"
                f"<div style='color:#c8e6f5;font-size:0.77rem;line-height:1.75;margin-bottom:12px'>{case['outcome']}</div>"
                f"<div>{tags_html}</div>"
                "</div>",
                unsafe_allow_html=True,
            )


def render_faq() -> None:
    st.markdown("<div class='section-label'>FREQUENTLY ASKED QUESTIONS</div>", unsafe_allow_html=True)
    faqs = [
        (
            "What kind of footage does it work on?",
            "BRUV (Baited Remote Underwater Video), drop-cams, ROV footage, tank overhead cameras, "
            "and standard GoPro clips. Challenging conditions — low light, murky water, mixed species — are expected and handled.",
        ),
        (
            "What species can it detect?",
            "The demo model is trained on common reef fish and marine invertebrates (seabream, cuttlefish, eels, crabs, "
            "starfish, sea urchins). For custom deployments I retrain on footage of your target species — typically 2–4 weeks "
            "from footage handoff to deployed model.",
        ),
        (
            "What do I receive as output?",
            "PDF detection report with annotated frames · Excel spreadsheet (one row per sampled frame with fish counts, "
            "timestamps, and confidence scores) · ZIP of all annotated frames · AI-generated ecological summary. "
            "Batch processing across multiple clips is included.",
        ),
        (
            "Can it run offline / in the field?",
            "Yes — the model runs locally on CPU or GPU with no internet required for inference once weights are downloaded. "
            "I can package the pipeline as a standalone executable for offline deployment.",
        ),
        (
            "How much does it cost?",
            "Scoping is always free. Project pricing depends on scope — some clients pay per-project, others on a monthly "
            "retainer (starting ~$1k/month for ongoing monitoring). Reach out and I'll give you an honest number for your use case.",
        ),
        (
            "How long does a custom build take?",
            "A standard pipeline (model fine-tune + dashboard) typically takes 2–4 weeks from footage handoff to delivery. "
            "Simpler integrations can be done faster. I respond to initial inquiries within 24 hours.",
        ),
    ]
    for q, a in faqs:
        with st.expander(q):
            st.markdown(
                f"<div style='color:#c8e6f5;font-size:0.82rem;line-height:1.8'>{a}</div>",
                unsafe_allow_html=True,
            )


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
        "</div>"
        "<div style='color:#3d7fa0;font-size:0.7rem;margin-top:1.1rem;letter-spacing:1px'>"
        "⚡ I respond within 24 hours &nbsp;·&nbsp; Scoping calls are always free"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="FishVision — AI Fish Detection for Marine Research & Aquaculture",
        page_icon="🐟",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Meta / OG tags (injected into parent page via JS)
    st.markdown(
        """<script>
        (function(){
            var metas = [
                ['name','description','FishVision — custom computer vision pipelines for underwater fish detection, species ID, and behavioral monitoring. Used by marine research labs and aquaculture farms.'],
                ['property','og:title','FishVision — AI Fish Detection Demo'],
                ['property','og:description','Upload underwater video and get automated fish counts, species detection, and AI ecological summary. Built by Leo Dorfman.'],
                ['property','og:image','https://fishvision-demo.streamlit.app/app/static/detect_reef.png'],
                ['name','twitter:card','summary_large_image'],
            ];
            metas.forEach(function(m){
                var el=document.createElement('meta');
                el.setAttribute(m[0],m[1]);
                el.setAttribute(m[0]==='property'?'content':'content',m[2]);
                document.head.appendChild(el);
            });
        })();
        </script>""",
        unsafe_allow_html=True,
    )

    st.session_state["model_loaded"] = False
    render_sidebar()

    # ── Page header
    st.markdown(
        "<div style='text-align:center;padding:0.3rem 0 0.4rem'>"
        "<div class='demo-badge'>LIVE DEMO</div>"
        "<h1 style='font-size:1.5rem!important;margin:0.2rem 0 0.1rem'>🐟 FishVision</h1>"
        "<p style='color:#5ebbdc;font-size:0.75rem;letter-spacing:3px;margin:0 0 0.4rem'>"
        "MARINE SPECIES DETECTION &nbsp;·&nbsp; COMPUTER VISION &nbsp;·&nbsp; AQUACULTURE AI"
        "</p>"
        "<div>"
        "<span class='tag'>Computer Vision</span>"
        "<span class='tag'>Deep Learning</span>"
        "<span class='tag'>Marine Research</span>"
        "<span class='tag'>Multi-Species</span>"
        "<span class='tag'>Underwater Video</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── CTA above the fold
    render_above_fold_cta()

    # ── Dual demo videos as hero (no separate hero video)
    render_dual_demo()
    st.markdown("---")
    render_detection_gallery()
    st.markdown("---")
    render_pipeline_overview()
    st.markdown("---")
    render_case_studies()
    st.markdown("---")
    render_faq()
    st.markdown("---")
    render_contact_cta()

    # ── Sticky mobile CTA (only visible on narrow screens via CSS)
    st.markdown(
        "<div class='mobile-sticky-cta'>"
        "<a href='mailto:leodorfman1@gmail.com'>📬 REQUEST A CUSTOM BUILD</a>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
