# FishVision — Underwater Fish Detection Demo

A computer vision app that detects and counts fish in underwater video using a custom-trained YOLOv8 model.

Built by Leo Dorfman · [github.com/eodorfman111](https://github.com/eodorfman111)

## Setup (takes ~2 min)

```bash
pip install streamlit ultralytics opencv-python-headless plotly pandas reportlab torch
```

1. Drop your model weights into the `models/` folder next to `demo.py`
2. Run: `streamlit run demo.py`
3. Open browser at `http://localhost:8501`

## Deploy free (get a public URL)

1. Push this folder to a GitHub repo
2. Go to share.streamlit.io → New app → point to `demo.py`
3. If your model is >100MB, host it on HuggingFace Hub (free) and load via URL
4. You get a live URL like `leodorfman-fishvision.streamlit.app`

## Structure

```
fishvision_demo/
├── demo.py       ← main app
├── models/       ← drop your .pt weights here
└── README.md
```

## Resume bullet

> Built and deployed a public fish detection web app using a custom YOLOv8 model and Streamlit —
> processes underwater video, outputs annotated frames with fish counts, detection timeline charts,
> and PDF reports. Live demo: [your-url.streamlit.app]
