import streamlit as st
import cv2
import numpy as np
import joblib
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import io
import base64

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScratchScan · AI Defect Detection",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject global CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* ── Reset & base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #07072D !important;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], footer { display: none !important; }
[data-testid="stAppViewContainer"] > section:first-child { padding: 0 !important; }
section.main > div { padding-top: 0 !important; }
[data-testid="stSidebar"] { background: #0d1526 !important; border-right: 1px solid rgba(255,255,255,0.07); }

/* ── Navigation pill ── */
.nav-pill {
    position: sticky; top: 16px; z-index: 999;
    margin: 16px auto 0; max-width: 780px;
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 20px;
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 9999px;
}
.nav-logo { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 15px;
    letter-spacing: 0.08em; text-transform: uppercase;
    background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.nav-links { display: flex; gap: 28px; font-size: 13px; color: #64748b; }
.nav-badge {
    font-size: 11px; font-family: 'DM Mono', monospace; letter-spacing: 0.06em;
    padding: 5px 14px; border-radius: 9999px;
    background: #fff; color: #020617; font-weight: 500; }

/* ── Hero ── */
.hero-wrap {
    position: relative; overflow: hidden;
    padding: 80px 40px 60px; text-align: center;
}
.hero-portal {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    width: 700px; height: 700px; border-radius: 50%;
    background: radial-gradient(circle at center,
        rgba(56, 189, 248, 0.06) 0%,
        rgba(15, 23, 42, 0.3) 40%,
        transparent 70%);
    pointer-events: none; animation: pulse 8s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.6; }
    50% { transform: translate(-50%, -50%) scale(1.08); opacity: 1; }
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: 0.2em;
    text-transform: uppercase; color: #38bdf8; margin-bottom: 20px;
    display: inline-block; padding: 4px 12px;
    border: 1px solid rgba(56,189,248,0.25); border-radius: 4px;
}
.hero-h1 {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: clamp(48px, 7vw, 82px); line-height: 1.0;
    letter-spacing: -0.04em;
    background: linear-gradient(180deg, #ffffff 0%, #94a3b8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 auto 20px; max-width: 700px;
}
.hero-sub {
    font-size: 16px; color: #64748b; max-width: 480px;
    margin: 0 auto 36px; line-height: 1.7;
}
.stat-row {
    display: flex; gap: 32px; justify-content: center;
    flex-wrap: wrap; margin-top: 40px;
}
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 16px 28px; text-align: center;
    min-width: 130px;
}
.stat-num {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 28px;
    background: linear-gradient(135deg, #fff 0%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-lbl { font-size: 11px; color: #475569; letter-spacing: 0.1em;
    text-transform: uppercase; margin-top: 2px; font-family: 'DM Mono', monospace; }

/* ── Divider ── */
.hr { height: 1px; background: rgba(255,255,255,0.06); margin: 48px 0; }

/* ── Section title ── */
.sec-title {
    font-family: 'Syne', sans-serif; font-weight: 700; font-size: 13px;
    letter-spacing: 0.2em; text-transform: uppercase; color: #38bdf8;
    margin-bottom: 24px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(255,255,255,0.12) !important;
    border-radius: 16px !important; padding: 32px !important;
}
[data-testid="stFileUploader"] label { color: #94a3b8 !important; }
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 20px; padding: 32px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(20px); margin-top: 24px;
}
.result-good { background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(2,6,23,0.6) 100%); border-color: rgba(16,185,129,0.25); }
.result-bad  { background: linear-gradient(135deg, rgba(239,68,68,0.08) 0%, rgba(2,6,23,0.6) 100%); border-color: rgba(239,68,68,0.25); }
.result-label {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 36px; letter-spacing: -0.03em; margin-bottom: 4px;
}
.result-good .result-label { color: #10b981; }
.result-bad  .result-label { color: #ef4444; }
.result-verdict { font-size: 13px; color: #64748b; font-family: 'DM Mono', monospace; }
.conf-bar-wrap { margin-top: 20px; }
.conf-label { font-size: 11px; color: #475569; font-family: 'DM Mono', monospace;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }
.conf-bar-bg { height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 3px; transition: width 0.8s cubic-bezier(0.34,1.56,0.64,1); }

/* ── Feature pill row ── */
.feat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 20px; }
.feat-pill {
    font-family: 'DM Mono', monospace; font-size: 11px;
    padding: 6px 14px; border-radius: 6px;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
    color: #94a3b8; letter-spacing: 0.04em;
}
.feat-pill span { color: #38bdf8; margin-left: 4px; }

/* ── Pipeline stage cards ── */
.stage-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-top: 16px; }
.stage-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 14px 12px; text-align: center;
}
.stage-num { font-family: 'DM Mono', monospace; font-size: 10px;
    color: #38bdf8; letter-spacing: 0.15em; margin-bottom: 6px; }
.stage-name { font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 13px; color: #e2e8f0; margin-bottom: 3px; }
.stage-desc { font-size: 11px; color: #475569; }

/* ── Bento grid ── */
.bento { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 16px; }
.bento-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 24px;
}
.bento-card.wide { grid-column: span 2; }
.bento-icon { font-size: 20px; margin-bottom: 10px; }
.bento-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 15px;
    color: #e2e8f0; margin-bottom: 4px; }
.bento-desc { font-size: 12px; color: #475569; line-height: 1.6; }

/* ── Footer ── */
.footer {
    text-align: center; padding: 40px 20px;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-family: 'DM Mono', monospace; font-size: 11px; color: #334155;
    letter-spacing: 0.08em; margin-top: 60px;
}

/* ── Matplotlib dark bg ── */
.stPlotlyChart, [data-testid="stImage"] img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── Constants (mirror predict.py) ────────────────────────────────────────────
RESIZE = (200, 200)
BLUR_KERNEL = 5
CANNY_LOW = 10
CANNY_HIGH = 70
DILATE_KERNEL = (3, 3)
DILATE_ITERATIONS = 1
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0]
MODEL_PATH = "svm_model.pkl"
SCALER_PATH = "scaler.pkl"

# ── Core pipeline functions ───────────────────────────────────────────────────
def preprocess(img_array: np.ndarray):
    gray    = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray    = cv2.resize(gray, RESIZE)
    blurred = cv2.medianBlur(gray, BLUR_KERNEL)
    edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    dilated = cv2.dilate(edges, np.ones(DILATE_KERNEL, np.uint8),
                         iterations=DILATE_ITERATIONS)
    return gray, blurred, edges, dilated

def extract_features(gray, dilated):
    edge_density = np.count_nonzero(dilated) / dilated.size
    contours, _  = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    max_ar = 1.0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 0 and h > 0:
            ar = max(h / w, w / h)
            if ar > max_ar:
                max_ar = ar
    glcm        = graycomatrix(gray, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                               levels=256, symmetric=True, normed=True)
    contrast    = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return np.array([[edge_density, max_ar, contrast, homogeneity]])

def make_pipeline_figure(orig, blurred, edges, dilated):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    fig.patch.set_facecolor('#020617')
    titles  = ['Original',   'Median Blur', 'Canny Edges', 'Dilated']
    images  = [orig,          blurred,       edges,         dilated]
    cmaps   = ['gray',        'gray',        'hot',         'hot']
    borders = ['#38bdf8',    '#64748b',     '#f59e0b',     '#ef4444']
    for ax, title, img, cmap, bc in zip(axes, titles, images, cmaps, borders):
        ax.imshow(img, cmap=cmap, interpolation='nearest')
        ax.set_title(title, color='#94a3b8', fontsize=10,
                     fontfamily='monospace', pad=8)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(bc); spine.set_linewidth(1.5); spine.set_visible(True)
    plt.tight_layout(pad=0.8)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, facecolor='#020617',
                bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# ── Navigation ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-pill">
  <span class="nav-logo">⬡ ScratchScan</span>
  <div class="nav-links">
    <span>Inspect</span>
    <span>Pipeline</span>
    <span>About</span>
  </div>
  <span class="nav-badge">NEU · 98% ACC</span>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-portal"></div>
  <div class="hero-eyebrow">Computer Vision · UCS532P</div>
  <h1 class="hero-h1">Surface<br/>Intelligence</h1>
  <p class="hero-sub">Upload a metal surface image. The RBF-SVM model inspects
  it in milliseconds — returning a verdict, confidence score, and full
  preprocessing breakdown.</p>
  <div class="stat-row">
    <div class="stat-card"><div class="stat-num">98%</div><div class="stat-lbl">Accuracy</div></div>
    <div class="stat-card"><div class="stat-num">661</div><div class="stat-lbl">Train samples</div></div>
    <div class="stat-card"><div class="stat-num">4D</div><div class="stat-lbl">Feature vector</div></div>
    <div class="stat-card"><div class="stat-num">RBF</div><div class="stat-lbl">SVM kernel</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ── Upload + Predict ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-title">— Inspect a surface</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop a metal surface image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

models_present = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)

if not models_present:
    st.markdown("""
    <div style="margin:20px 0; padding:18px 24px; border-radius:12px;
         background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.25);
         font-family:'DM Mono',monospace; font-size:12px; color:#f59e0b;">
    ⚠ &nbsp; <strong>svm_model.pkl</strong> and <strong>scaler.pkl</strong> not found in the working
    directory. Run <code>python train_svm.py</code> first, then relaunch the app from the
    same folder.
    </div>
    """, unsafe_allow_html=True)

if uploaded is not None:
    # Decode image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Layout: image | result
    col_img, col_res = st.columns([1, 1.4], gap="large")

    with col_img:
        st.image(img_rgb, caption="Uploaded image", use_container_width=True)

    with col_res:
        if models_present:
            model  = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)

            gray, blurred, edges, dilated = preprocess(img_rgb)
            feats     = extract_features(gray, dilated)
            feats_sc  = scaler.transform(feats)
            pred      = model.predict(feats_sc)[0]
            proba     = model.predict_proba(feats_sc)[0]

            is_good       = pred == 0
            card_cls      = "result-good" if is_good else "result-bad"
            label_txt     = "GOOD SURFACE" if is_good else "DEFECTIVE"
            verdict_txt   = "No scratch patterns detected." if is_good else "Scratch / surface defect detected."
            conf_good_pct = f"{proba[0]*100:.1f}"
            conf_bad_pct  = f"{proba[1]*100:.1f}"
            conf_main_w   = f"{max(proba)*100:.1f}%"
            bar_color     = "#10b981" if is_good else "#ef4444"
            bar_w_good    = f"{proba[0]*100:.1f}%"
            bar_w_bad     = f"{proba[1]*100:.1f}%"

            # Feature values
            ed  = f"{feats[0,0]:.5f}"
            mar = f"{feats[0,1]:.3f}"
            con = f"{feats[0,2]:.3f}"
            hom = f"{feats[0,3]:.5f}"

            st.markdown(f"""
            <div class="result-card {card_cls}">
              <div class="result-label">{label_txt}</div>
              <div class="result-verdict">{verdict_txt}</div>

              <div class="conf-bar-wrap">
                <div class="conf-label">Good confidence</div>
                <div class="conf-bar-bg">
                  <div class="conf-bar-fill" style="width:{bar_w_good};background:#10b981;"></div>
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:11px;color:#64748b;
                     margin-top:4px;">{conf_good_pct}%</div>
              </div>

              <div class="conf-bar-wrap" style="margin-top:12px;">
                <div class="conf-label">Defective confidence</div>
                <div class="conf-bar-bg">
                  <div class="conf-bar-fill" style="width:{bar_w_bad};background:#ef4444;"></div>
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:11px;color:#64748b;
                     margin-top:4px;">{conf_bad_pct}%</div>
              </div>

              <div class="feat-row">
                <div class="feat-pill">edge_density<span>{ed}</span></div>
                <div class="feat-pill">max_ar<span>{mar}</span></div>
                <div class="feat-pill">contrast<span>{con}</span></div>
                <div class="feat-pill">homogeneity<span>{hom}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Load model artifacts to run inference.")

    # ── Pipeline visualisation ────────────────────────────────────────────
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">— Preprocessing pipeline</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="stage-grid">
      <div class="stage-card">
        <div class="stage-num">01 / GRAYSCALE</div>
        <div class="stage-name">Original</div>
        <div class="stage-desc">Resize → 200×200 · BGR→GRAY</div>
      </div>
      <div class="stage-card">
        <div class="stage-num">02 / DENOISE</div>
        <div class="stage-name">Median blur</div>
        <div class="stage-desc">kernel = 5 · salt-pepper removal</div>
      </div>
      <div class="stage-card">
        <div class="stage-num">03 / EDGES</div>
        <div class="stage-name">Canny</div>
        <div class="stage-desc">low = 10 · high = 70</div>
      </div>
      <div class="stage-card">
        <div class="stage-num">04 / MORPHOLOGY</div>
        <div class="stage-name">Dilation</div>
        <div class="stage-desc">3×3 kernel · 1 iteration</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    buf = make_pipeline_figure(gray, blurred, edges, dilated)
    st.image(buf, use_container_width=True)

# ── About / Bento ─────────────────────────────────────────────────────────────
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">— Model architecture</div>', unsafe_allow_html=True)

st.markdown("""
<div class="bento">
  <div class="bento-card">
    <div class="bento-title">Handcrafted feature vector</div>
    <div class="bento-desc">Four mathematically derived features — edge density,
    max contour aspect ratio, GLCM contrast, and GLCM homogeneity — encode
    both structural and textural properties of the surface.</div>
  </div>
  <div class="bento-card">
    <div class="bento-title">RBF-SVM classifier</div>
    <div class="bento-desc">Radial Basis Function kernel with balanced class
    weights and probability calibration. StandardScaler normalises the
    feature space before inference.</div>
  </div>
  <div class="bento-card">
    <div class="bento-title">NEU Surface Defect Dataset</div>
    <div class="bento-desc">314 metal surface images split 80 / 20.
    Stratified sampling preserves class ratio. RANDOM_STATE = 42
    guarantees reproducible splits.</div>
  </div>
  <div class="bento-card">
    <div class="bento-title">Confusion matrix results</div>
    <div class="bento-desc">15 true negatives · 47 true positives · 0 false
    positives · 1 false negative → 98.4% overall accuracy on 63 test
    samples.</div>
  </div>
  <div class="bento-card wide">
    <div class="bento-title">End-to-end pipeline</div>
    <div class="bento-desc" style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:12px;">
      <div style="text-align:center;font-family:'DM Mono',monospace;font-size:11px;color:#64748b;">
        <div style="font-size:18px;margin-bottom:4px;">📁</div>extract_features.py<br/>
        <span style="color:#38bdf8;">→ features.csv</span>
      </div>
      <div style="text-align:center;font-size:18px;color:#334155;padding-top:12px;">→</div>
      <div style="text-align:center;font-family:'DM Mono',monospace;font-size:11px;color:#64748b;">
        <div style="font-size:18px;margin-bottom:4px;">🤖</div>train_svm.py<br/>
        <span style="color:#38bdf8;">→ .pkl artifacts</span>
      </div>
      <div style="text-align:center;font-size:18px;color:#334155;padding-top:12px;">→</div>
      <div style="text-align:center;font-family:'DM Mono',monospace;font-size:11px;color:#64748b;">
        <div style="font-size:18px;margin-bottom:4px;">🔬</div>app.py (this)<br/>
        <span style="color:#38bdf8;">→ live inference</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  UCS532P · Thapar Institute of Engineering and Technology · 2024–25 &nbsp;·&nbsp;
  RBF-SVM · OpenCV · scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
