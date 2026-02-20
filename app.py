import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
from collections import Counter
import os

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="COVID CT Scan Search",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLLECTION_NAME = "covid_ct_scans"
EMBEDDING_DIM   = 1024
CLASSES         = ["Covid", "Healthy", "Others"]
QDRANT_PATH     = "./qdrant_storage"   # chemin local vers la base Qdrant exportée

CLASS_COLORS = {
    "Covid":   "#ef4444",
    "Healthy": "#22c55e",
    "Others":  "#f97316",
}

CLASS_ICONS = {
    "Covid":   "🦠",
    "Healthy": "✅",
    "Others":  "🔶",
}

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

.main { background-color: #0d1117; }

.hero {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}

.hero h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #58a6ff;
    margin: 0;
    letter-spacing: -1px;
}

.hero p {
    color: #8b949e;
    font-size: 1rem;
    margin-top: 0.5rem;
}

.stat-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.stat-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #58a6ff;
}

.stat-card .label {
    font-size: 0.8rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.2rem;
}

.result-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 0.8rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}

.result-card:hover { border-color: #58a6ff; }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}

.score-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 6px;
    margin-top: 6px;
}

.prediction-box {
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}

.prediction-box .pred-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
}

.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #21262d;
}

.stButton>button {
    background: #238636;
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    width: 100%;
    transition: background 0.2s;
}

.stButton>button:hover { background: #2ea043; }

.stSelectbox label, .stSlider label, .stFileUploader label {
    color: #8b949e !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

div[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}

.uploadedFile { background: #161b22 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL & QDRANT (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    class FeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            base = models.densenet121(weights="IMAGENET1K_V1")
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.features(x)
            x = torch.relu(x)
            x = self.pool(x)
            return torch.flatten(x, 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureExtractor().to(device)
    model.eval()
    return model, device


@st.cache_resource
def load_qdrant():
    return QdrantClient(path=QDRANT_PATH)


@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# ─────────────────────────────────────────────
#  SEARCH FUNCTION
# ─────────────────────────────────────────────
def search(image: Image.Image, k: int = 5, filter_class: str = None):
    model, device = load_model()
    transform = get_transform()
    qdrant = load_qdrant()

    t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(t).cpu().numpy()[0].tolist()

    query_filter = None
    if filter_class and filter_class != "Toutes les classes":
        query_filter = Filter(
            must=[FieldCondition(key="class_name", match=MatchValue(value=filter_class))]
        )

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=emb,
        limit=k,
        query_filter=query_filter,
        with_payload=True
    ).points

    return [{"path": r.payload["path"],
             "label": r.payload["class_name"],
             "score": round(r.score, 4)} for r in results]


def get_stats():
    qdrant = load_qdrant()
    info = qdrant.get_collection(COLLECTION_NAME)
    total = info.points_count
    stats = {}
    for cls in CLASSES:
        f = Filter(must=[FieldCondition(key="class_name", match=MatchValue(value=cls))])
        count = qdrant.count(collection_name=COLLECTION_NAME, count_filter=f).count
        stats[cls] = count
    return total, stats


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">⚙️ Paramètres</div>', unsafe_allow_html=True)

    k = st.slider("Nombre de résultats (k)", min_value=1, max_value=10, value=5)

    filter_class = st.selectbox(
        "Filtrer par classe",
        ["Toutes les classes"] + CLASSES
    )

    st.markdown("---")
    st.markdown('<div class="section-title">📊 Base de données</div>', unsafe_allow_html=True)

    try:
        total, stats = get_stats()
        st.markdown(f"""
        <div class="stat-card" style="margin-bottom:0.5rem">
            <div class="value">{total}</div>
            <div class="label">Images indexées</div>
        </div>
        """, unsafe_allow_html=True)
        for cls, count in stats.items():
            color = CLASS_COLORS.get(cls, "#58a6ff")
            icon = CLASS_ICONS.get(cls, "•")
            pct = round(count / total * 100, 1) if total > 0 else 0
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:0.4rem 0;border-bottom:1px solid #21262d;font-size:0.85rem">
                <span>{icon} <b style="color:{color}">{cls}</b></span>
                <span style="font-family:'IBM Plex Mono',monospace;color:#8b949e">{count} ({pct}%)</span>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.warning("Base Qdrant non trouvée.\nVérifiez le chemin `qdrant_storage`.")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#8b949e;text-align:center">
        DenseNet121 + Qdrant<br>COVID CT Scan Search Engine
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🫁 COVID CT Scan Search</h1>
    <p>Moteur de recherche d'images médicales par similarité visuelle</p>
</div>
""", unsafe_allow_html=True)

col_upload, col_results = st.columns([1, 2], gap="large")

with col_upload:
    st.markdown('<div class="section-title">📤 Image requête</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Uploader un CT scan", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Image uploadée", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🔍 Lancer la recherche")

        if run:
            with st.spinner("Extraction des features..."):
                results = search(img, k=k, filter_class=filter_class)

            # Prédiction par vote majoritaire
            pred = Counter([r["label"] for r in results]).most_common(1)[0][0]
            color = CLASS_COLORS.get(pred, "#58a6ff")
            icon = CLASS_ICONS.get(pred, "")

            st.markdown(f"""
            <div class="prediction-box" style="background:{color}22;border:2px solid {color}">
                <div style="font-size:0.75rem;color:{color};text-transform:uppercase;
                            letter-spacing:2px;margin-bottom:0.3rem">Classe prédite</div>
                <div class="pred-label" style="color:{color}">{icon} {pred}</div>
            </div>
            """, unsafe_allow_html=True)

            st.session_state["results"] = results
            st.session_state["pred"] = pred

with col_results:
    st.markdown('<div class="section-title">🔎 Résultats similaires</div>', unsafe_allow_html=True)

    if "results" in st.session_state and st.session_state["results"]:
        results = st.session_state["results"]

        for i, r in enumerate(results):
            label = r["label"]
            score = r["score"]
            color = CLASS_COLORS.get(label, "#58a6ff")
            icon  = CLASS_ICONS.get(label, "")
            pct   = int(score * 100)

            img_col, info_col = st.columns([1, 2])
            with img_col:
                try:
                    st.image(Image.open(r["path"]), use_container_width=True)
                except Exception:
                    st.error("Image non trouvée")

            with info_col:
                st.markdown(f"""
                <div style="padding:0.5rem 0">
                    <div style="font-size:0.75rem;color:#8b949e;margin-bottom:4px">
                        #{i+1} — {os.path.basename(r['path'])}
                    </div>
                    <span class="badge" style="background:{color}22;color:{color}">
                        {icon} {label}
                    </span>
                    <div style="margin-top:0.6rem;font-family:'IBM Plex Mono',monospace">
                        <span style="color:#8b949e;font-size:0.8rem">Score : </span>
                        <span style="color:{color};font-size:1rem;font-weight:600">{score}</span>
                    </div>
                    <div class="score-bar-bg">
                        <div style="background:{color};width:{pct}%;height:6px;border-radius:4px"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<hr style='border-color:#21262d;margin:0.5rem 0'>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#8b949e;
                    border:1px dashed #21262d;border-radius:12px">
            <div style="font-size:3rem;margin-bottom:1rem">🫁</div>
            <div>Upload une image CT scan et lance la recherche</div>
        </div>
        """, unsafe_allow_html=True)
