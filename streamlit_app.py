import streamlit as st
import numpy as np
import librosa
import joblib

SR = 16000
CLIP = SR

st.set_page_config(page_title="Poultry Sound Health Checker", page_icon="🐔")
st.title("🐔 Poultry Sound Health Checker")
st.write(
    "Upload a short poultry sound (.wav or .mp3). The model estimates Healthy, "
    "Unhealthy, or Noise. Research demo trained on one small public dataset - "
    "not a veterinary tool."
)

@st.cache_resource
def load_model():
    return joblib.load("final_svm.joblib")

model = load_model()
classes = list(model.classes_)

def feats(seg):
    mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=40)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "ogg", "flac"])

if uploaded is not None:
    st.audio(uploaded)
    y_, _ = librosa.load(uploaded, sr=SR, duration=60)
    if len(y_) < CLIP:
        y_ = np.pad(y_, (0, CLIP - len(y_)))
    starts = list(range(0, len(y_) - CLIP + 1, CLIP // 2))
    X_new = np.array([feats(y_[s:s + CLIP]) for s in starts])
    p = model.predict_proba(X_new).mean(axis=0)
    best = classes[int(np.argmax(p))]
    st.subheader(f"Result: {best}")
    for c, v in zip(classes, p):
        st.write(f"{c}: {v:.1%}")
        st.progress(float(v))
