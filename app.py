

# import streamlit as st
# import numpy as np
# import librosa
# import joblib
# import tempfile

# MODEL_PATH = "deepfake_audio_detector_lgbm_37.pkl"

# model = joblib.load(MODEL_PATH)

# st.title("🎧 Deepfake Audio Detection")
# st.write("Upload an audio file to check whether it is Genuine or Deepfake.")

# def extract_features(file_path):
#     audio, sr = librosa.load(file_path, sr=None)

#     if len(audio) == 0:
#         audio = np.zeros(2048)

#     if len(audio) < 2048:
#         audio = np.pad(audio, (0, 2048 - len(audio)))

#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

#     if mfcc.shape[1] >= 9:
#         delta = librosa.feature.delta(mfcc)
#         delta2 = librosa.feature.delta(mfcc, order=2)
#     else:
#         delta = np.zeros_like(mfcc)
#         delta2 = np.zeros_like(mfcc)

#     try:
#         chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
#     except:
#         chroma = np.zeros((12, 1))

#     spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
#     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
#     spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
#     zcr = librosa.feature.zero_crossing_rate(audio)
#     rms = librosa.feature.rms(y=audio)

#     features = np.hstack([
#         np.mean(mfcc, axis=1),
#         np.mean(delta, axis=1),
#         np.mean(delta2, axis=1),
#         np.mean(chroma, axis=1),
#         np.mean(spectral_centroid),
#         np.mean(spectral_bandwidth),
#         np.mean(spectral_rolloff),
#         np.mean(zcr),
#         np.mean(rms)
#     ])

#     return features

# uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "flac", "ogg"])

# if uploaded_file is not None:
#     st.audio(uploaded_file)

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
#         temp_audio.write(uploaded_file.read())
#         temp_path = temp_audio.name

#     features = extract_features(temp_path).reshape(1, -1)

#     prediction = model.predict(features)[0]
#     probabilities = model.predict_proba(features)[0]
#     confidence = np.max(probabilities) * 100

#     if prediction == 0:
#         st.success("Prediction: GENUINE / HUMAN AUDIO")
#     else:
#         st.error("Prediction: DEEPFAKE / AI-GENERATED AUDIO")

#     st.metric("Confidence Score", f"{confidence:.2f}%")

#     st.write("Class Probabilities:")
#     st.write({
#         "Genuine": f"{probabilities[0] * 100:.2f}%",
#         "Deepfake": f"{probabilities[1] * 100:.2f}%"
#     })



import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import tempfile

MODEL_PATH = "deepfake_audio_detector_lgbm_37.pkl"

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 Deepfake Audio Detection System")

st.markdown("""
This system detects whether an uploaded speech recording is:

- **GENUINE (Human Speech)**
- **DEEPFAKE (AI-Generated Speech)**

### Model Information

- Model: LightGBM
- Features: 137 Acoustic Features
- Test Accuracy: **91.11%**
- Test EER: **8.78%**
""")

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=None)

    if len(audio) == 0:
        audio = np.zeros(2048)

    if len(audio) < 2048:
        audio = np.pad(audio, (0, 2048 - len(audio)))

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    if mfcc.shape[1] >= 9:
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
    else:
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)

    try:
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sr
        )
    except Exception:
        chroma = np.zeros((12, 1))

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    zcr = librosa.feature.zero_crossing_rate(audio)

    rms = librosa.feature.rms(y=audio)

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.mean(delta2, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.mean(spectral_rolloff),
        np.mean(zcr),
        np.mean(rms)
    ])

    return features

uploaded_file = st.file_uploader(
    "Upload Speech Audio",
    type=["wav", "mp3"]
)

if uploaded_file:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    audio, sr = librosa.load(temp_path, sr=None)

    features = extract_features(temp_path)
    features = features.reshape(1, -1)

    probs = model.predict_proba(features)[0]

    prediction = np.argmax(probs)

    confidence = np.max(probs) * 100

    st.subheader("Prediction")

    if prediction == 0:
        st.success("✅ GENUINE AUDIO")
    else:
        st.error("🚨 DEEPFAKE AUDIO")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    with col2:
        st.metric(
            "Genuine Probability",
            f"{probs[0] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Deepfake Probability",
            f"{probs[1] * 100:.2f}%"
        )

    if confidence < 60:
        st.warning(
            "Low confidence prediction. Audio may be noisy, compressed, or outside the training distribution."
        )

    st.subheader("Waveform")

    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    st.pyplot(fig)

    st.subheader("Spectrogram")

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio)),
        ref=np.max
    )

    fig2, ax2 = plt.subplots(figsize=(10, 4))

    img = librosa.display.specshow(
        D,
        sr=sr,
        x_axis="time",
        y_axis="log",
        ax=ax2
    )

    fig2.colorbar(
        img,
        ax=ax2,
        format="%+2.0f dB"
    )

    st.pyplot(fig2)


