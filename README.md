# Deepfake Audio Detection

A machine learning-based system for detecting whether a speech recording is **Genuine (Human)** or **Deepfake (AI-Generated)**.

## Features
- Detects genuine and AI-generated speech
- Supports WAV, MP3, and OGG audio files
- Displays prediction confidence scores
- Interactive Streamlit web application
- Waveform and spectrogram visualization

## Model
**Augmented Generalized LightGBM**

### Feature Extraction
- MFCCs
- Delta & Delta-Delta MFCCs
- Chroma Features
- Spectral Features
- Zero Crossing Rate
- RMS Energy

**Total: 137 acoustic features**

## Datasets
- FOR (Fake-or-Real) Dataset
- ASVspoof 2019 LA Dataset

## Results

| Dataset | Accuracy | EER |
|----------|----------:|----------:|
| FOR Test Set | 92.73% | 7.47% |
| ASVspoof 2019 LA | 88.00% | 11.00% |

## Tech Stack
- Python
- Librosa
- LightGBM
- Scikit-learn
- Streamlit

## Live Demo
https://deepfake-audio-detection-kngjwlj7exmmhasn5yevdd.streamlit.app/
