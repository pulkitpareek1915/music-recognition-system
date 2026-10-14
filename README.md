# 🎵 Music Recognition System

A deep learning system that identifies the **artist** of a music track from raw audio. It extracts a rich set of audio features with Librosa, trains a regularized deep neural network (TensorFlow/Keras) on tracks from the Free Music Archive (FMA) dataset, and includes a Tkinter desktop GUI for predicting the artist of any audio file with confidence scores.

## Features

- Audio feature extraction pipeline (540 features per track) using Librosa
- Handles FMA metadata with both single and multi-level CSV headers
- Automatic filtering of rare artists and stratified train/test split
- Class-weight balancing to handle imbalanced artist classes
- Deep Neural Network with Batch Normalization, Dropout and L2 regularization
- Early stopping and learning-rate scheduling during training
- Top-5 artist predictions with confidence percentages
- Save and load model, scaler and label encoder for reuse
- Desktop GUI (Tkinter) for selecting a file and viewing predictions, with threaded inference so the UI stays responsive

## Project Structure

```
music-recognition-system/
├── music_calssification_final_model.py   # Feature extraction, training, prediction, GUI
├── fma_features_with_embeddings1.csv     # Pre-extracted features (998 tracks, 249 artists)
└── README.md
```

## Tech Stack

- **Language:** Python 3
- **Deep Learning:** TensorFlow, Keras
- **Audio Processing:** Librosa
- **Machine Learning:** Scikit-learn (StandardScaler, LabelEncoder, train_test_split, class weights)
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib
- **GUI:** Tkinter
- **Dataset:** Free Music Archive (FMA)

## How It Works

1. **Load metadata**: read the FMA `tracks.csv` to map each track ID to its title, artist and album.
2. **Extract features**: for each 30-second clip (22,050 Hz), compute and pool (mean and standard deviation) the following into one 540-dimensional vector:
   - 40 MFCCs plus delta and delta-delta
   - 128-band log-mel spectrogram
   - Spectral contrast
   - Chroma
   - Spectral centroid, rolloff and zero-crossing rate
3. **Prepare data**: drop artists with too few tracks, encode artist labels, standardize features, compute balanced class weights, and split 80/20 (stratified).
4. **Train**: a DNN with three hidden layers (512, 256, 128 units), each followed by Batch Normalization and Dropout, and a softmax output over artists. Optimized with Adam, early stopping and ReduceLROnPlateau.
5. **Predict**: extract features from a new audio file, scale them, and return the top-5 artists with confidence scores.
6. **GUI**: load the saved model and predict from a file picker.

## Getting Started

```bash
git clone https://github.com/pulkitpareek1915/music-recognition-system.git
cd music-recognition-system
pip install tensorflow librosa numpy pandas scikit-learn matplotlib tqdm
```

1. Download the [FMA dataset](https://github.com/mdeff/fma) (`fma_metadata` and `fma_medium`).
2. Update the path variables at the top of `music_calssification_final_model.py` (`METADATA_PATH`, `AUDIO_DIR`, `FEATURES_CSV`, `MODEL_PATH`, etc.).
3. Run the pipeline:

```python
from music_calssification_final_model import *

metadata_df = load_fma_metadata(METADATA_PATH)
features_df = process_fma_dataset(AUDIO_DIR, metadata_df, FEATURES_CSV)

X_train, X_test, y_train, y_test, scaler, label_encoder, class_weight = \
    prepare_data_for_training(features_df, min_samples_per_artist=2)

model, history = train_model(X_train, y_train, X_test, y_test,
                             len(label_encoder.classes_), class_weight=class_weight)

save_model_and_preprocessing(model, scaler, label_encoder)
launch_gui()
```

To skip feature extraction, load the included `fma_features_with_embeddings1.csv` and continue from the data preparation step.

## Future Improvements

- Use pretrained audio embeddings (YAMNet / VGGish) or CNNs on mel-spectrograms
- Add genre classification and similar-track recommendation
- Train on a larger subset of FMA for more artists and tracks per artist
- Add evaluation reporting (confusion matrix, top-k accuracy)
- Build a web interface with Streamlit or Flask

## Author

**Pulkit Pareek**
GitHub: [@pulkitpareek1915](https://github.com/pulkitpareek1915)
