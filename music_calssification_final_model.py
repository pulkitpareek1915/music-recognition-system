# music_recognition.py
import os
import librosa
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ML libs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

# GUI libs (optional)
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

print("="*70)
print("CHECKING INSTALLED LIBRARIES...")
print("="*70)
print(f"✓ TensorFlow Version: {tf.__version__}")
print(f"✓ Librosa Version: {librosa.__version__}")
print(f"✓ NumPy Version: {np.__version__}")
print(f"✓ Pandas Version: {pd.__version__}")
print("="*70)
print("All libraries loaded successfully!")
print("="*70)

# ---------------- PATHS (update if needed) ----------------
BASE_DIR = r"A:\pulkit mini project\archive"
METADATA_PATH = r"A:\pulkit mini project\archive\fma_metadata\tracks.csv"
AUDIO_DIR = r"A:\pulkit mini project\archive\fma_medium\fma_medium"  # your chosen folder
FEATURES_CSV = r"A:\fma_features_with_embeddings2.csv"
MODEL_PATH = r"A:\music_recognition_model.h5"
SCALER_PATH = r"A:\scaler.pkl"
ENCODER_PATH = r"A:\label_encoder.pkl"
# --------------------------------------------------------

def check_paths():
    print("\n" + "="*70)
    print("CHECKING FILE PATHS...")
    print("="*70)
    paths_to_check = {
        "Metadata CSV": METADATA_PATH,
        "Audio Directory": AUDIO_DIR
    }
    all_exist = True
    for name, path in paths_to_check.items():
        exists = os.path.exists(path)
        status = "✓ FOUND" if exists else "✗ NOT FOUND"
        print(f"{status} - {name}: {path}")
        if not exists:
            all_exist = False
    if not all_exist:
        print("\n⚠ WARNING: Some paths are missing! Update the paths at the top of the script.")
    else:
        print("\n✓ All paths are valid!")
    print("="*70)
    return all_exist

check_paths()

# ------------------ Metadata loading ------------------
def inspect_fma_metadata(metadata_path):
    print("Inspecting FMA metadata structure...")
    try:
        tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
        print(f"\n✓ Multi-level header detected")
        return tracks, True
    except Exception:
        tracks = pd.read_csv(metadata_path, index_col=0)
        print(f"\n✓ Single-level header detected")
        return tracks, False

def load_fma_metadata(metadata_path):
    print("Loading FMA metadata...")
    tracks, is_multilevel = inspect_fma_metadata(metadata_path)
    if is_multilevel:
        top_level_cols = tracks.columns.get_level_values(0).unique().tolist()
        metadata_dict = {'track_id': tracks.index}
        if 'track' in top_level_cols:
            for title_col in ['title', 'name', 'track_title']:
                if ('track', title_col) in tracks.columns:
                    metadata_dict['title'] = tracks[('track', title_col)].values
                    break
        if 'artist' in top_level_cols:
            for artist_col in ['name', 'title', 'artist_name']:
                if ('artist', artist_col) in tracks.columns:
                    metadata_dict['artist'] = tracks[('artist', artist_col)].values
                    break
        if 'album' in top_level_cols:
            for album_col in ['title', 'name', 'album_title']:
                if ('album', album_col) in tracks.columns:
                    metadata_dict['album'] = tracks[('album', album_col)].values
                    break
        if 'title' not in metadata_dict:
            metadata_dict['title'] = [f"Track_{i}" for i in tracks.index]
            print("\n⚠ Warning: 'title' column not found, using track IDs")
        if 'artist' not in metadata_dict:
            metadata_dict['artist'] = ['Unknown Artist'] * len(tracks)
            print("\n⚠ Warning: 'artist' column not found, using 'Unknown Artist'")
        if 'album' not in metadata_dict:
            metadata_dict['album'] = ['Unknown Album'] * len(tracks)
            print("\n⚠ Warning: 'album' column not found, using 'Unknown Album'")
        metadata_df = pd.DataFrame(metadata_dict)
    else:
        metadata_dict = {'track_id': tracks.index}
        title_cols = [col for col in tracks.columns if 'title' in col.lower()]
        if title_cols:
            metadata_dict['title'] = tracks[title_cols[0]].values
        else:
            metadata_dict['title'] = [f"Track_{i}" for i in tracks.index]
        artist_cols = [col for col in tracks.columns if 'artist' in col.lower()]
        if artist_cols:
            metadata_dict['artist'] = tracks[artist_cols[0]].values
        else:
            metadata_dict['artist'] = ['Unknown Artist'] * len(tracks)
        album_cols = [col for col in tracks.columns if 'album' in col.lower()]
        if album_cols:
            metadata_dict['album'] = tracks[album_cols[0]].values
        else:
            metadata_dict['album'] = ['Unknown Album'] * len(tracks)
        metadata_df = pd.DataFrame(metadata_dict)
    print(f"\n✓ Successfully loaded {len(metadata_df)} tracks")
    print(f"\nSample data:")
    print(metadata_df.head())
    return metadata_df

# ------------------ Feature extraction (improved) ------------------
def extract_mfcc_features(audio_path, n_mfcc=40, duration=30, sr=22050, hop_length=512, n_mels=128):
    """
    Returns a 1D feature vector combining:
    - MFCC mean/std
    - delta MFCC mean/std
    - delta2 MFCC mean/std
    - log-mel mean/std
    - spectral contrast mean/std
    - chroma mean/std
    - spectral centroid/rolloff/zcr mean/std
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        if len(y) < 2048:
            y = np.pad(y, (0, max(0, 2048 - len(y))), mode='constant')

        # MFCCs + deltas
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=2048)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        # Mel spectrogram (log)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=2048)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Spectral contrast
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=2048, hop_length=hop_length)

        # Chroma and other spectral features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=2048)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)

        # Pooling helper
        def pool(feat):
            return np.concatenate([np.mean(feat, axis=1), np.std(feat, axis=1)])

        mfcc_feat = pool(mfccs)            # 2*n_mfcc
        mfcc_d_feat = pool(mfcc_delta)
        mfcc_d2_feat = pool(mfcc_delta2)
        logmel_feat = pool(log_mel)        # 2*n_mels
        contrast_feat = pool(spec_contrast)
        chroma_feat = pool(chroma)
        centroid_feat = np.array([np.mean(spectral_centroid), np.std(spectral_centroid)])
        rolloff_feat = np.array([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        zcr_feat = np.array([np.mean(zcr), np.std(zcr)])

        feature_vector = np.concatenate([
            mfcc_feat, mfcc_d_feat, mfcc_d2_feat,
            logmel_feat,
            contrast_feat,
            chroma_feat,
            centroid_feat, rolloff_feat, zcr_feat
        ])

        return feature_vector
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# ------------------ Dataset processing ------------------
def process_fma_dataset(audio_dir, metadata_df, output_csv, sample_limit=None):
    print(f"\nProcessing audio files from: {audio_dir}")
    features_list = []
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                audio_files.append(os.path.join(root, file))
    if sample_limit:
        audio_files = audio_files[:sample_limit]
    print(f"Found {len(audio_files)} audio files to process")
    for audio_path in tqdm(audio_files, desc="Extracting features"):
        filename = os.path.basename(audio_path)
        name_no_ext = os.path.splitext(filename)[0]
        try:
            track_id = int(name_no_ext)
        except Exception:
            # skip non-conforming filenames
            continue
        track_info = metadata_df[metadata_df['track_id'] == track_id]
        if len(track_info) == 0:
            continue
        features = extract_mfcc_features(audio_path)
        if features is not None:
            feature_dict = {
                'track_id': track_id,
                'title': track_info['title'].values[0],
                'artist': track_info['artist'].values[0],
                'album': track_info['album'].values[0] if 'album' in track_info.columns else 'Unknown',
                'features': features.tolist()
            }
            features_list.append(feature_dict)
    features_df = pd.DataFrame(features_list)
    features_df.to_csv(output_csv, index=False)
    print(f"\nFeatures saved to {output_csv}")
    print(f"Processed {len(features_df)} tracks successfully")
    return features_df

# ------------------ Prepare data (class weights) ------------------
from sklearn.utils.class_weight import compute_class_weight

def prepare_data_for_training(features_df, min_samples_per_artist=2, apply_pca=False, pca_components=128):
    """
    Prepares data and handles rare classes automatically.
    - If min_samples_per_artist >= 2 (recommended), artists with fewer songs are removed so stratified split works.
    - If min_samples_per_artist == 1 and any class has only 1 sample, function will fall back to non-stratified split.
    Returns: X_train, X_test, y_train, y_test, scaler, label_encoder, class_weight_dict
    """
    print("\nPreparing data for training...")

    # Convert feature column -> numpy arrays
    X_list = []
    for f in features_df['features']:
        if isinstance(f, str):
            arr = np.array(eval(f), dtype=np.float32)
        else:
            arr = np.array(f, dtype=np.float32)
        X_list.append(arr)
    X_all = np.vstack([x.reshape(1, -1) for x in X_list])
    print(f"Raw feature shape: {X_all.shape}")

    # Filter artists with enough samples
    artist_counts = features_df['artist'].value_counts()
    valid_artists = artist_counts[artist_counts >= min_samples_per_artist].index
    features_df_filtered = features_df[features_df['artist'].isin(valid_artists)].reset_index(drop=True)

    removed_count = len(features_df) - len(features_df_filtered)
    print(f"Removed {removed_count} samples with artists having < {min_samples_per_artist} songs")

    # Build X_filtered and labels
    X_filtered_list = []
    for f in features_df_filtered['features']:
        if isinstance(f, str):
            arr = np.array(eval(f), dtype=np.float32)
        else:
            arr = np.array(f, dtype=np.float32)
        X_filtered_list.append(arr)
    if len(X_filtered_list) == 0:
        raise ValueError("No samples left after filtering. Lower min_samples_per_artist or provide more data.")
    X = np.vstack([x.reshape(1, -1) for x in X_filtered_list])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(features_df_filtered['artist'])

    print(f"Number of unique artists: {len(label_encoder.classes_)}")
    print(f"Label shape: {y.shape}")
    print(f"Final dataset size: {len(features_df_filtered)} samples")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if apply_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(pca_components, X_scaled.shape[1]))
        X_scaled = pca.fit_transform(X_scaled)
        print(f"PCA reduced features to: {X_scaled.shape[1]} dims")

    # compute class weights (on filtered set)
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
    print("Class weights (sample):", dict(list(class_weight_dict.items())[:5]))

    # Decide whether we can stratify
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    if min_count < 2:
        # cannot stratify — fallback but warn the user
        print("\n⚠ WARNING: At least one class has only 1 sample after filtering.")
        print("Splitting WITHOUT stratify. Results may be imbalanced / unstable. Prefer min_samples_per_artist>=2.")
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=stratify
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, scaler, label_encoder, class_weight_dict

# ------------------ Model & training ------------------
def build_dnn_model(input_shape, num_classes, l2=1e-4):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(X_train, y_train, X_test, y_test, num_classes, class_weight=None, epochs=80, batch_size=32):
    print("\nBuilding and training the model...")
    model = build_dnn_model(X_train.shape[1], num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight,
        verbose=1
    )
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    return model, history

# ------------------ Plotting ------------------
def plot_training_history(history, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history.get('accuracy', []), label='Train Accuracy')
    ax1.plot(history.history.get('val_accuracy', []), label='Val Accuracy')
    ax1.set_title('Model Accuracy'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend(); ax1.grid(True)
    ax2.plot(history.history.get('loss', []), label='Train Loss')
    ax2.plot(history.history.get('val_loss', []), label='Val Loss')
    ax2.set_title('Model Loss'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.show()

# ------------------ Prediction helpers ------------------
def predict_song(audio_path, model, scaler, label_encoder, top_k=5):
    print(f"\nPredicting for: {audio_path}")
    features = extract_mfcc_features(audio_path)
    if features is None:
        print("Failed to extract features")
        return None
    features_scaled = scaler.transform(features.reshape(1, -1))
    predictions = model.predict(features_scaled, verbose=0)
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    results = []
    for idx in top_indices:
        artist = label_encoder.inverse_transform([idx])[0]
        confidence = predictions[0][idx] * 100
        results.append((artist, confidence))
    print(f"\nTop {top_k} Predictions:")
    for i, (artist, confidence) in enumerate(results, 1):
        print(f"{i}. {artist}: {confidence:.2f}%")
    return {
        'top_artist': results[0][0],
        'confidence': results[0][1],
        'all_predictions': results
    }

def save_model_and_preprocessing(model, scaler, label_encoder, 
                                  model_path=MODEL_PATH,
                                  scaler_path=SCALER_PATH,
                                  encoder_path=ENCODER_PATH):
    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"\nModel and preprocessing objects saved!")
    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Label Encoder: {encoder_path}")

def load_model_and_preprocessing(model_path=MODEL_PATH,
                                  scaler_path=SCALER_PATH,
                                  encoder_path=ENCODER_PATH):
    model = keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print("Model and preprocessing objects loaded!")
    return model, scaler, label_encoder

# ------------------ GUI (optional) ------------------
class MusicRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Recognition System")
        self.root.geometry("900x650")
        self.root.configure(bg='#2C3E50')
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.selected_file = None
        self.setup_ui()
        self.load_model_on_startup()

    def setup_ui(self):
        title_frame = tk.Frame(self.root, bg='#34495E', pady=20)
        title_frame.pack(fill=tk.X)
        title_label = tk.Label(title_frame, text="🎵 Music Recognition System", 
                               font=('Arial', 24, 'bold'), bg='#34495E', fg='white')
        title_label.pack()
        main_frame = tk.Frame(self.root, bg='#2C3E50')
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        btn_frame = tk.Frame(main_frame, bg='#2C3E50')
        btn_frame.pack(pady=10)
        self.select_btn = tk.Button(btn_frame, text="📁 Select Audio File", command=self.select_file,
                                     font=('Arial', 12, 'bold'), bg='#3498DB', fg='white', padx=15, pady=8, cursor='hand2')
        self.select_btn.pack(side=tk.LEFT, padx=10)
        self.predict_btn = tk.Button(btn_frame, text="🎯 Predict", command=self.predict_file,
                                      font=('Arial', 12, 'bold'), bg='#2ECC71', fg='white', padx=15, pady=8, cursor='hand2', state=tk.DISABLED)
        self.predict_btn.pack(side=tk.LEFT, padx=10)
        self.file_label = tk.Label(main_frame, text="No file selected", font=('Arial', 12), bg='#2C3E50', fg='#ECF0F1')
        self.file_label.pack(pady=8)
        result_frame = tk.Frame(main_frame, bg='#34495E', relief=tk.RIDGE, bd=2)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        result_title = tk.Label(result_frame, text="Prediction Results", font=('Arial', 16, 'bold'), bg='#34495E', fg='white')
        result_title.pack(pady=10)
        self.result_text = tk.Text(result_frame, height=18, width=90, font=('Courier', 11), bg='#ECF0F1', fg='#2C3E50',
                                    relief=tk.FLAT, padx=10, pady=10)
        self.result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(self.result_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.result_text.yview)
        self.status_label = tk.Label(self.root, text="Ready", font=('Arial', 10), bg='#2C3E50', fg='#95A5A6')
        self.status_label.pack(side=tk.BOTTOM, pady=6)

    def load_model_on_startup(self):
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH):
                self.model, self.scaler, self.label_encoder = load_model_and_preprocessing()
                self.update_status("✓ Model loaded successfully!", "green")
                self.predict_btn.config(state=tk.NORMAL)
            else:
                self.update_status("⚠ Model not found. Train & save model first.", "orange")
        except Exception as e:
            self.update_status(f"✗ Error loading model: {str(e)}", "red")

    def select_file(self):
        file_path = filedialog.askopenfilename(title="Select Audio File", filetypes=[("Audio Files", ".mp3 *.wav *.flac *.ogg"), ("All Files", ".*")])
        if file_path:
            self.selected_file = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"Selected: {filename}")
            self.predict_btn.config(state=tk.NORMAL if self.model is not None else tk.DISABLED)
            self.update_status(f"File selected: {filename}", "blue")

    def predict_file(self):
        if not self.selected_file:
            messagebox.showwarning("No File", "Please select an audio file first!")
            return
        if self.model is None:
            messagebox.showerror("Model Not Loaded", "Please train and load the model first!")
            return
        self.predict_btn.config(state=tk.DISABLED)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Processing... Please wait...\n")
        self.update_status("Analyzing audio...", "blue")
        thread = threading.Thread(target=self.run_prediction)
        thread.start()

    def run_prediction(self):
        try:
            result = predict_song(self.selected_file, self.model, self.scaler, self.label_encoder, top_k=5)
            if result:
                self.root.after(0, self.display_results, result)
            else:
                self.root.after(0, self.display_error)
        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def display_results(self, result):
        self.result_text.delete(1.0, tk.END)
        output = "="*70 + "\n"
        output += "           PREDICTION RESULTS\n"
        output += "="*70 + "\n\n"
        output += f"🎤 TOP ARTIST: {result['top_artist']}\n"
        output += f"📊 CONFIDENCE: {result['confidence']:.2f}%\n\n"
        output += "-"*70 + "\n"
        output += "TOP PREDICTIONS:\n"
        output += "-"*70 + "\n\n"
        for i, (artist, conf) in enumerate(result['all_predictions'], 1):
            bar_length = int(conf / 2)
            bar = "█" * bar_length
            output += f"{i}. {artist}\n"
            output += f"   {bar} {conf:.2f}%\n\n"
        output += "="*70 + "\n"
        self.result_text.insert(tk.END, output)
        self.predict_btn.config(state=tk.NORMAL)
        self.update_status("✓ Prediction complete!", "green")

    def display_error(self, error_msg="Failed to process audio"):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"ERROR: {error_msg}\n")
        self.predict_btn.config(state=tk.NORMAL)
        self.update_status("✗ Prediction failed", "red")

    def update_status(self, message, color="white"):
        color_map = {"green": "#2ECC71", "red": "#E74C3C", "blue": "#3498DB", "orange": "#E67E22", "white": "#ECF0F1"}
        self.status_label.config(text=message, fg=color_map.get(color, color_map["white"]))

def launch_gui():
    root = tk.Tk()
    app = MusicRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FMA Music Recognition System - Ready!")
    print("="*70)
    print("\nQuick start (small test recommended):")
    print("1) Load metadata:")
    print("metadata_df = load_fma_metadata(METADATA_PATH)")
    print("2) Extract features (test small sample first):")
    print("features_df = process_fma_dataset(AUDIO_DIR, metadata_df, FEATURES_CSV, sample_limit=20)")
    print("3) Prepare data (returns class_weight):")
    print("X_train, X_test, y_train, y_test, scaler, label_encoder, class_weight = prepare_data_for_training(features_df, min_samples_per_artist=1)")
    print("4) Train:")
    print("model, history = train_model(X_train, y_train, X_test, y_test, len(label_encoder.classes_), class_weight=class_weight, epochs=20)")
    print("5) Save:")
    print("save_model_and_preprocessing(model, scaler, label_encoder)")
    print("6) Launch GUI (after saving model):")
    print("launch_gui()")
    print("\nNote: For best results later: use more data per artist, try apply_pca=True if feature vector is too large, or use pretrained audio embeddings (YAMNet/VGGish).")
    print("="*70)
