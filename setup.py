content = """import numpy as np
import librosa

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, duration=4, offset=0.0)
        
        # Normalize loudness so volume doesn't affect prediction
        audio = librosa.util.normalize(audio)
        
        min_length = sample_rate * 1
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)))

        # MFCC - captures tone and voice quality
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        mfcc_std = np.std(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)

        # Delta MFCC - captures how voice changes over time
        delta_mfccs = np.mean(librosa.feature.delta(
            librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)).T, axis=0)

        # Chroma - captures pitch
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)

        # Mel spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)

        # Zero crossing rate - captures voice texture
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
        zcr_std = np.std(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)

        # Spectral features - captures voice brightness
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate).T, axis=0)
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate).T, axis=0)
        spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate).T, axis=0)

        # Combine all features - NO raw RMS so loudness doesn't dominate
        combined = np.hstack([
            mfccs,          # 40 - most important
            mfcc_std,       # 40 - variation in voice
            delta_mfccs,    # 40 - voice changes
            chroma,         # 12 - pitch info
            mel,            # 128 - sound texture
            zcr,            # 1 - voice texture
            zcr_std,        # 1 - variation
            spec_centroid,  # 1 - brightness
            spec_rolloff,   # 1 - frequency rolloff
            spec_bandwidth  # 1 - bandwidth
        ])

        return combined

    except Exception as e:
        print(f"Error: {e}")
        return None


def extract_features_augmented(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, duration=4, offset=0.0)
        audio = librosa.util.normalize(audio)

        min_length = sample_rate * 1
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)))

        all_features = []

        # Original + noise augmentation only (fast and safe)
        audios = [
            audio,
            audio + np.random.randn(len(audio)) * 0.005
        ]

        for aug_audio in audios:
            try:
                aug_audio = librosa.util.normalize(aug_audio)
                if len(aug_audio) < min_length:
                    aug_audio = np.pad(aug_audio, (0, min_length - len(aug_audio)))

                mfccs = np.mean(librosa.feature.mfcc(y=aug_audio, sr=sample_rate, n_mfcc=40).T, axis=0)
                mfcc_std = np.std(librosa.feature.mfcc(y=aug_audio, sr=sample_rate, n_mfcc=40).T, axis=0)
                delta_mfccs = np.mean(librosa.feature.delta(
                    librosa.feature.mfcc(y=aug_audio, sr=sample_rate, n_mfcc=40)).T, axis=0)
                chroma = np.mean(librosa.feature.chroma_stft(y=aug_audio, sr=sample_rate).T, axis=0)
                mel = np.mean(librosa.feature.melspectrogram(y=aug_audio, sr=sample_rate).T, axis=0)
                zcr = np.mean(librosa.feature.zero_crossing_rate(y=aug_audio).T, axis=0)
                zcr_std = np.std(librosa.feature.zero_crossing_rate(y=aug_audio).T, axis=0)
                spec_centroid = np.mean(librosa.feature.spectral_centroid(y=aug_audio, sr=sample_rate).T, axis=0)
                spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=aug_audio, sr=sample_rate).T, axis=0)
                spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=aug_audio, sr=sample_rate).T, axis=0)

                features = np.hstack([
                    mfccs, mfcc_std, delta_mfccs, chroma, mel,
                    zcr, zcr_std, spec_centroid, spec_rolloff, spec_bandwidth
                ])
                all_features.append(features)

            except Exception as e:
                continue

        return all_features if all_features else None

    except Exception as e:
        print(f"Error: {e}")
        return None
"""

with open("utils/feature_extraction.py", "w") as f:
    f.write(content)

print("feature_extraction.py updated successfully!")