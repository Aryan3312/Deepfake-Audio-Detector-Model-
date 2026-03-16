import tensorflow as tf
import argparse
import librosa
import numpy as np
import os

# parse arguments
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-m", "--model", dest="model_path", required=True, help="Keras model")
arg_parser.add_argument("-c", "--clips", dest="clips", nargs="+", required=True, help="Audio clips to classify")
arg_parser.add_argument("-n", "--n-mels", default=91, dest="n_mels", help="Number of mel bands in the spectrogram")

args = arg_parser.parse_args()

# load clips
x = []

for clip in args.clips:
    try:
        audio, _ = librosa.load(clip, sr=16000, mono=True)
        print(f"Loaded {clip}: length={len(audio)}")

        if len(audio) == 0:
            raise ValueError(f"Audio file {clip} is empty or could not be loaded properly")

        # MEL SPECTROGRAM (MATCH TRAINING SETTINGS)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_mels=int(args.n_mels),
            hop_length=512,
            n_fft=1024,
            power=2.0
        )

        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # FIX SHAPE
        max_time_steps = 150
        if mel_spectrogram.shape[1] < max_time_steps:
            mel_spectrogram = np.pad(
                mel_spectrogram,
                ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])),
                mode='constant'
            )
        else:
            mel_spectrogram = mel_spectrogram[:, :max_time_steps]

        x.append(mel_spectrogram)

    except Exception as e:
        print(f"Error processing {clip}: {str(e)}")
        continue

if not x:
    print("No audio files were successfully processed. Exiting.")
    exit(1)

x = np.array(x)
x = x[..., np.newaxis]

# load keras model
args.model_path = os.path.abspath(args.model_path)
model = tf.keras.models.load_model(args.model_path)

predictions = model.predict(x)

for p in predictions:
    print("--------------------------------")
    print("Raw prediction vector:", p)

    human_prob = p[0] * 100
    ai_prob = p[1] * 100

    print("chance of human:", round(human_prob, 1), "%")
    print("chance of AI:", round(ai_prob, 1), "%")

    if human_prob > ai_prob:
        print("Human!")
    else:
        print("AI!")
