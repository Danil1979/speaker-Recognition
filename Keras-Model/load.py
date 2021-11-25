
"""
## Setup
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import colorama 

colorama.init(strip=False)

DATASET_ROOT = os.path.join(os.path.expanduser("~"), "Downloads/16000_pcm_speeches")
DATASET_ROOT_DOWNLOAD = os.path.join(os.path.expanduser("~"), "Downloads")
# The folders in which we will put the training data and testing data
AUDIO_SUBFOLDER = "audio"


DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)

DATASET_TEST_PATH = os.path.join(DATASET_ROOT_DOWNLOAD, "train")
# Percentage of samples to use for validation
VALID_SPLIT = 0.1

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43

# The sampling rate to use.
# This is the one used in all of the audio samples.
# We will resample all of the noise to this sampling rate.
# This will also be the output size of the audio wave samples
# (since all samples are of 1 second long)
SAMPLING_RATE = 16000

# The factor to multiply the noise with according to:
#   noisy_sample = sample + noise * prop * scale
#      where prop = sample_amplitude / noise_amplitude
SCALE = 0.5

BATCH_SIZE = 25
EPOCHS = 100

"""
## Data preparation
The dataset is composed of 5 folders
- Speech samples, with 7 folders for 7 different speakers. Each folder contains
1400 audio files, each 1 second long and sampled at 16000 Hz.
"""


"""
## Dataset generation
"""

def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale

    return audio


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


# Get the list of audio file paths along with their corresponding labels
class_names = os.listdir(DATASET_AUDIO_PATH)
audio_paths = []
labels = []
for label, name in enumerate(class_names):
    print("Processing speaker {}".format(name,))
    dir_path = Path(DATASET_AUDIO_PATH) / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)

print(
    "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
)


# Shuffle
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)

# Split into training and validation
num_val_samples = int(VALID_SPLIT * len(audio_paths))
print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]
valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

valid_ds = valid_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)

print("Our class names: {}".format(class_names,))
test_folder = os.listdir(DATASET_TEST_PATH)
test_paths = []
test_labels = []
for label, name in enumerate(test_folder):
    print("Processing speaker for testing {}".format(name,))
    dir_path = Path(DATASET_TEST_PATH) / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    test_paths += speaker_sample_paths
    test_labels += [label] * len(speaker_sample_paths)
    print(test_labels)

model = keras.models.load_model('speaker-Recognition/saved_model/my_model(v4)')
model.summary()
print(model.evaluate(valid_ds))

"""
We get ~ 98% validation accuracy.
"""

"""
## Demonstration
Let's take some samples and:
- Predict the speaker
- Compare the prediction with the real speaker
- Listen to the audio to see that despite the samples being noisy,
the model is still pretty accurate
"""

SAMPLES_TO_DISPLAY = BATCH_SIZE

# test_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
test_ds = paths_and_labels_to_dataset(test_paths, test_labels)
test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
    BATCH_SIZE
)

correct = 0
total = 0
results = [None] * BATCH_SIZE
for i in range(BATCH_SIZE):
    results[i] = [None] * 3

# print(results)
for audios, labels in test_ds.take(1):
    # Get the signal FFT
    ffts = audio_to_fft(audios)
    # Predict
    y_pred = model.predict(ffts)
    # Take random samples
    rnd = np.random.randint(0, BATCH_SIZE, SAMPLES_TO_DISPLAY)
    print(rnd)
    audios = audios.numpy()[rnd, :, :]
    labels = labels.numpy()[rnd]
    y_pred = np.argmax(y_pred, axis=-1)[rnd]
    print(y_pred)
    for index in range(SAMPLES_TO_DISPLAY):
        # For every sample, print the true and predicted label
        # as well as run the voice with the noise
        results[index][0] = (class_names[labels[index]])
        results[index][1] = (class_names[y_pred[index]])
        if labels[index] == y_pred[index]:
            correct += 1
            results[index][2] = ("True")
        else:
            results[index][2] = ("False")
        total += 1
        
        print(
            "Speaker:\033{} {}\033[0m\tPredicted:\033{} {}\033[0m".format(
                "[42m" if labels[index] == y_pred[index] else "[41m",
                class_names[labels[index]],
                "[42m" if labels[index] == y_pred[index] else "[41m",
                class_names[y_pred[index]],
            )
        )
        # display(Audio(audios[index, :, :].squeeze(), rate=SAMPLING_RATE) )
print(correct/total)
df = pd.DataFrame(results)
results_csv_file = 'results13.csv'
with open(results_csv_file, mode='w') as f:
    df.to_csv(f)