# Inner-Speech

This is a repository to translate inner speech to human-readable text
### [Dataset](https://openneuro.org/datasets/ds003626/versions/2.1.2)

## Pipeline Overview

1. Preprocessing

    Load the dataset (BIDS or raw format).

    Apply filtering, artifact removal, epoching, and feature extraction (e.g., MFCCs for audio, bandpass for EEG).

2. Model for Inner Speech Decoding

    Fine-tune a model (e.g., BiLSTM, Transformer, or wav2vec2.0) on the features to decode imagined or inner speech.

    Output: Probabilistic text sequences or word predictions.

3. Streaming Simulation

    Simulate a real-time stream by chunking test data into time slices.

    Send decoded output (inference result) via WebSocket or local pipe.

4. Chat-Studio Interface

    Text from the model appears live in a frontend interface.

    Allow interaction or replay of decoded segments.


### EEG Electrode Placement:

![](https://www.biosemi.com/pics/cap_128_layout_medium.jpg)

## Roadmap

I'm going to notch filter with 60hz the eeg data, allow a bandpass butterworth filter of 0.5 Hz to 100 Hz, cut the sample from 1.5s to 3.5s to identify useful signal. Then I have 128 channels of data that I will take the 0 mean and unit variance of with a StandardScaler. I will preprocess all the data related to inner speech runs within the first 2 session for all 10 subjects. Then I will use this data to train a LSTM CNN model with 4 outputs using the associated training labels. 

Then I will create a simulation where the data is randomly selected and a simulated stream of information is sent to the classifier and then sent to chat-studio and displayed on the screen as text. This is the visualization of an inner speech pipeline. 