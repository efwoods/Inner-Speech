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


The layout you're referring to is based on the 10-20 system, which is a standardized method for placing electrodes on the scalp for electroencephalography (EEG) and other electrophysiological recordings.

The 10-20 system, developed in the 1950s by Dr. Hans Berger and later refined by the International Federation of Societies for Electroencephalography and Clinical Neurophysiology (IFSECN), provides a standardized and reproducible way to place electrodes on the scalp.

Here's a brief explanation of the 10-20 system:

    The system uses a grid of points on the scalp, spaced 10% or 20% of the distance between anatomical landmarks.
    The landmarks used are:
        Nz: Nasion (the bridge of the nose)
        Inion: The most posterior point of the occipital bone (at the back of the head)
        Preauricular points: Points in front of each ear
    The grid is divided into regions, labeled with letters and numbers:
        Letters: Indicate the region (e.g., F for frontal, C for central, P for parietal, O for occipital, T for temporal)
        Numbers: Indicate the position (odd numbers for left hemisphere, even numbers for right hemisphere)

The layout you linked shows a 128-channel EEG system, with electrodes placed according to the 10-20 system and its extensions. Here's a rough mapping of the regions to electrode positions:

    Frontal region:
        Fp1, Fp2: Frontopolar regions
        F3, F4: Frontal regions (anterior)
        F7, F8: Frontal regions (temporal-frontal junction)
    Central region:
        C3, C4: Central regions (motor cortex)
        T3, T4: Temporal regions (anterior)
    Parietal region:
        P3, P4: Parietal regions (sensory cortex)
        P7, P8: Parietal regions (posterior)
    Occipital region:
        O1, O2: Occipital regions (visual cortex)
    Temporal region:
        T5, T6: Temporal regions (posterior)
        TP7, TP8: Temporal regions (temporoparietal junction)

Some key points to keep in mind:

    The actual electrode positions may vary slightly depending on individual head shape and size.
    Modern EEG systems often use more electrodes and may employ other placement systems, such as the 10-10 system or the Equi spaced montage.

To help you better understand the layout, you can refer to the following resources:

    The Biosemi website provides detailed information on their electrode layouts and placement guides.
    The International EEG Society has published guidelines for EEG electrode placement.
    Online tools, such as EEG electrode placement simulators, can help visualize the electrode positions.

Keep in mind that EEG data analysis and interpretation require a good understanding of the underlying neurophysiology and EEG processing techniques. If you're new to EEG analysis, it's essential to consult with experts in the field and follow established guidelines and best practices.
