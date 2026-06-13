# Edge AI Signal Classifier 

An end-to-end Machine Learning pipeline designed to classify raw acoustic telemetry signals from industrial machinery to detect structural friction faults before catastrophic failures happen.

## The Problem & Approach
In industrial setups, machines give off distinct sound signatures. Standard 1D audio time-series data is incredibly chaotic and full of factory floor background noise, making it hard for basic models to spot anomalies. 

To solve this, I used an ECE-based signal processing approach: converting 1D audio waves into 2D frequency images (spectrograms) so a neural network can easily map out a clean decision boundary.

## How it Works
1. **Signal Generation:** Generated a balanced simulation dataset of 200 samples representing both smooth normal hums and high-frequency friction screeches.
2. **Feature Extraction (STFT & Mel-Scale):** Applied a Short-Time Fourier Transform to segment frequencies over time, then warped them onto a logarithmic Mel Scale to match physical machine response bands.
3. **Data Engineering:** Reshaped raw 22,050 data points into structured 3D array shapes of `(1, 128, 44)` to mimic image tensor dimensions.
4. **Neural Network Training:** Built a Multi-Layer Perceptron (MLP) using Scikit-Learn with ReLU activation layers and an Adam optimizer.

## Results
Despite adding heavy random Gaussian white noise to simulate real factory environments, the network achieved a **1.00 F1-Score** on entirely unseen test splits, showing zero false negatives.

## Tech Stack
* **Language:** Python
* **Signal Processing:** Librosa, NumPy
* **Machine Learning:** Scikit-Learn
* **Environment:** VS Code / Jupyter Notebooks
