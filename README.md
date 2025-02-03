# Brain-Computer Interface (BCI) - EEG Hand Movement Classification

## Introduction

Individuals suffering from spinal cord injuries often lose the ability to move their limbs. This project aims to leverage machine learning techniques to analyze EEG data and classify brain wave patterns related to hand movements. With high accuracy, such models can help create Brain-Computer Interfaces (BCIs) that allow users to control mechanical limbs or communicate through brain signals.

## Dataset

The dataset used in this project is obtained from the EMOTIV Insight + 5 Channel EEG device. The device records brain activity from the following regions:

- AF3
- AF4
- T7
- T8
- Pz

The dataset consists of brainwave recordings during visual stimuli that represent different hand movements:

- **Right movement:** Rightward arrow stimulus
- **Left movement:** Leftward arrow stimulus
- **No movement:** Circle stimulus

The sampling frequency of the device is **128 Hz**, and the raw EEG signals were pre-processed for better classification.

## Data Collection Protocol

- EEG data was recorded over a **36-minute protocol** with alternating stimuli.
- Participants observed visual cues while their brain responses were recorded.
- Pre-processing steps involved transforming the data into the frequency domain and computing power values for **Alpha, Beta, Theta, and Gamma waves**.

## Pre-Processing

1. Transforming EEG signals into the **frequency domain**
2. Computing weighted and arithmetic mean for different frequency bands
3. Constructing a **5×5 feature matrix** representing power values
4. Standardizing and cleaning the data

## Machine Learning Models and Evaluation

Several classification models were tested to determine the best-performing approach. The key models evaluated include:

| Model | Accuracy (%) |
|--------|--------------|
| Logistic Regression (LR) | 54.03 |
| Decision Tree (DT) | 64.75 |
| K-Nearest Neighbors (KNN) | **98.25** |
| Naive Bayes (NB) | 42.53 |
| AdaBoost (AdaB) | 52.40 |
| Gradient Boosting (GBM) | 61.23 |
| Random Forest (RF) | **89.58** |
| Support Vector Machine (SVM) | 54.35 |

### Best Performing Models:
- **K-Nearest Neighbors (KNN):** Achieved **98% accuracy**, making it the most effective model for classifying hand movements based on EEG data.
- **Random Forest (RF):** Achieved **89.58% accuracy** after hyperparameter tuning.

## Hyperparameter Optimization

### Random Forest Optimization
- **Best Parameters:** `{n_estimators: 200, min_samples_split: 4, min_samples_leaf: 2, max_features: 'sqrt', max_depth: 18}`
- **Best Accuracy:** `86.31% (Cross-validation)`

### KNN Optimization
- **Best Parameters:** `{metric: 'manhattan', n_neighbors: 1, weights: 'uniform'}`
- **Best Accuracy:** `93.77% (Cross-validation)`

## Results and Conclusion
- The **KNN model** performed exceptionally well, achieving **98% accuracy** on the test set.
- The **Random Forest model** was also effective, with **89.58% accuracy** after optimization.
- These results demonstrate the potential of EEG-based machine learning models for **Brain-Computer Interfaces (BCIs)**.

## Repository Structure
```
├── data/                   # Raw and pre-processed EEG data
├── models/                 # Trained machine learning models
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/                # Python scripts for data processing and training
├── results/                # Model evaluation and visualizations
└── README.md               # Project documentation
```

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/BCI-EEG-Hand-Movement.git
   cd BCI-EEG-Hand-Movement
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run data pre-processing:
   ```bash
   python scripts/preprocess_data.py
   ```
4. Train the models:
   ```bash
   python scripts/train_model.py
   ```
5. Evaluate performance:
   ```bash
   python scripts/evaluate_model.py
   ```

## Author
- **Çağatay Elik**

## References
- [Brain wave data from hands movement of EEG (Kaggle)](https://www.kaggle.com/datasets/fabriciotorquato/brain-wave-data-from-hands-movement-of-eeg)

