# crispr-ml-classifier

Simulate and classify CRISPR diagnostic signals using ML (Random Forest) on noisy data.

## Overview

A Python project to simulate CRISPR biosensor data and use machine learning to classify positive and negative detection results.

The project demonstrates how to:
- Simulate realistic biosensor data using sigmoid kinetics models
- Add noise to mimic real-world conditions
- Train a machine learning classifier to distinguish positive from negative samples

## Features

- Synthetic data generation with adjustable noise levels.
- Curve fitting to extract reaction kinetics.
- Random Forest classifier for binary classification (positive/negative detection).
- Visualisation of results including confusion matrices and feature importance.

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SciPy
- Pandas
- scikit-learn
- seaborn

## Installation

**Clone:**
```bash
git clone https://github.com/ef1izen/crispr-ml-classifier.git
cd crispr-ml-classifier
```

**Install dependencies:**
```bash
pip install numpy matplotlib scipy pandas scikit-learn seaborn
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

**Run:**
```bash
python biosensor_analysis.py
```

The script will:
1. Generate synthetic biosensor data with noise
2. Demonstrate curve fitting on noisy data
3. Create a dataset of 1000 samples (positive and negative)
4. Train a Random Forest classifier
5. Display performance metrics and visualisations

## How It Works

### Signal Modelling

The code uses a logistic (sigmoid) function to model CRISPR sensor kinetics:

```
Signal = L / (1 + exp(-k * (t - t0))) + b
```

Where:
- `L` = Maximum signal
- `k` = Reaction rate constant
- `t0` = Time of half-maximum signal
- `b` = Background fluorescence

### Feature Extraction

Four features are extracted from each time-series curve:
- **Final Value**: Average of last 5 time points
- **Total Rise**: Maximum - Minimum signal
- **Max Slope**: Steepest gradient in the curve
- **Standard Deviation**: Signal variability (noise proxy)

### Classification

A Random Forest classifier uses these features to predict whether a sample is positive (target detected) or negative (no target).

## Results

Typical performance on synthetic data:
- **Accuracy**: ~95-99%
- **Precision**: High for both classes

## Customisation

You can adjust parameters in the code:

**Number of samples:**
```python
df_ml = generate_mixed_dataset(1000)
```

**Noise levels:**
```python
noise_level = np.random.uniform(30, 100)
```

**Random Forest settings:**
```python
clf = RandomForestClassifier(n_estimators=100, random_state=42)
```

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements - always welcome!

## License

MIT License - feel free to use this code for educational or research purposes.

## Author

ef1izen
