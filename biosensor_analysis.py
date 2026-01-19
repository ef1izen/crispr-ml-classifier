import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

#random seed so results are reproducible
np.random.seed(42)

def sigmoid_model (t, L, k, t0, b):
    """
    Standard Logistic Function for modeling reaction kinetics.
    t: Time points
    L: Max signal (amplitude)
    k: Rate constant
    t0: Time of half-max signal
    b: Background offset
    """
    return L / (1 + np.exp(-k * (t - t0))) + b

# ========================================
# DEMONSTRATION SECTION (Optional)
# Uncomment to see curve fitting examples
# ========================================
##
# 1. Create Time Points (0 to 60 minutes, taking a reading every minute)
#t = np.linspace(0, 60, 61)

# 2. Define our "True" Parameters (What we want to discover)
#L = 1000   # Max fluorescence
#k = 0.3    # Reaction rate
#t0 = 20    # Reaction hits 50% at 20 minutes
#b = 50     # Background noise is 50 units

# 3. Generate the "Perfect" Data
#perfect_signal = sigmoid_model(t, L, k, t0, b)

# 4. Add "Real World" Noise
# Lab instruments aren't perfect.
#noise_level = 25  # How messy is the machine?
#random_noise = np.random.normal(0, noise_level, len(time_points))

# 5. Create the "Observed" Data (Signal + Noise)
#observed_data = perfect_signal + random_noise

#plt.scatter(time_points, observed_data, color='red', s=10, label='Noisy Lab Data')
#plt.plot(time_points, perfect_signal, color='blue', linestyle='--', label='True Signal')
#plt.legend()
#plt.title("Simulated CRISPR Sensor Data")
#plt.xlabel("Time (min)")
#plt.ylabel("Fluorescence (RFU)")
#plt.show()

# 1. Guess the starting values (Optional, but helps the math)
# Guess: L=max(data), k=1, t0=median time, b=min(data)
#p0_guess = [max(observed_data), 1, np.median(t), min(observed_data)]

# 2. Run the Curve Fit
# This tries to find the L, k, t0, b that best fit the red dots
#popt, pcov = curve_fit(sigmoid_model, t, observed_data, p0=p0_guess)

# 3. Extract the "Found" Parameters
#calc_L, calc_k, calc_t0, calc_b = popt

#print(f"True Rate (k): {k}")
#print(f"Calculated Rate (k): {calc_k:.4f}")
#print(f"True Max (L): {L}")
#print(f"Calculated Max (L): {calc_L:.2f}")

# 4. Generate the "Fitted Curve" for plotting
#fitted_curve = sigmoid_model(t, *popt)

# 5. Plot everything
#plt.figure(figsize=(8,5))
#plt.scatter(time_points, observed_data, color='red', alpha=0.5, label='Raw Noisy Data')
#plt.plot(time_points, fitted_curve, color='green', linewidth=2, label=f'Fitted Model (Rate={calc_k:.2f})')
#plt.plot(time_points, perfect_signal, color='blue', linestyle=':', label='Ground Truth')
#plt.legend()
#plt.title("Curve Fitting: Extracting Signal from Noise")
#plt.xlabel("Time (min)")
#plt.ylabel("Fluorescence")
#plt.show()

#Defining generator function
#def generate_field_samples (num_samples = 50):

    #empty list to store synthetic data
#    samples = []

    #loop for 50
#    for i in range (num_samples):
#        """
#        Randomising biological parameters between weak and strong
#        L: 500 - 3000
#        k: 0.05 - 0.4
#        t0: 15 - 30
#        b: 50 - 100
#        """
#        rand_L = np.random.uniform(500, 3000)
#        rand_k = np.random.uniform(0.05, 0.4)
#        rand_t0 = np.random.uniform(15, 30)
#        rand_b = np.random.uniform(50, 100)

        #Randomise machine noise
#        noise_percent = np.random.uniform(0.01, 0.05)
#        noise_sigma = rand_L * noise_percent

        #Saving "sample" data
#        samples.append({
#            "Sample_ID": i,
#            "True_L": rand_L,
#            "True_k": rand_k,
#            "True_t0": rand_t0,
#            "True_b": rand_b,
#            "Noise_level": noise_sigma
#        })

#    return pd.DataFrame(samples)

#Generate and view
#df_params = generate_field_samples(50)
#print(df_params.head())

#Visualising variety by plotting 5 random curves
#t = np.linspace(0, 60, 61)

#plt.figure(figsize = (10, 6))
#for index, row in df_params.head(5).iterrows():

    #Calculate curve
#    y = sigmoid_model(t, row['True_L'], row['True_k'], row['True_t0'], row['True_b'])

    #Noise
#    noise = np.random.normal(0, row['Noise_level'], len(t))
#    y_noise = y + noise

#    plt.plot(t, y_noise, label = f"Sample {row['Sample_ID']} (k = {row['True_k']:.2f}")

#plt.title("5 Simulated Field Samples (Variable Kinetics)")
#plt.xlabel("Time (min)")
#plt.ylabel("Fluorescence (RFU)")
#plt.legend()
#plt.show()

# 2. Let's analyze the specific samples you plotted (0 to 4)
#results = []

#print(f"{'Sample':<8} | {'True k':<10} | {'Calc k':<10} | {'Accuracy':<10}")
#print("-" * 50)

#plt.figure(figsize=(12, 8))

# Loop through the first 5 samples
#for index, row in df_params.head(5).iterrows():
    
    # --- A. Re-create the Noisy Data (Simulating the Lab Reading) ---
    # (In real life, you'd just load this from a CSV file)
#    t = np.linspace(0, 60, 61)
#    perfect_y = sigmoid_model(t, row['True_L'], row['True_k'], row['True_t0'], row['True_b'])
#    noise = np.random.normal(0, row['Noise_level'], len(t))
#    noisy_data = perfect_y + noise
    
    # --- B. The Analysis (Curve Fitting) ---
#    try:
        # We give the fitter a rough guess to start:
        # L = max of data, k = 0.1, t0 = 20, b = min of data
#        initial_guess = [max(noisy_data), 0.1, 20, min(noisy_data)]
        
        # The Magic Line: This finds the best parameters
#        popt, pcov = curve_fit(sigmoid_model, t, noisy_data, p0=initial_guess, maxfev=5000)
        
#        calc_L, calc_k, calc_t0, calc_b = popt
        
        # --- C. Compare Truth vs. Math ---
#        error = abs(row['True_k'] - calc_k) / row['True_k'] * 100
#        print(f"Sample {int(row['Sample_ID']):<1} | {row['True_k']:.4f}     | {calc_k:.4f}     | {100-error:.1f}%")
        
        # --- D. Plot the Result ---
        # Plot dots for "Raw Data" and a dashed line for "Fitted Model"
#        plt.scatter(t, noisy_data, s=10, alpha=0.4) # faint dots
#        plt.plot(t, sigmoid_model(t, *popt), linestyle='--', linewidth=2, label=f"Fit Sample {int(row['Sample_ID'])}")

#    except Exception as e:
#        print(f"Sample {int(row['Sample_ID'])} Failed to fit! (Too much noise?)")

#plt.title("Recovering Biological Parameters from Noisy Data")
#plt.xlabel("Time (min)")
#plt.ylabel("Fluorescence")
#plt.legend()
#plt.show()

# 1. Update Generator to create Positives AND Negatives
def generate_mixed_dataset(num_samples=1000):
    samples = []
    
    for i in range(num_samples):
        # Flip a coin: Is this sample Positive (1) or Negative (0)?
        is_positive = np.random.choice([0, 1])
        
        if is_positive:
            # POSITIVE: Similar to before (variable L, k, t0)
            true_L = np.random.uniform(100, 1000) # Allow some very weak ones!
            true_k = np.random.uniform(0.02, 0.15)
            true_t0 = np.random.uniform(15, 30)
            true_b = np.random.uniform(50, 100)
            
            # Create the curve
            t = np.linspace(0, 60, 61)
            y = sigmoid_model(t, true_L, true_k, true_t0, true_b)
            
        else:
            # NEGATIVE: Flat line (L=0) + Background
            true_L = 0
            true_k = 0
            true_t0 = 0
            true_b = np.random.uniform(50, 100)
            
            # Create the flat line
            t = np.linspace(0, 60, 61)
            y = np.full(len(t), true_b)

        # Add Noise to everyone
        noise_level = np.random.uniform(30, 100) # Randomize how messy it is
        noise = np.random.normal(0, noise_level, len(t))
        y_noisy = y + noise
        
        # Save the raw time-series data (61 points) AND the label
        samples.append({
            "Label": is_positive,
            "Raw_Data": y_noisy,
            "True_L": true_L # Just for reference
        })
        
    return pd.DataFrame(samples)

# Generate the big dataset
df_ml = generate_mixed_dataset(1000)
print(f"Generated {len(df_ml)} samples.")
print(df_ml.head())

def extract_features(raw_row):
    data = raw_row['Raw_Data']
    
    # Feature 1: Final Value (Endpoint)
    # Use average of last 5 points to smooth out noise
    final_val = np.mean(data[-5:])
    
    # Feature 2: Total Rise (Max - Min)
    total_rise = np.max(data) - np.min(data)
    
    # Feature 3: Max Slope (Gradient)
    # This finds the steepest jump between any two minutes
    gradient = np.gradient(data)
    max_slope = np.max(gradient)
    
    # Feature 4: Signal-to-Noise Proxy (Standard Deviation)
    # High variability might mean noise, or a strong reaction.
    std_dev = np.std(data)
    
    return pd.Series([final_val, total_rise, max_slope, std_dev])

# Apply this to our whole dataset
# This creates a new table X with just the 4 features
X = df_ml.apply(extract_features, axis=1)
X.columns = ['Final_Val', 'Total_Rise', 'Max_Slope', 'Std_Dev']

# Our Target (Y) is the Label (0 or 1)
y = df_ml['Label']

print("Feature Table:")
print(X.head())

# 1. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize the Model
# n_estimators=100 means "Create 100 decision trees and vote on the answer"
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 3. Train it!
clf.fit(X_train, y_train)

# 4. Predict on the Test Set
y_pred = clf.predict(X_test)

# 5. Evaluate
print("Accuracy Report:")
print(classification_report(y_test, y_pred))

# 6. The Confusion Matrix (The visual proof)
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (0=Negative, 1=Positive)")
plt.show()

# Extract feature importance from the trained model
importances = clf.feature_importances_
feature_names = ['Final_Val', 'Total_Rise', 'Max_Slope', 'Std_Dev']

# Sort them for better plotting
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")
plt.title("What did the AI look at? (Feature Importance)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()