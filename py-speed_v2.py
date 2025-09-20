import argparse
import time
import numpy as np
import pandas as pd
import os
import sys
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Set up proper error handling and argument parsing
parser = argparse.ArgumentParser(
    description="CPU/GPU benchmark using XGBoost model training"
)
parser.add_argument(
    "-n",
    "--num_passes",
    type=int,
    default=5,
    help="Number of passes to run (default: 5)",
)
parser.add_argument(
    "-s", "--size", type=int, default=10000, help="Dataset size (default: 10000)"
)

parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "gpu"],
    default="cpu",
    help="Device to use for training: cpu or gpu (default: cpu)",
)
parser.add_argument(
    "--gpu_id",
    type=int,
    default=0,
    help="GPU ID to use when device=gpu (default: 0)",
)
args = parser.parse_args()

# Validate arguments
if args.num_passes <= 0:
    print("Error: Number of passes must be positive")
    sys.exit(1)
if args.size <= 10:
    print("Error: Dataset size must be greater than 10")
    sys.exit(1)

# Set parameters with validated arguments
dataset_size = args.size
num_passes = args.num_passes
device = args.device

# Configure XGBoost parameters based on device choice
if device == "gpu":
    xgb_params = {"tree_method": "hist", "device": f"cuda:{args.gpu_id}"}
    print(f"Using GPU device (CUDA:{args.gpu_id}) for training")
else:
    xgb_params = {"tree_method": "hist", "device": "cpu", "nthread": 24}
    print("Using CPU for training with 24 threads")

# Set random seed for reproducibility
np.random.seed(args.seed)

print("\nGenerating dataset... (only done once)")
dataset_prep_start_time = time.time()
dlen = int(dataset_size / 2)
X_11 = pd.Series(np.random.normal(2, 2, dlen))
X_12 = pd.Series(np.random.normal(9, 2, dlen))
X_1 = pd.concat([X_11, X_12]).reset_index(drop=True)
X_21 = pd.Series(np.random.normal(1, 3, dlen))
X_22 = pd.Series(np.random.normal(7, 3, dlen))
X_2 = pd.concat([X_21, X_22]).reset_index(drop=True)
X_31 = pd.Series(np.random.normal(3, 1, dlen))
X_32 = pd.Series(np.random.normal(3, 4, dlen))
X_3 = pd.concat([X_31, X_32]).reset_index(drop=True)
X_41 = pd.Series(np.random.normal(1, 1, dlen))
X_42 = pd.Series(np.random.normal(5, 2, dlen))
X_4 = pd.concat([X_41, X_42]).reset_index(drop=True)
Y = pd.Series(np.repeat([0, 1], dlen))
df = pd.concat([X_1, X_2, X_3, X_4, Y], axis=1)
df.columns = ["X1", "X2", "X3", "X_4", "Y"]
# df.head()
TRAIN_SIZE = 0.80
X = df.drop(["Y"], axis=1).values
y = df["Y"]
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
# Encode labels
y = label_encoder.fit_transform(y)
# identify shape and indices
num_rows, num_columns = df.shape
delim_index = int(num_rows * TRAIN_SIZE)
# Splitting the dataset in training and test sets
X_train, y_train = X[:delim_index, :], y[:delim_index]
X_test, y_test = X[delim_index:, :], y[delim_index:]
# Checking dimensions in percentages
total = X_train.shape[0] + X_test.shape[0]
dataset_prep_stop_time = time.time()
print(f"Fitting model...(done {num_passes} times.)")
fit_time_list = []
try:
    for n in range(num_passes):
        print(f"  Pass {n+1}/{num_passes}...", end="", flush=True)
        n_model_fit_starting_time = time.time()
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)
        n_model_fit_stop_time = time.time()
        elapsed = n_model_fit_stop_time - n_model_fit_starting_time
        fit_time_list.append(elapsed)
        print(f" completed in {elapsed:.4f} seconds")

    mean_fitting_time = np.mean(fit_time_list)
    std_fitting_time = np.std(fit_time_list)
except Exception as e:
    print(f"\nError during model fitting: {str(e)}")
    if device == "gpu":
        print("GPU training failed. This could be due to:")
        print("- No GPU available")
        print("- XGBoost not compiled with GPU support")
        print("- CUDA/ROCm drivers not properly installed")
        print("Try running with --device cpu")
    sys.exit(1)

print(f"Predicting with fitted model...(done {num_passes} times.)")
pred_time_list = []
try:
    for n in range(num_passes):
        print(f"  Pass {n+1}/{num_passes}...", end="", flush=True)
        pred_start_time = time.time()
        if device == "gpu":
            # Try different approaches to avoid device mismatch warning
            try:
                # Method 1: Create DMatrix without device parameter (older XGBoost versions)
                import xgboost as xgb
                dtest = xgb.DMatrix(X_test)
                # Set device on the booster before prediction
                booster = model.get_booster()
                booster.set_param({"device": f"cuda:{args.gpu_id}"})
                sk_pred = booster.predict(dtest)
            except Exception:
                # Fallback to standard prediction (will show warning)
                sk_pred = model.predict(X_test)
        else:
            sk_pred = model.predict(X_test)
        sk_pred = np.round(sk_pred)
        pred_stop_time = time.time()
        elapsed = pred_stop_time - pred_start_time
        pred_time_list.append(elapsed)
        print(f" completed in {elapsed:.4f} seconds")

    mean_pred_time = np.mean(pred_time_list)
    std_pred_time = np.std(pred_time_list)
    sk_acc = accuracy_score(y_test, sk_pred)
except Exception as e:
    print(f"\nError during prediction: {str(e)}")
    sys.exit(1)
# Calculate total benchmark time
data_prep_time = dataset_prep_stop_time - dataset_prep_start_time
total_benchmark_time = data_prep_time + (mean_fitting_time * num_passes) + (mean_pred_time * num_passes)

# Print results
print("\n" + "=" * 50)
print(f"BENCHMARK RESULTS")
print("=" * 50)
print(f"Device:                 {device.upper()}")
if device == "gpu":
    print(f"GPU ID:                 {args.gpu_id}")
print(f"Model accuracy:         {100 * sk_acc:.2f}%")
print(f"Dataset size:           {dataset_size}")
print(f"Number of passes:       {num_passes}")
print(f"Data prep time:         {data_prep_time:.3f} seconds")
print(f"Avg. fitting time:      {mean_fitting_time:.4f} seconds (±{std_fitting_time:.4f})")
print(f"Avg. prediction time:   {mean_pred_time:.4f} seconds (±{std_pred_time:.4f})")
print(f"Total benchmark time:   {total_benchmark_time:.3f} seconds")
print("=" * 50)

print("\nBenchmark completed!")
