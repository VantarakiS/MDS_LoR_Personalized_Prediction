import pandas as pd
import numpy as np
import os
import csv
from tqdm import tqdm
from scipy.spatial.distance import euclidean, cosine
from fastdtw import fastdtw

# ----------------------------------------
# Feature set for safety and variability
# Ensures consistent row order across patient files
# ----------------------------------------
feature_set = [ 
    'Λευκά αιμοσφαίρα/WBC (Κ/μL)', 'NEU/ Ουδετρόφιλα (Κ/μL)', 'Λεμφοκύτταρα -Lymph (Κ/μL)',
    'Ερυθρά αιμοσφαίρια (RBC)', 'MCV', 'MCH', 'Hb', 'Hct', 'ΑΜΠ/PLTs/αιμοπετάλια (Κ/μL)', 
    'Glu/Gl/Γλυκόζη', 'U/Ur/ουρια', 'UA/ ουρικό οξύ', 'SGPT/ALT', 'γGT', 'LDH', 
    'gl/Glob/Σφαιρίνες', 'K', 'Na', 'Ca', 'P', 'CRP'
]

def log_debug_info(message, filename="nan_debug_log.txt"):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def normalize_fixed(df):
    normalized = df.copy()
    for feature in df.index:
        base_value = df.loc[feature].iloc[0]
        shifted = df.loc[feature] - base_value
        max_abs = shifted.abs().max()
        normalized.loc[feature] = shifted / max_abs if max_abs != 0 else 0

        if df.loc[feature].isna().any():
            log_debug_info(f"⚠️ NaNs in input before normalizing '{feature}':\n{df.loc[feature]}")
        if normalized.loc[feature].isna().any():
            log_debug_info(f"⚠️ NaNs in normalized output for '{feature}':\n{normalized.loc[feature]}")
    return normalized

def compute_euclidean(series1, series2):
    return euclidean(series1, series2)

def compute_cosine(series1, series2):
    return cosine(series1, series2)

def compute_dtw(series1, series2):
    distance, _ = fastdtw(series1, series2)
    return distance

def load_patient(filepath):
    df = pd.read_excel(filepath, header=0, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')

    loss_response = df.loc["Loss response"].iloc[0]
    df = df.drop("Loss response")

    if df.isnull().values.any():
        log_debug_info(f"📁 File '{filepath}' has NaNs:\n{df.isnull().sum()}")

    return df, loss_response

def find_most_similar_patients(new_patient_df, existing_patients_dict, loss_times_dict, metric="euclidean", top_k=4):
    similarities = []
    new_patient_flat = new_patient_df.values.flatten()

    for patient_name, patient_df in existing_patients_dict.items():
        candidate_flat = patient_df.values.flatten()

        if new_patient_flat.shape != candidate_flat.shape:
            log_debug_info(f"⚠️ Shape mismatch: new_patient={new_patient_flat.shape}, candidate={candidate_flat.shape}, name={patient_name}")
            continue

        if (np.isnan(new_patient_flat).any() or np.isnan(candidate_flat).any() or
            np.isinf(new_patient_flat).any() or np.isinf(candidate_flat).any()):
            log_debug_info(f"🚨 Invalid values comparing {patient_name}")
            continue

        if metric == "euclidean":
            dist = compute_euclidean(new_patient_flat, candidate_flat)
        elif metric == "cosine":
            dist = compute_cosine(new_patient_flat, candidate_flat)
        elif metric == "dtw":
            dist = compute_dtw(new_patient_flat, candidate_flat)
        else:
            raise ValueError("Unknown metric")

        similarities.append((dist, patient_name))

    if len(similarities) < top_k:
        return [], np.nan

    top_k_matches = sorted(similarities, key=lambda x: x[0])[:top_k]
    estimated_times = [loss_times_dict[name] for _, name in top_k_matches]
    return top_k_matches, np.mean(estimated_times)

def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100

def evaluate_predictions(actual, predicted):
    mape = safe_mape(actual, predicted)

    within_10 = [(a, p) for a, p in zip(actual, predicted) if abs(a - p) <= 10]
    within_5 = [(a, p) for a, p in zip(actual, predicted) if abs(a - p) <= 5]

    def adj_mape(pairs):
        if not pairs:
            return None
        return safe_mape([a for a, _ in pairs], [p for _, p in pairs])

    return {
        "mape": mape,
        "perc_10": 100 * len(within_10) / len(actual),
        "adj_mape_10": adj_mape(within_10),
        "perc_5": 100 * len(within_5) / len(actual),
        "adj_mape_5": adj_mape(within_5)
    }

def run_evaluation(patient_paths, base_folder):
    actual_losses, predicted_losses = [], []

    for path in patient_paths:
        patient_name = os.path.basename(path).replace("_imputed_.xlsx", "")  #naming differences
        target_df, actual_loss = load_patient(path)

        if target_df.shape[1] < 3 or any(f not in target_df.index for f in feature_set):
            continue

        target_df = target_df.loc[feature_set].iloc[:, :3]
        new_patient_df = normalize_fixed(target_df)

        others, losses = {}, {}
        for other_path in patient_paths:
            if other_path == path:
                continue
            name = os.path.basename(other_path).replace("_imputed_.xlsx", "")  #put naming differences
            df, loss = load_patient(other_path)

            if df.shape[1] < 3 or any(f not in df.index for f in feature_set):
                continue

            temp_df = df.loc[feature_set].iloc[:, :3]
            others[name] = normalize_fixed(temp_df)
            losses[name] = loss

        if len(others) < 3:
            continue

        _, est = find_most_similar_patients(new_patient_df, others, losses)
        if not np.isnan(est):
            actual_losses.append(actual_loss)
            predicted_losses.append(est)

    if any(np.isnan(actual_losses)) or any(np.isnan(predicted_losses)):
        log_debug_info(f"❌ NaNs before evaluation.\nActual: {actual_losses}\nPredicted: {predicted_losses}")

    evals = evaluate_predictions(actual_losses, predicted_losses)

    # Save predictions
    results_folder = os.path.join(base_folder, "predictions")
    os.makedirs(results_folder, exist_ok=True)
    output_path = os.path.join(results_folder, f"results_final_set.csv")

    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Patient", "Actual_Loss_Time", "Estimated_Loss_Time", "Error (%)"])
        for p, a, p_ in zip([os.path.basename(p).replace("_imputed_only.xlsx", "") for p in patient_paths], actual_losses, predicted_losses):
            error_pct = abs(a - p_) / max(abs(a), 1e-8) * 100
            writer.writerow([p, a, p_, error_pct])

    return evals

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    base_folder = r'path\to\your\folder'
    imputed_folder = os.path.join(base_folder, "imputed")  # the paths
    patient_files = [os.path.join(imputed_folder, f) for f in os.listdir(imputed_folder) if f.startswith("P")] #eg simulated

    print("Running evaluation on all patients...")
    results = run_evaluation(patient_files, base_folder)

    # Save summary
    output_csv = os.path.join(base_folder, "final_evaluation.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["MAPE", "%<=10", "Adj_MAPE_10", "%<=5", "Adj_MAPE_5"])
        writer.writerow([
            results["mape"],
            results["perc_10"],
            results["adj_mape_10"],
            results["perc_5"],
            results["adj_mape_5"]
        ])

    print("✅ Evaluation completed. Results written to:", output_csv)
