import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import logging
import time


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# --- Logging ---
current_script = os.path.basename(__file__).removesuffix(os.path.splitext(os.path.basename(__file__))[1])
logger = logging.getLogger(current_script)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler(f"logs/{current_script}.log", encoding="utf-8")
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[%(asctime)s][%(levelname)s] - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --- System Information ---
def log_system_info():
    """Log system information for performance monitoring."""
    cpu_cores = cpu_count()
    if PSUTIL_AVAILABLE:
        memory_gb = psutil.virtual_memory().total / (1024**3)
        logger.info(f"System Info: {cpu_cores} CPU cores, {memory_gb:.1f}GB RAM")
    else:
        logger.info(f"System Info: {cpu_cores} CPU cores")
    logger.info(f"🔧 Process will use nested parallelization for optimal CPU utilization")
    return cpu_cores


# --- Configuration ---
BASE_FOLDER = r'path\to\your\folder'
REAL_FOLDER = os.path.join(BASE_FOLDER, "output/imputed")
OUTPUT_FOLDER = os.path.join(BASE_FOLDER, "output/simulated_patients")
VISUALS_FOLDER = os.path.join(OUTPUT_FOLDER, "visuals")
BATCH_SIZE = 50
TOTAL_simulated = 300

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VISUALS_FOLDER, exist_ok=True)

# --- Parameters ---
GAUSSIAN_STD = 0.05
RANDOM_SCALE_RANGE = (0.95, 1.05)

# --- Helper Functions ---
def load_real_patients(folder):
    patient_dfs = {}
    for file in os.listdir(folder):
        if file.endswith("_imputed_.xlsx"):
            name = file.replace("_imputed_.xlsx", "")
            df = pd.read_excel(os.path.join(folder, file), index_col=0)
            df = df.apply(pd.to_numeric, errors='coerce')
            patient_dfs[name] = df
    return patient_dfs

def perturb_column(column_values, median_vals):
    rand = random.random()
    if rand < 0.3:  # 30% chance to change
        perturb_type = random.choices(
            ['strong', 'mild', 'none'],
            weights=[0.2, 0.4, 0.4],
            k=1
        )[0]
        if perturb_type == 'strong':
            return median_vals * (1 + np.random.uniform(-0.10, 0.10))
        elif perturb_type == 'mild':
            return median_vals * (1 + np.random.uniform(-0.05, 0.05))
    return column_values

def perturb_loss_of_response(value):
    rand = random.random()
    if rand < 0.2:
        return value  # 20% chance unchanged
    elif rand < 0.6:
        return value + random.choice([-1, 1])  # 40% chance ±1
    else:
        return value + random.choice([-2, 2])  # 40% chance ±2

def apply_statistical_noise(df):
    noisy_df = df.copy()
    timepoints = df.columns.copy()
    for row in noisy_df.index[:-1]:  # avoid LoR row
        method = random.choice(['gaussian', 'scale'])
        for col in timepoints:
            val = df.at[row, col]
            if pd.isna(val):
                continue
            if method == 'gaussian':
                noise = np.random.normal(loc=0, scale=GAUSSIAN_STD)
                noisy_val = val + noise
            elif method == 'scale':
                scale = np.random.uniform(*RANDOM_SCALE_RANGE)
                noisy_val = val * scale
            noisy_df.at[row, col] = noisy_val
    return noisy_df

def generate_simulated_patient(real_df):
    simulated_df = real_df.copy()
    timepoints = real_df.columns.copy()
    metric_medians = real_df.median(axis=1, skipna=True)

    # Apply column-wise timepoint perturbation
    for col in timepoints:
        perturbed_col = perturb_column(simulated_df[col], metric_medians)
        simulated_df[col] = perturbed_col

    # Add noise
    simulated_df = apply_statistical_noise(simulated_df)

    # LoR adjustment (assuming LoR is last row)
    lor_row = simulated_df.index[-1]
    lor_value = simulated_df.loc[lor_row].dropna().values[0]
    simulated_df.loc[lor_row] = perturb_loss_of_response(lor_value)

    # Restore timepoints
    simulated_df.columns = timepoints
    return simulated_df

def distribute_counts(n_real, total):
    high = math.ceil(total / n_real)
    low = math.floor(total / n_real)
    x = (total - (low * n_real)) // (high - low)
    y = n_real - x
    return x, y, high, low  # x gets high, y gets low

def generate_simulated_patient_task(args):
    """Generate a single simulated patient - designed for multiprocessing."""
    patient_name, real_df, synth_index, output_folder = args
    try:
        synth_df = generate_simulated_patient(real_df)
        synth_name = f"synth_{patient_name}_{synth_index}"
        output_path = os.path.join(output_folder, f"{synth_name}.xlsx")
        synth_df.to_excel(output_path, index=True)
        return synth_name, synth_df if synth_index == 1 else None  # Return df only for first simulated
    except Exception as e:
        logger.error(f"Error generating simulated patient {patient_name}_{synth_index}: {e}")
        return None, None

def plot_real_vs_simulated_comparison(real_df, simulated_df, patient_id, save_path=None):
    real_flat = real_df.values.flatten()
    synth_flat = simulated_df.values.flatten()

    # --- Violin Plot ---
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=[real_flat, synth_flat], palette=["#1f77b4", "#ff7f0e"])
    plt.xticks([0, 1], ["Real", "simulated"])
    plt.ylabel("Feature Value Distribution")
    plt.title(f"Distribution Comparison - Patient {patient_id}")
    if save_path:
        plt.savefig(os.path.join(save_path, f"{patient_id}_violin.png"), bbox_inches='tight', dpi=100)
    plt.close()  # Use close() instead of show() for non-interactive mode

    # --- Line Plot of Selected Features ---
    num_to_plot = min(4, real_df.shape[0])
    fig, axs = plt.subplots(num_to_plot, 1, figsize=(10, 2.5 * num_to_plot), sharex=True)
    if num_to_plot == 1:
        axs = [axs]
    for i in range(num_to_plot):
        feature_name = real_df.index[i]
        axs[i].plot(real_df.columns, real_df.iloc[i], label='Real', color='blue')
        axs[i].plot(simulated_df.columns, simulated_df.iloc[i], label='simulated', color='orange', linestyle='dashed')
        axs[i].set_title(f"{feature_name}")
        axs[i].legend()

    plt.suptitle(f"Time Series Comparison - Patient {patient_id}")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{patient_id}_lines.png"), bbox_inches='tight', dpi=100)
    plt.close()  # Use close() instead of show() for non-interactive mode

    # KS test
    ks_stat, p_value = ks_2samp(real_flat, synth_flat)
    logger.info(f"Patient {patient_id} - KS Statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")

# --- Main Generation ---
def main():
    # Log system information
    available_cores = log_system_info()
    start_time = time.time()
    
    real_patients = load_real_patients(REAL_FOLDER)
    real_patient_names = list(real_patients.keys())
    random.shuffle(real_patient_names)

    n_real = len(real_patient_names)
    if n_real == 0:
        raise ValueError("No real patients found. Please check your REAL_FOLDER.")

    x, y, high, low = distribute_counts(n_real, TOTAL_simulated)

    patients_high = real_patient_names[:x]
    patients_low = real_patient_names[x:]

    logger.info(f"\n➡️  Generating {TOTAL_simulated} simulated patients from {n_real} real patients")
    logger.info(f"   - {x} patients → {high} simulated each")
    logger.info(f"   - {y} patients → {low} simulated each")
    logger.info(f"Using {(cpu_count() - (cpu_count() // 4))} CPU cores for parallel generation\n")

    # Prepare all tasks for parallel processing
    tasks = []
    for patient_name in patients_high + patients_low:
        real_df = real_patients[patient_name]
        count = high if patient_name in patients_high else low
        
        for i in range(1, count + 1):
            tasks.append((patient_name, real_df, i, OUTPUT_FOLDER))

    logger.info(f"📋 Prepared {len(tasks)} simulated patient generation tasks")

    # Process tasks in parallel
    synth_counter = 0
    batch_counter = 1
    current_batch = []
    plot_tasks = []  # Store first simulateds for plotting

    with ProcessPoolExecutor(max_workers=(cpu_count() - (cpu_count() // 4))) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(generate_simulated_patient_task, task): task for task in tasks}
        
        # Collect results with progress tracking
        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), 
                          desc="Generating simulated patients"):
            result = future.result()
            if result[0] is not None:  # Successful generation
                synth_name, synth_df = result
                current_batch.append(synth_name)
                synth_counter += 1
                
                # Store first simulated for plotting
                if synth_df is not None:
                    patient_name = synth_name.split('_')[1]  # Extract original patient name
                    plot_tasks.append((real_patients[patient_name], synth_df, patient_name))

                if len(current_batch) == BATCH_SIZE:
                    logger.info(f"\n[Batch {batch_counter}] {len(current_batch)} patients saved")
                    current_batch = []
                    batch_counter += 1

    # Final batch summary
    if current_batch:
        logger.info(f"\n[Batch {batch_counter}] {len(current_batch)} patients saved")

    # Generate plots for first simulateds
    logger.info(f"\n📊 Generating comparison plots for {len(plot_tasks)} patients...")
    for real_df, synth_df, patient_name in plot_tasks:
        plot_real_vs_simulated_comparison(real_df, synth_df, patient_id=patient_name, save_path=VISUALS_FOLDER)

    # Performance summary
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"simulated generation complete!")
    logger.info(f"Total generated: {synth_counter} patients")
    logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Average time per simulated: {total_time/synth_counter:.3f} seconds")

if __name__ == "__main__":
    main()


