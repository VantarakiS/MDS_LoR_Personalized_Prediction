import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import csv
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.spatial.distance import euclidean, cosine
from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error
import logging
import time
import re
import uuid
import json

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
    logger.info(f"Process will use nested parallelization for optimal CPU utilization")
    return cpu_cores


# ---------- NORMALIZATION & IMPUTATION ---------- #
def normalize_to_minus1_plus1(series):
    series = pd.to_numeric(series, errors='coerce')
    shifted = series - series.iloc[0]
    max_abs = shifted.abs().max()
    if max_abs == 0 or pd.isna(max_abs):
        return pd.Series(0, index=series.index)
    return shifted / max_abs

def impute_row(row):
    num_missing = row.isna().sum()
    missing_ratio = num_missing / len(row)
    if missing_ratio < 0.25:
        return row.interpolate(method='linear', limit_direction='both')
    else:
        return row.fillna(row.median())

# ---------- LOAD SINGLE PATIENT ---------- #
def load_patient(filepath):
    df = pd.read_excel(filepath, header=None)
    timestamps = df.iloc[0, 1:].values
    metrics = df.iloc[1:29, 0].values  # First 28 rows are dynamic metrics
    data = df.iloc[1:29, 1:]

    clean_df = pd.DataFrame(data.values, columns=timestamps, index=metrics)
    clean_df = clean_df.apply(pd.to_numeric, errors='coerce')

    # Separate Loss response before processing
    lor_row = clean_df.loc['Loss response'] if 'Loss response' in clean_df.index else None
    to_process = clean_df.drop(index='Loss response') if lor_row is not None else clean_df

    # Impute
    imputed_df = to_process.apply(impute_row, axis=1)

    # Reinsert Loss response row untouched
    if lor_row is not None:
        imputed_df.loc['Loss response'] = lor_row

    # Normalize for plotting (excluding Loss response)
    norm_df = imputed_df.drop(index='Loss response') if 'Loss response' in imputed_df.index else imputed_df
    norm_df = norm_df.apply(normalize_to_minus1_plus1, axis=1)
    if lor_row is not None:
        norm_df.loc['Loss response'] = lor_row

    return imputed_df, norm_df

def process_single_patient_file(args):
    """Process a single patient file - designed for multiprocessing."""
    filepath, patient_files_path, main_output_path = args
    try:
        patient_id = os.path.basename(filepath).split('.')[0]
        
        # Load and process patient data
        imputed_df, normalized_df = load_patient(filepath)
        
        # Save imputed data
        imputed_folder = os.path.join(main_output_path, "imputed")
        os.makedirs(imputed_folder, exist_ok=True)
        output_path = os.path.join(imputed_folder, f"{patient_id}_imputed_only.xlsx")
        imputed_df.to_excel(output_path)
        
        return patient_id, imputed_df, normalized_df
    except Exception as e:
        logger.error(f"Error processing patient file {filepath}: {e}")
        return None

def generate_patient_plots(args):
    """Generate plots for a single patient - designed for multiprocessing."""
    patient_id, dynamic_df, save_folder = args
    try:
        patient_folder = os.path.join(save_folder, patient_id)
        os.makedirs(patient_folder, exist_ok=True)
        
        plots_generated = 0
        # Use Agg backend for thread safety in multiprocessing
        plt.switch_backend('Agg')
        
        for metric_name in dynamic_df.index:
            try:
                time_points = dynamic_df.columns.astype(float)
                values = dynamic_df.loc[metric_name].values
                
                if len(time_points) == len(values) and not pd.isna(values).all():
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(time_points, values, linewidth=2, marker='o', markersize=4)
                    ax.set_title(f"Patient {patient_id} - Metric: {metric_name}", fontsize=12)
                    ax.set_xlabel("Time (Months)", fontsize=10)
                    ax.set_ylabel("Normalized Value", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    # Clean filename
                    filename_safe = re.sub(r'[<>:"/\\|?*]', '_', metric_name)
                    filepath = os.path.join(patient_folder, f"{filename_safe}.png")
                    
                    plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    plots_generated += 1
                else:
                    logger.debug(f"Skipping {metric_name} for patient {patient_id}: invalid data")
            except Exception as e:
                logger.warning(f"Error plotting {metric_name} for patient {patient_id}: {e}")
                continue
        
        return patient_id, plots_generated
    except Exception as e:
        logger.error(f"Error generating plots for patient {patient_id}: {e}")
        return patient_id, 0

# ---------- PLOTTING UTILS ---------- #
def plot_missingness(df, title='Missingness'):
    plt.figure(figsize=(12,6))
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if df.empty or df.size == 0:
        logger.info(f"Skipping {title} plot - DataFrame is empty")
        plt.close()
        return
    sns.heatmap(df.isnull(), cbar=False)
    plt.title(title)
    plt.xlabel("Measurements")
    plt.ylabel("Metrics or Patients")
    plt.tight_layout()
    plt.show()

def plot_patient_metric_time_series_parallel(patients_dynamic, save_folder=None, max_workers=None):
    """Generate time series plots for all patients in parallel."""
    if not patients_dynamic:
        logger.warning("No patient data to plot")
        return
    
    if max_workers is None:
        max_workers = min((cpu_count() - (cpu_count() // 4)), len(patients_dynamic))
    
    # Prepare arguments for parallel processing
    plot_args = [(patient_id, dynamic_df, save_folder) 
                 for patient_id, dynamic_df in patients_dynamic.items()]
    
    total_plots = 0
    logger.info(f"Generating plots for {len(patients_dynamic)} patients using {max_workers} workers")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all plotting tasks
        future_to_patient = {executor.submit(generate_patient_plots, args): args[0] 
                           for args in plot_args}
        
        # Collect results with progress tracking
        for future in tqdm(as_completed(future_to_patient), total=len(future_to_patient), 
                          desc="Generating patient plots"):
            patient_id, plots_count = future.result()
            total_plots += plots_count
            
    logger.info(f"Generated {total_plots} plots across {len(patients_dynamic)} patients")

# ---------- MAIN EDA FUNCTION ---------- #
def full_eda(folder_path):
    # Log system information
    available_cores = log_system_info()
    start_time = time.time()
    
    patients_imputed = {}
    patients_normalized = {}
    patient_files_path = os.path.join(folder_path, "real")

    # Get all patient files
    patient_files = []
    for filename in os.listdir(patient_files_path):
        if filename.endswith('.xlsx') and filename.startswith('P'):
            filepath = os.path.join(patient_files_path, filename)
            patient_files.append((filepath, patient_files_path, os.path.join(folder_path, "output")))
    
    if not patient_files:
        logger.warning("No patient files found in the specified directory")
        return
    
    logger.info(f"Found {len(patient_files)} patient files to process")
    
    # Process all patients in parallel
    max_workers = min((cpu_count() - (cpu_count() // 4)), len(patient_files))
    logger.info(f"Processing patients using {max_workers} workers")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all patient processing tasks
        future_to_file = {executor.submit(process_single_patient_file, args): args[0] 
                         for args in patient_files}
        
        # Collect results with progress tracking
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), 
                          desc="Processing patient files"):
            result = future.result()
            if result is not None:
                patient_id, imputed_df, normalized_df = result
                patients_imputed[patient_id] = imputed_df
                patients_normalized[patient_id] = normalized_df

    logger.info(f"Successfully loaded {len(patients_imputed)} patients.")
    
    if not patients_imputed:
        logger.error("No patients were successfully processed")
        return

    logger.info("Analyzing data missingness...")
    dynamic_df_all = pd.concat(patients_imputed, axis=0)
    missingness_stats = dynamic_df_all.isnull().mean() * 100
    logger.debug(f"\nDynamic metrics missingness (%):\n{missingness_stats}")
    
    # Plot missingness
    plot_missingness(dynamic_df_all, title='Dynamic Metrics Missingness')

    # Generate plots in parallel
    plots_folder = os.path.join(folder_path, "output/plots")
    os.makedirs(plots_folder, exist_ok=True)
    
    logger.info("Generating time series plots...")
    plot_patient_metric_time_series_parallel(patients_normalized, save_folder=plots_folder)
    
    # Performance summary
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total EDA execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Average time per patient: {total_time/len(patients_imputed):.2f} seconds")
    
    return patients_imputed, patients_normalized

def main():
    try:
        folder_path = r'path\to\your\folder'
        if not os.path.exists(folder_path):
            logger.error("Please set the correct folder path to run the EDA.")
            exit(1)
        
        logger.info("Starting Enhanced Multi-threaded EDA Analysis")
        logger.info("=" * 60)
        
        # Check if real folder exists
        real_folder = os.path.join(folder_path, "real_patients") #change folder
        if not os.path.exists(real_folder):
            logger.error(f"Real data folder not found at: {real_folder}")
            exit(1)
            
        # Count available files
        patient_count = len([f for f in os.listdir(real_folder) 
                           if f.endswith('.xlsx') and f.startswith('P')])
        logger.info(f"Found {patient_count} patient files to process")
        
        if patient_count == 0:
            logger.warning("No patient files found to process")
            exit(0)
        
        # Run the main execution function
        result = full_eda(folder_path)
        
        if result:
            patients_imputed, patients_normalized = result
            logger.info("=" * 60)
            logger.info(f"EDA completed successfully!")
            logger.info(f"Processed {len(patients_imputed)} patients")
            logger.info(f"Generated plots for {len(patients_normalized)} patients")
        else:
            logger.error("EDA failed to complete")
            
        
    except Exception as e:
        logger.error(f"Critical error in EDA execution: {e}")
        exit(1)

# ---------- RUN ---------- #
if __name__ == "__main__":
    main()