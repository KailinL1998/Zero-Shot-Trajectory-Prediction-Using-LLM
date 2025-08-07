import json
import numpy as np

# Change to the correct input filename as per the provided JSON structure
input_json_filename = 'ollama_model_predictions.json'

try:
    with open(input_json_filename, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: '{input_json_filename}' not found. Please ensure the file is in the correct directory.")
    exit() # Exit if the file isn't found, as we can't proceed without it.

def calculate_ade(predicted_steps, ground_truth_steps):
    """
    Calculates the Average Displacement Error (ADE) between predicted and ground truth trajectories.
    ADE is the average Euclidean distance between corresponding points in the trajectories.
    """
    displacements = []
    # Ensure both lists have the same length for accurate comparison
    min_len = min(len(predicted_steps), len(ground_truth_steps))
    for i in range(min_len):
        pred_point = np.array(predicted_steps[i])
        gt_point = np.array(ground_truth_steps[i])
        displacements.append(np.linalg.norm(pred_point - gt_point))
    return np.mean(displacements) if displacements else 0.0

def calculate_fde(predicted_steps, ground_truth_steps):
    """
    Calculates the Final Displacement Error (FDE) between predicted and ground truth trajectories.
    FDE is the Euclidean distance between the last points of the trajectories.
    """
    if not predicted_steps or not ground_truth_steps:
        return 0.0 # Return 0 if either list is empty or invalid to avoid errors

    # Ensure both lists have at least one point for FDE calculation
    if len(predicted_steps) == 0 or len(ground_truth_steps) == 0:
        return 0.0

    pred_final_point = np.array(predicted_steps[-1])
    gt_final_point = np.array(ground_truth_steps[-1])
    return np.linalg.norm(pred_final_point - gt_final_point)

# Function to calculate Standard Error of the Mean (SEM)
def calculate_sem(data_list):
    """
    Calculates the Standard Error of the Mean (SEM) for a list of data.
    Returns "N/A" if the list is empty or has only one element (as SEM is not meaningful).
    """
    if not data_list or len(data_list) < 2:
        return "N/A"
    return np.std(data_list) / np.sqrt(len(data_list))

# Define the trajectory index ranges you're interested in
# ranges = {
#     "0-9": (0, 9),
#     "10-19": (10, 19),
#     "20-29": (20, 29),
#     "30-39": (30, 39),
#     "40-49": (40, 49),
#     "50-59": (50, 59),
#     "60-69": (60, 69),
# }

ranges = {
    "0-49": (0, 49),
    "50-99": (50, 99),
    "100-149": (100, 149),
    "150-199": (150, 199),
    "200-249": (200, 249),
    "250-299": (250, 299),
}

results = {}

# Process data for each defined range
for name, (start, end) in ranges.items():
    # Filter the data to include only trajectories within the current range
    range_data = [d for d in data if start <= d["trajectory_index"] <= end]

    if not range_data:
        results[name] = {
            "llama3": {"ADE_mean": "N/A", "ADE_sem": "N/A", "FDE_mean": "N/A", "FDE_sem": "N/A",
                       "invalid_rate": "N/A"},
            "linear": {"ADE_mean": "N/A", "ADE_sem": "N/A", "FDE_mean": "N/A", "FDE_sem": "N/A",
                       "invalid_rate": "N/A"},           
            # "deepseek": {"ADE_mean": "N/A", "ADE_sem": "N/A", "FDE_mean": "N/A", "FDE_sem": "N/A",
                        #  "invalid_rate": "N/A"},
            "phi4": {"ADE_mean": "N/A", "ADE_sem": "N/A", "FDE_mean": "N/A", "FDE_sem": "N/A",
                       "invalid_rate": "N/A"},
            "kalman": {"ADE_mean": "N/A", "ADE_sem": "N/A", "FDE_mean": "N/A", "FDE_sem": "N/A",
                       "invalid_rate": "N/A"},
            # "qwen3": {"ADE_mean": "N/A", "ADE_sem": "N/A", "FDE_mean": "N/A", "FDE_sem": "N/A",
                        #  "invalid_rate": "N/A"},
        }
        print(f"No data found for trajectory_index range {name}. Skipping calculations for this range.")
        continue

    # Lists to store raw and normalized metrics for each model
    ades_llama = []
    fdes_llama = []
    invalid_llama_count = 0
    ades_linear = []
    fdes_linear = []
    invalid_linear_count = 0
    ades_phi4 = []
    fdes_phi4 = []
    invalid_phi4_count = 0
    ades_kalman = []
    fdes_kalman = []
    invalid_kalman_count = 0

    # ades_deepseek = []
    # fdes_deepseek = []
    # invalid_deepseek_count = 0
    # ades_qwen3 = []
    # fdes_qwen3 = []
    # invalid_qwen3_count = 0

    total_trajectories_in_range = len(range_data)

    for item in range_data:
        ground_truth_steps = item.get("ground_truth_steps", [])

        # Process Llama3 predictions
        predicted_steps_llama = item.get("predicted_steps_llama3")
        if predicted_steps_llama is None or not ground_truth_steps:
            invalid_llama_count += 1
        else:
            current_ade = calculate_ade(predicted_steps_llama, ground_truth_steps)
            current_fde = calculate_fde(predicted_steps_llama, ground_truth_steps)
            ades_llama.append(current_ade)
            fdes_llama.append(current_fde)

        # Process Deepseek predictions
        # predicted_steps_deepseek = item.get("predicted_steps_deepseek")
        # if predicted_steps_deepseek is None or not ground_truth_steps:
        #     invalid_deepseek_count += 1
        # else:
        #     current_ade = calculate_ade(predicted_steps_deepseek, ground_truth_steps)
        #     current_fde = calculate_fde(predicted_steps_deepseek, ground_truth_steps)
        #     ades_deepseek.append(current_ade)
        #     fdes_deepseek.append(current_fde)

        # Process Linear predictions
        predicted_steps_linear = item.get("predicted_steps_linear")
        if predicted_steps_linear is None or not ground_truth_steps:
            invalid_linear_count += 1
        else:
            current_ade = calculate_ade(predicted_steps_linear, ground_truth_steps)
            current_fde = calculate_fde(predicted_steps_linear, ground_truth_steps)
            ades_linear.append(current_ade)
            fdes_linear.append(current_fde)

        # Process Phi4 predictions
        predicted_steps_phi4 = item.get("predicted_steps_phi")
        if predicted_steps_phi4 is None or not ground_truth_steps:
            invalid_phi4_count += 1
        else:
            current_ade = calculate_ade(predicted_steps_phi4, ground_truth_steps)
            current_fde = calculate_fde(predicted_steps_phi4, ground_truth_steps)
            ades_phi4.append(current_ade)
            fdes_phi4.append(current_fde)

        # Process Qwen3 predictions
        # predicted_steps_qwen3 = item.get("predicted_steps_qwen3")
        # if predicted_steps_qwen3 is None or not ground_truth_steps:
        #     invalid_qwen3_count += 1
        # else:
        #     current_ade = calculate_ade(predicted_steps_qwen3, ground_truth_steps)
        #     current_fde = calculate_fde(predicted_steps_qwen3, ground_truth_steps)
        #     ades_qwen3.append(current_ade)
        #     fdes_qwen3.append(current_fde)

         # Process Kalman predictions
        predicted_steps_kalman = item.get("predicted_steps_kalman")
        if predicted_steps_kalman is None or not ground_truth_steps:
            invalid_kalman_count += 1
        else:
            current_ade = calculate_ade(predicted_steps_kalman, ground_truth_steps)
            current_fde = calculate_fde(predicted_steps_kalman, ground_truth_steps)
            ades_kalman.append(current_ade)
            fdes_kalman.append(current_fde)


    # Calculate metrics for Llama3
    results_llama = {}
    results_llama["invalid_rate"] = (invalid_llama_count / total_trajectories_in_range) * 100 if total_trajectories_in_range > 0 else "N/A"

    results_llama["ADE_mean"] = np.mean(ades_llama) if ades_llama else "N/A"
    results_llama["ADE_sem"] = calculate_sem(ades_llama)
    results_llama["FDE_mean"] = np.mean(fdes_llama) if fdes_llama else "N/A"
    results_llama["FDE_sem"] = calculate_sem(fdes_llama)

    # Calculate metrics for Deepseek
    # results_deepseek = {}
    # results_deepseek["invalid_rate"] = (invalid_deepseek_count / total_trajectories_in_range) * 100 if total_trajectories_in_range > 0 else "N/A"

    # results_deepseek["ADE_mean"] = np.mean(ades_deepseek) if ades_deepseek else "N/A"
    # results_deepseek["ADE_sem"] = calculate_sem(ades_deepseek)
    # results_deepseek["FDE_mean"] = np.mean(fdes_deepseek) if fdes_deepseek else "N/A"
    # results_deepseek["FDE_sem"] = calculate_sem(fdes_deepseek)

    # Calculate metrics for Linear
    results_linear = {}
    results_linear["invalid_rate"] = (invalid_linear_count / total_trajectories_in_range) * 100 if total_trajectories_in_range > 0 else "N/A"

    results_linear["ADE_mean"] = np.mean(ades_linear) if ades_linear else "N/A"
    results_linear["ADE_sem"] = calculate_sem(ades_linear)
    results_linear["FDE_mean"] = np.mean(fdes_linear) if fdes_linear else "N/A"
    results_linear["FDE_sem"] = calculate_sem(fdes_linear)

    # Calculate metrics for Llama3
    results_phi4 = {}
    results_phi4["invalid_rate"] = (invalid_phi4_count / total_trajectories_in_range) * 100 if total_trajectories_in_range > 0 else "N/A"

    results_phi4["ADE_mean"] = np.mean(ades_phi4) if ades_phi4 else "N/A"
    results_phi4["ADE_sem"] = calculate_sem(ades_phi4)
    results_phi4["FDE_mean"] = np.mean(fdes_phi4) if fdes_phi4 else "N/A"
    results_phi4["FDE_sem"] = calculate_sem(fdes_phi4)

    # Calculate metrics for Deepseek
    # results_qwen3 = {}
    # results_qwen3["invalid_rate"] = (invalid_qwen3_count / total_trajectories_in_range) * 100 if total_trajectories_in_range > 0 else "N/A"

    # results_qwen3["ADE_mean"] = np.mean(ades_qwen3) if ades_qwen3 else "N/A"
    # results_qwen3["ADE_sem"] = calculate_sem(ades_qwen3)
    # results_qwen3["FDE_mean"] = np.mean(ades_qwen3) if ades_qwen3 else "N/A"
    # results_qwen3["FDE_sem"] = calculate_sem(ades_qwen3)

    # Calculate metrics for Kalman
    results_kalman = {}
    results_kalman["invalid_rate"] = (invalid_kalman_count / total_trajectories_in_range) * 100 if total_trajectories_in_range > 0 else "N/A"

    results_kalman["ADE_mean"] = np.mean(ades_kalman) if ades_kalman else "N/A"
    results_kalman["ADE_sem"] = calculate_sem(ades_kalman)
    results_kalman["FDE_mean"] = np.mean(fdes_kalman) if fdes_kalman else "N/A"
    results_kalman["FDE_sem"] = calculate_sem(fdes_kalman)

    results[name] = {
        "llama3": results_llama,
        # "deepseek": results_deepseek,
        "linear": results_linear,
        "phi4": results_phi4,
        # "qwen3": results_qwen3,
        "kalman": results_kalman
    }

# Print the computed results
print(f"\n--- Trajectory Prediction Evaluation Results from '{input_json_filename}' ---")
for r, models_data in results.items():
    print(f"\nFor Trajectory Index Range {r}:")
    for model_name, metrics in models_data.items():
        print(f"  Model: {model_name.upper()}:")
        if isinstance(metrics["invalid_rate"], float):
            print(f"    Invalid Prediction Rate: {metrics['invalid_rate']:.2f}%")
        else:
            print(f"    Invalid Prediction Rate: {metrics['invalid_rate']}")

        # Raw Metrics
        if isinstance(metrics["ADE_mean"], float):
            print(f"    Raw ADE: {metrics['ADE_mean']:.2f} (SEM: {metrics['ADE_sem']:.2f})" if isinstance(metrics['ADE_sem'], float) else f"    Raw ADE: {metrics['ADE_mean']:.2f} (SEM: {metrics['ADE_sem']})")
        else:
            print(f"    Raw ADE: {metrics['ADE_mean']} (SEM: {metrics['ADE_sem']})")

        if isinstance(metrics["FDE_mean"], float):
            print(f"    Raw FDE: {metrics['FDE_mean']:.2f} (SEM: {metrics['FDE_sem']:.2f})" if isinstance(metrics['FDE_sem'], float) else f"    Raw FDE: {metrics['FDE_mean']:.2f} (SEM: {metrics['FDE_sem']})")
        else:
            print(f"    Raw FDE: {metrics['FDE_mean']} (SEM: {metrics['FDE_sem']})")

print("\n--- End of Evaluation ---")