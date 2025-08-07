import json
import ollama
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from ollama import Client
import time
from scipy.interpolate import interp1d
import simdkalman

def load_trajectory_from_json(filename):
    """Loads trajectory data from a JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded trajectory data from {filename}.")
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filename}'. Check file format.")
        return None
    
    
def normalize_and_tokenize(data):
    def normalize(x):
        return int(round((x + 1) * 50))  # [-1,1] → [0,100]
    
    return ', '.join(f"{normalize(x):2d} {normalize(y):2d}" for x, y in data)


def predict_trajectory_with_ollama(given_trajectory_steps, num_future_steps=12, model_name=""):
    if not given_trajectory_steps:
        print("Error: No trajectory steps provided for prediction.")
        return None

    trajectory_input_str = str(normalize_and_tokenize(given_trajectory_steps))

    prompt = f"""
    Given the following sequence of 2D points representing a trajectory, where each point is formatted as a space-separated x y pair, and the pairs are separated by commas:
    {trajectory_input_str}

    Please predict the next {num_future_steps} future steps in this trajectory.
    Respond *only* with a Python list of lists, like [[x1, y1], [x2, y2], ...].
    Do NOT include any other text, explanations, markdown code blocks (```python), or line breaks outside the list structure.
    Ensure the output can be directly parsed by Python's ast.literal_eval().
    """
    print(prompt)

    print(f"\nSending prompt to Ollama model '{model_name}'...")
    print(f"Observed Trajectory: {trajectory_input_str}")

    # ① Create the client with a timeout setting
    client = Client(timeout=120)  # timeout in seconds

    def call_ollama():
        return client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0}
        )

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(call_ollama)
            # ② Wait up to 60s for the result
            response = future.result(timeout=120)
    except concurrent.futures.TimeoutError:
        print("❗ Ollama call timed out after 120 seconds")
        # Handle timeout: retry, fallback, or raise an exception
    except Exception as e:
        print("❗ Ollama call failed:", e)
    else:
        print(f"\n--- LLM Raw Output ({model_name}) ---")
        llm_output = response['message']['content']
        # Add a format change specifically for llama3:8b
        if model_name == "llama3:8b":
            llm_output = re.sub(r'(\d+)\s+(\d+)', r'\1, \2', llm_output)
        print(llm_output)
        print("-------------------------------------")

        try:
            match = re.search(r'\[\s*\[.*?\](?:\s*,\s*\[.*?\])*\s*\]', llm_output, re.DOTALL)

            if match:
                python_list_str = match.group(0)
                predicted_trajectory = ast.literal_eval(python_list_str)

                if isinstance(predicted_trajectory, list) and \
                   all(isinstance(p, list) and len(p) == 2 and all(isinstance(coord, (int, float)) for coord in p) for p in predicted_trajectory):

                    if len(predicted_trajectory) == num_future_steps:
                        print(f"\n--- LLM Cleaned Output ({model_name}) ---")
                        print(predicted_trajectory)
                        print("-------------------------------------")
                        denormalized_pred = [[round(x / 50 - 1, 4), round(y / 50 - 1, 4)] for x, y in predicted_trajectory]
                        print(f"\n--- LLM Cleaned Output After Denormalization ({model_name}) ---")
                        print(denormalized_pred)
                        print("-------------------------------------")
                        return denormalized_pred
                    else:
                        print(f"Error: Model '{model_name}' predicted {len(predicted_trajectory)} steps, but {num_future_steps} were requested.")
                        return None
                else:
                    print(f"Error: Parsed Python list from '{model_name}' is not in the expected format (list of 2D numerical points).")
                    return None
            else:
                print(f"Error: No valid Python list structure found in '{model_name}' LLM output after regex search.")
                return None

        except (ValueError, SyntaxError) as e:
            print(f"Error: Could not parse '{model_name}' LLM output as a Python list using ast.literal_eval. Detail: {e}")
            return None


def predict_linear_extrapolation(given_steps, num_future_steps=12):
    """
    Linear extrapolation using scipy's interp1d with fill_value='extrapolate'.
    Applies separately to x and y over uniform time steps.
    No clipping.
    """
    if len(given_steps) < 2:
        return None

    given_steps = np.asarray(given_steps, dtype=float)  # shape (n, 2)
    n = given_steps.shape[0]
    obs_frames = np.arange(n)
    pred_frames = np.arange(n, n + num_future_steps)

    preds = np.zeros((num_future_steps, 2), dtype=float)
    for j in range(2):  # for x and y
        f = interp1d(obs_frames, given_steps[:, j], fill_value='extrapolate')
        preds[:, j] = f(pred_frames)

    return preds.tolist()


def predict_kalman_constant_velocity(given_steps, num_future_steps=12):
    """
    Kalman filter with constant velocity model using simdkalman.
    Separately applies a 1D Kalman filter to x and y coordinates.
    """
    if len(given_steps) < 2:
        return None

    given_steps = np.asarray(given_steps, dtype=float)  # shape (n, 2)
    n = given_steps.shape[0]

    # Transpose shape to (1, n) for simdkalman input
    obs_x = given_steps[:, 0].reshape(1, -1)
    obs_y = given_steps[:, 1].reshape(1, -1)

    # Define 1D constant velocity Kalman filter
    kf = simdkalman.KalmanFilter(
        state_transition=np.array([[1, 1], [0, 1]]),
        process_noise=np.diag([0.01, 0.01]),
        observation_model=np.array([[1, 0]]),
        observation_noise=1e-3
    )

    # Predict future steps (num_future_steps) beyond observation
    pred_x = kf.predict(obs_x, num_future_steps).observations.mean.reshape(-1)
    pred_y = kf.predict(obs_y, num_future_steps).observations.mean.reshape(-1)

    # Stack into list of (x, y) points
    preds = np.stack([pred_x, pred_y], axis=1)
    return preds.tolist()



# --- Main execution ---
if __name__ == "__main__":

    # 1. Load all given trajectories from the JSON file
    json_filename = "my_trajectory_data_normalized.json"
    full_trajectory_data = load_trajectory_from_json(json_filename)

    if full_trajectory_data is None:
        print("Exiting due to an error loading trajectory data.")
    else:
        all_predictions_and_groundtruths = [] # To store data for all trajectories

        for i, traj in enumerate(full_trajectory_data):
            # Define how many steps from the end of the loaded trajectory to give to the LLM
            num_given_steps_for_llm = 8
            # Take the first `num_given_steps_for_llm` steps as input for the LLM
            given_steps_for_prediction = traj[:num_given_steps_for_llm]

            print(f"\nUsing the first {len(given_steps_for_prediction)} steps for prediction (Trajectory {i+1}):")
            print(given_steps_for_prediction)

            num_future_steps_to_predict = 12

            # 2. Predict using Llama3:8b with retry mechanism
            predicted_future_steps_llama = None
            max_retries = 5 # Define the maximum number of retries
            for attempt in range(1, max_retries + 1):
                print(f"\nAttempt {attempt}/{max_retries} for llama3:8b prediction...")
                predicted_future_steps_llama = predict_trajectory_with_ollama(
                    given_steps_for_prediction,
                    num_future_steps=num_future_steps_to_predict,
                    model_name="llama3:8b"
                )
                if predicted_future_steps_llama is not None:
                    print(f"Llama prediction successful on attempt {attempt}.")
                    break # Exit the loop if prediction is successful
                else:
                    print(f"Llama prediction failed on attempt {attempt}. Retrying in 1 second...")
                    time.sleep(1) # Wait for 1 second before the next retry

            if predicted_future_steps_llama is None:
                print(f"Llama prediction failed after {max_retries} attempts for Trajectory {i+1}.")


            # 3. Predict using deepseek-r1:1.5b with retry mechanism
            # predicted_future_steps_deepseek = None
            # max_retries = 5 # Define the maximum number of retries
            # for attempt in range(1, max_retries + 1):
            #     print(f"\nAttempt {attempt}/{max_retries} for deepseek-r1:1.5b prediction...")
            #     predicted_future_steps_deepseek = predict_trajectory_with_ollama(
            #         given_steps_for_prediction,
            #         num_future_steps=num_future_steps_to_predict,
            #         model_name="deepseek-r1:1.5b"
            #         )
            #     if predicted_future_steps_deepseek is not None:
            #         print(f"Deepseek prediction successful on attempt {attempt}.")
            #         break # Exit the loop if prediction is successful
            #     else:
            #         print(f"Deepseek prediction failed on attempt {attempt}. Retrying in 1 second...")
            #         time.sleep(1) # Wait for 1 second before the next retry

            # if predicted_future_steps_deepseek is None:
            #     print(f"Deepseek prediction failed after {max_retries} attempts for Trajectory {i+1}.")

            # 4. Predict using phi4:14b with retry mechanism
            predicted_future_steps_phi = None
            max_retries = 5 # Define the maximum number of retries
            for attempt in range(1, max_retries + 1):
                print(f"\nAttempt {attempt}/{max_retries} for phi4:14b prediction...")
                predicted_future_steps_phi = predict_trajectory_with_ollama(
                    given_steps_for_prediction,
                    num_future_steps=num_future_steps_to_predict,
                    model_name="phi4:14b"
                )
                if predicted_future_steps_phi is not None:
                    print(f"Phi prediction successful on attempt {attempt}.")
                    break # Exit the loop if prediction is successful
                else:
                    print(f"Phi prediction failed on attempt {attempt}. Retrying in 1 second...")
                    time.sleep(1) # Wait for 1 second before the next retry

            if predicted_future_steps_phi is None:
                print(f"Phi prediction failed after {max_retries} attempts for Trajectory {i+1}.")

            # 5. Predict using qwen3:1.7b
            # predicted_future_steps_qwen3 = None
            # max_retries = 5 # Define the maximum number of retries
            # for attempt in range(1, max_retries + 1):
            #     print(f"\nAttempt {attempt}/{max_retries} for qwen3:1.7b prediction...")
            #     predicted_future_steps_qwen3 = predict_trajectory_with_ollama(
            #         given_steps_for_prediction,
            #         num_future_steps=num_future_steps_to_predict,
            #         model_name="qwen3:1.7b"
            #     )
            #     if predicted_future_steps_qwen3 is not None:
            #         print(f"Qwen3 prediction successful on attempt {attempt}.")
            #         break # Exit the loop if prediction is successful
            #     else:
            #         print(f"Qwen3 prediction failed on attempt {attempt}. Retrying in 1 second...")
            #         time.sleep(1) # Wait for 1 second before the next retry

            # if predicted_future_steps_qwen3 is None:
            #     print(f"Qwen3 prediction failed after {max_retries} attempts for Trajectory {i+1}.")

            # >>> ADDED: Predict using direct linear extrapolation (no LLM)
            predicted_future_steps_linear = predict_linear_extrapolation(
                given_steps_for_prediction,
                num_future_steps=num_future_steps_to_predict
            )

            # >>> ADDED: Predict using constant-velocity 2D Kalman filter (no LLM)
            predicted_future_steps_kalman = predict_kalman_constant_velocity(
                given_steps_for_prediction,
                num_future_steps=num_future_steps_to_predict
            )

            ground_truth_steps = []
            if len(traj) > num_given_steps_for_llm:
                start_of_ground_truth = num_given_steps_for_llm
                end_of_ground_truth = num_given_steps_for_llm + num_future_steps_to_predict
                ground_truth_steps = traj[start_of_ground_truth:end_of_ground_truth]

            prediction_data = {
                "trajectory_index": i,
                "given_steps": given_steps_for_prediction,
                "ground_truth_steps": ground_truth_steps,
                "predicted_steps_llama3": predicted_future_steps_llama,
                # "predicted_steps_deepseek": predicted_future_steps_deepseek,
                "predicted_steps_phi": predicted_future_steps_phi,
                # "predicted_steps_qwen3": predicted_future_steps_qwen3,
                # >>> ADDED: Save classical baselines
                "predicted_steps_linear": predicted_future_steps_linear,
                "predicted_steps_kalman": predicted_future_steps_kalman
            }
            all_predictions_and_groundtruths.append(prediction_data)

            # Plotting for specific indices
            if i in (0, 1, 50, 51, 100, 101, 150, 151, 200, 201, 250, 251): # Example indices to plot
                plt.figure(figsize=(12, 8))

                # Convert lists to NumPy arrays for easier plotting
                given_np = np.array(given_steps_for_prediction)
                ground_truth_np = np.array(ground_truth_steps)

                # Plot given steps
                plt.plot(given_np[:, 0], given_np[:, 1], 'bo-', label='Given Steps (Input)', markersize=6)

                # Mark the last given step to show the handover point
                if given_np.shape[0] > 0:
                    plt.plot(given_np[-1, 0], given_np[-1, 1], 'ko', markersize=8, label='Last Given Step')

                # Plot ground truth steps
                if ground_truth_np.shape[0] > 0:
                    plt.plot(ground_truth_np[:, 0], ground_truth_np[:, 1], 'go-', label='Ground Truth Future Steps', markersize=6)

                # Plot predicted steps for Llama3
                if predicted_future_steps_llama:
                    predicted_llama_np = np.array(predicted_future_steps_llama)
                    plt.plot(predicted_llama_np[:, 0], predicted_llama_np[:, 1], 'rx--', label='Predicted Steps (Llama3:8b)', markersize=6)

                # # Plot predicted steps for Deepseek
                # if predicted_future_steps_deepseek:
                #     predicted_deepseek_np = np.array(predicted_future_steps_deepseek)
                #     plt.plot(predicted_deepseek_np[:, 0], predicted_deepseek_np[:, 1], 'cv-.', label='Predicted Steps (deepseek-r1:1.5b)', markersize=6)
                
                # Plot predicted steps for Phi4
                if predicted_future_steps_phi:
                    predicted_phi_np = np.array(predicted_future_steps_phi)
                    plt.plot(predicted_phi_np[:, 0], predicted_phi_np[:, 1], 'ms:', label='Predicted Steps (Phi4:14b)', markersize=6)

                # # Plot predicted steps for qwen3
                # if predicted_future_steps_qwen3:
                #     predicted_qwen_np = np.array(predicted_future_steps_qwen3)
                #     plt.plot(predicted_qwen_np[:, 0], predicted_qwen_np[:, 1], 'y-D', label='Predicted Steps (qwen3:1.7b)', markersize=6)

                # >>> ADDED: Plot direct linear extrapolation
                if predicted_future_steps_linear:
                    pred_lin_np = np.array(predicted_future_steps_linear)
                    plt.plot(pred_lin_np[:, 0], pred_lin_np[:, 1], 'p--', label='Predicted Steps (Linear Extrap.)', markersize=6)

                # >>> ADDED: Plot Kalman filter predictions
                if predicted_future_steps_kalman:
                    pred_kf_np = np.array(predicted_future_steps_kalman)
                    plt.plot(pred_kf_np[:, 0], pred_kf_np[:, 1], 'h-.', label='Predicted Steps (Kalman CV)', markersize=6)

                plt.title(f'Trajectory Prediction Comparison for Index {i}')
                plt.xlabel('X-coordinate')
                plt.ylabel('Y-coordinate')
                plt.legend()
                plt.grid(True)
                plt.tight_layout() # Adjust layout to prevent labels overlapping

                # Save the plot
                plot_filename = f"trajectory_prediction_comparison_plot_{i}.png"
                plt.savefig(plot_filename)
                plt.close() # Close the plot to free memory
                print(f"Comparison plot saved to {plot_filename}")


        # Save all predictions and their corresponding ground truths to a JSON file
        output_json_filename = "ollama_model_predictions.json"
        try:
            with open(output_json_filename, 'w') as f:
                json.dump(all_predictions_and_groundtruths, f, indent=4)
            print(f"\nSuccessfully saved all predictions and ground truths to {output_json_filename}")
        except IOError as e:
            print(f"Error saving predictions to file: {e}")
