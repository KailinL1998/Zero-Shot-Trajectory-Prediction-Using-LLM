import numpy as np
import matplotlib.pyplot as plt
import json

import numpy as np

def normalize_and_tokenize(data):
    def normalize(x):
        return int(round((x + 1) * 50))  # [-1,1] → [0,100]
    
    return ', '.join(f"{normalize(x):2d} {normalize(y):2d}" for x, y in data)

def normalize_trajectory(trajectory):
    """
    Normalizes a 2D trajectory to fit within the [-1, 1] x [-1, 1] scale.

    Args:
        trajectory (list): A list of [x, y] positions.

    Returns:
        list: The normalized list of [x, y] positions.
    """
    if not trajectory:
        return []

    x_coords = [p[0] for p in trajectory]
    y_coords = [p[1] for p in trajectory]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    normalized_trajectory = []
    for x, y in trajectory:
        # Normalize x
        if max_x == min_x: # Handle the case of a vertical line or single point
            norm_x = 0.0
        else:
            norm_x = 2 * ((x - min_x) / (max_x - min_x)) - 1

        # Normalize y
        if max_y == min_y: # Handle the case of a horizontal line or single point
            norm_y = 0.0
        else:
            norm_y = 2 * ((y - min_y) / (max_y - min_y)) - 1

        normalized_trajectory.append([np.round(norm_x, 4), np.round(norm_y, 4)])

    return normalized_trajectory


def generate_straight_line_trajectory(
    start_pos,
    direction_vector,
    num_steps,
    initial_speed=1.0,
    acceleration=0.0,
    speed_variance=0.0,
    constant_speed=True
):
    """
    Generates a straight-line pedestrian trajectory.

    Args:
        start_pos (list): [x, y] starting position.
        direction_vector (list): [dx, dy] normalized direction vector.
        num_steps (int): Number of time steps in the trajectory.
        initial_speed (float): Initial speed of the pedestrian.
        acceleration (float): Constant acceleration (positive for speeding up).
        speed_variance (float): Standard deviation for speed variation (if not constant_speed).
        constant_speed (bool): If True, speed is constant. If False, speed varies.

    Returns:
        list: A list of [x, y] positions representing the trajectory.
    """
    trajectory = []
    current_pos = np.array(start_pos, dtype=float)
    direction_vector = np.array(direction_vector, dtype=float)
    direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize

    current_speed = initial_speed

    for _ in range(num_steps):
        trajectory.append(np.round(current_pos, 4).tolist())

        if not constant_speed:
            # Varying speed: introduce random fluctuation
            current_speed += np.random.normal(0, speed_variance)
            current_speed = max(0.1, current_speed)  # Ensure speed doesn't drop too low

        displacement = current_speed * direction_vector
        current_pos += displacement
        current_speed += acceleration  # Apply acceleration for next step

    return trajectory

def generate_quadratic_trajectory(start_pos, initial_direction_vector, num_steps, initial_speed=1.0, curvature_factor=0.01):
    """
    Generates a quadratic pedestrian trajectory.

    Args:
        start_pos (list): [x, y] starting position.
        initial_direction_vector (list): [dx, dy] normalized initial direction vector.
        num_steps (int): Number of time steps in the trajectory.
        initial_speed (float): Initial speed of the pedestrian.
        curvature_factor (float): Factor determining the strength of the quadratic curve.

    Returns:
        list: A list of [x, y] positions representing the trajectory.
    """
    trajectory = []
    pos = np.array(start_pos, dtype=float)
    dir_vec = np.array(initial_direction_vector, dtype=float)
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    t = 0
    for _ in range(num_steps):
        offset = curvature_factor * (t ** 2)
        # Add curvature orthogonal to direction vector
        ortho_vec = np.array([-dir_vec[1], dir_vec[0]])
        curved_pos = pos + dir_vec * initial_speed * t + offset * ortho_vec
        trajectory.append(np.round(curved_pos, 4).tolist())
        t += 1
    return trajectory

def generate_cubic_trajectory(start_pos, initial_direction_vector, num_steps, initial_speed=1.0, curvature_factor=0.001):
    """
    Generates a cubic pedestrian trajectory.

    Args:
        start_pos (list): [x, y] starting position.
        initial_direction_vector (list): [dx, dy] normalized initial direction vector.
        num_steps (int): Number of time steps in the trajectory.
        initial_speed (float): Initial speed of the pedestrian.
        curvature_factor (float): Factor determining the strength of the cubic curve.

    Returns:
        list: A list of [x, y] positions representing the trajectory.
    """
    trajectory = []
    pos = np.array(start_pos, dtype=float)
    dir_vec = np.array(initial_direction_vector, dtype=float)
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    t = 0
    for _ in range(num_steps):
        offset = curvature_factor * (t ** 3)
        ortho_vec = np.array([-dir_vec[1], dir_vec[0]])
        curved_pos = pos + dir_vec * initial_speed * t + offset * ortho_vec
        trajectory.append(np.round(curved_pos, 4).tolist())
        t += 1
    return trajectory

def generate_sine_trajectory(start_pos, initial_direction_vector, num_steps, initial_speed=1.0, amplitude=1.0, frequency=0.2):
    """
    Generates a sine-wave like pedestrian trajectory.

    Args:
        start_pos (list): [x, y] starting position.
        initial_direction_vector (list): [dx, dy] normalized initial direction vector.
        num_steps (int): Number of time steps in the trajectory.
        initial_speed (float): Initial speed of the pedestrian.
        amplitude (float): Amplitude of the sine wave.
        frequency (float): Frequency of the sine wave.

    Returns:
        list: A list of [x, y] positions representing the trajectory.
    """
    trajectory = []
    pos = np.array(start_pos, dtype=float)
    dir_vec = np.array(initial_direction_vector, dtype=float)
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    ortho_vec = np.array([-dir_vec[1], dir_vec[0]])
    for t in range(num_steps):
        main_motion = dir_vec * initial_speed * t
        sine_offset = amplitude * np.sin(2 * np.pi * frequency * t)
        curved_pos = pos + main_motion + sine_offset * ortho_vec
        trajectory.append(np.round(curved_pos, 4).tolist())
    return trajectory

def generate_cosine_trajectory(start_pos, initial_direction_vector, num_steps, initial_speed=1.0, amplitude=1.0, frequency=0.2):
    """
    Generates a cosine-wave like pedestrian trajectory.

    Args:
        start_pos (list): [x, y] starting position.
        initial_direction_vector (list): [dx, dy] normalized initial direction vector.
        num_steps (int): Number of time steps in the trajectory.
        initial_speed (float): Initial speed of the pedestrian.
        amplitude (float): Amplitude of the cosine wave.
        frequency (float): Frequency of the cosine wave.

    Returns:
        list: A list of [x, y] positions representing the trajectory.
    """
    trajectory = []
    pos = np.array(start_pos, dtype=float)
    dir_vec = np.array(initial_direction_vector, dtype=float)
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    ortho_vec = np.array([-dir_vec[1], dir_vec[0]])
    for t in range(num_steps):
        main_motion = dir_vec * initial_speed * t
        cosine_offset = amplitude * np.cos(2 * np.pi * frequency * t)
        curved_pos = pos + main_motion + cosine_offset * ortho_vec
        trajectory.append(np.round(curved_pos, 4).tolist())
    return trajectory

def generate_pedestrian_dataset(num_samples_per_type=1, trajectory_length=20):
    """
    Generates a synthetic pedestrian trajectory dataset with varying complexities,
    with all trajectories normalized to a [-1, 1] scale.

    Args:
        num_samples_per_type (int): Number of trajectories to generate for each type.
        trajectory_length (int): Length of each trajectory (number of timestamps).

    Returns:
        list: A list of normalized trajectories, where each trajectory is a list of [x, y] positions.
    """
    dataset = []

    # --- Part 1: Straight Trajectories Constant Speed ---
    print(f"Generating {num_samples_per_type} samples with constant speed...")
    for i in range(num_samples_per_type):
        start_x = np.random.uniform(-5, 5) # Increased range for more diverse initial positions
        start_y = np.random.uniform(-5, 5)
        direction_x = np.random.uniform(-1, 1)
        direction_y = np.random.uniform(-1, 1)
        initial_speed = np.random.uniform(0.5, 1.5)
        trajectory = generate_straight_line_trajectory(
            start_pos=[start_x, start_y],
            direction_vector=[direction_x, direction_y],
            num_steps=trajectory_length,
            initial_speed=initial_speed,
            constant_speed=True
        )
        dataset.append(normalize_trajectory(trajectory))

    # --- Part 2: Straight Trajectories Positive Constant Acceleration ---
    print(f"Generating {num_samples_per_type} samples with positive constant acceleration...")
    for i in range(num_samples_per_type):
        start_x = np.random.uniform(-5, 5)
        start_y = np.random.uniform(-5, 5)
        direction_x = np.random.uniform(-1, 1)
        direction_y = np.random.uniform(-1, 1)
        initial_speed = np.random.uniform(0.5, 1.5)
        acceleration = np.random.uniform(0.02, 0.1)
        trajectory = generate_straight_line_trajectory(
            start_pos=[start_x, start_y],
            direction_vector=[direction_x, direction_y],
            num_steps=trajectory_length,
            initial_speed=initial_speed,
            acceleration=acceleration,
            constant_speed=True
        )
        dataset.append(normalize_trajectory(trajectory))

    # # --- Part 3: Straight Trajectories Varying Speed ---
    # print(f"Generating {num_samples_per_type} samples with varying speed...")
    # for i in range(num_samples_per_type):
    #     start_x = np.random.uniform(-5, 5)
    #     start_y = np.random.uniform(-5, 5)
    #     direction_x = np.random.uniform(-1, 1)
    #     direction_y = np.random.uniform(-1, 1)
    #     initial_speed = np.random.uniform(1.0, 2.5)
    #     speed_variance = np.random.uniform(0.1, 0.5)
    #     trajectory = generate_straight_line_trajectory(
    #         start_pos=[start_x, start_y],
    #         direction_vector=[direction_x, direction_y],
    #         num_steps=trajectory_length,
    #         initial_speed=initial_speed,
    #         speed_variance=speed_variance,
    #         constant_speed=False
    #     )
    #     dataset.append(normalize_trajectory(trajectory))

    # --- Part 4: Quadratic Trajectories ---
    print(f"Generating {num_samples_per_type} samples with quadratic trajectory...")
    for i in range(num_samples_per_type):
        start_x = np.random.uniform(-2, 2)
        start_y = np.random.uniform(-2, 2)
        direction_x = np.random.uniform(-0.1, 0.1)
        direction_y = np.random.uniform(3.0, 10.0) # Larger Y direction for stronger curve along X
        initial_speed = np.random.uniform(0.8, 1.2)
        curvature_factor = np.random.uniform(1.0, 5.0)
        trajectory = generate_quadratic_trajectory(
            start_pos=[start_x, start_y],
            initial_direction_vector=[direction_x, direction_y],
            num_steps=trajectory_length,
            initial_speed=initial_speed,
            curvature_factor=curvature_factor
        )
        dataset.append(normalize_trajectory(trajectory))

    # --- Part 5: Cubic Trajectories ---
    print(f"Generating {num_samples_per_type} samples with cubic trajectory...")
    for i in range(num_samples_per_type):
        start_x = np.random.uniform(-2, 2)
        start_y = np.random.uniform(-2, 2)
        direction_x = np.random.uniform(-0.1, 0.1)
        direction_y = np.random.uniform(3.0, 10.0)
        initial_speed = np.random.uniform(0.8, 1.2)
        curvature_factor = np.random.uniform(0.2, 1.0)
        trajectory = generate_cubic_trajectory(
            start_pos=[start_x, start_y],
            initial_direction_vector=[direction_x, direction_y],
            num_steps=trajectory_length,
            initial_speed=initial_speed,
            curvature_factor=curvature_factor
        )
        dataset.append(normalize_trajectory(trajectory))

    # --- Part 6: Sine Trajectories ---
    print(f"Generating {num_samples_per_type} samples with sine trajectory...")
    for i in range(num_samples_per_type):
        start_x = np.random.uniform(-5, 5)
        start_y = np.random.uniform(-5, 5)
        direction_x = np.random.uniform(-1, 1)
        direction_y = np.random.uniform(-1, 1)
        initial_speed = np.random.uniform(0.8, 1.5)
        amplitude = np.random.uniform(1.5, 4.0)
        frequency = np.random.uniform(0.1, 0.3)
        trajectory = generate_sine_trajectory(
            start_pos=[start_x, start_y],
            initial_direction_vector=[direction_x, direction_y],
            num_steps=trajectory_length,
            initial_speed=initial_speed,
            amplitude=amplitude,
            frequency=frequency
        )
        dataset.append(normalize_trajectory(trajectory))

    # --- Part 7: Cosine Trajectories ---
    print(f"Generating {num_samples_per_type} samples with cosine trajectory...")
    for i in range(num_samples_per_type):
        start_x = np.random.uniform(-5, 5)
        start_y = np.random.uniform(-5, 5)
        direction_x = np.random.uniform(-1, 1)
        direction_y = np.random.uniform(-1, 1)
        initial_speed = np.random.uniform(0.8, 1.5)
        amplitude = np.random.uniform(1.5, 4.0)
        frequency = np.random.uniform(0.1, 0.3)
        trajectory = generate_cosine_trajectory(
            start_pos=[start_x, start_y],
            initial_direction_vector=[direction_x, direction_y],
            num_steps=trajectory_length,
            initial_speed=initial_speed,
            amplitude=amplitude,
            frequency=frequency
        )
        dataset.append(normalize_trajectory(trajectory))

    return dataset

def plot_samples(trajectory_points, filename):
    """
    Plots a list of numerical samples with a dot at each corresponding timestamp.

    Args:
        samples (list): A list of numerical values representing the samples.
    """
    if not all(len(point) == 2 for point in trajectory_points):
        raise ValueError("Each point in the trajectory_points list must contain exactly two values (x, y).")

    x_values = [point[0] for point in trajectory_points]
    y_values = [point[1] for point in trajectory_points]

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, 'o-')  # 'o-' for dots and lines

    plt.xlabel("X-axis Value")
    plt.ylabel("Y-axis Value")
    plt.title("2D Trajectory Plot (Normalized to [-1, 1])")
    plt.grid(True)
    plt.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    plt.xlim([-1.1, 1.1]) # Set explicit limits to show the -1 to 1 range
    plt.ylim([-1.1, 1.1])
    plt.axis('equal') # Ensure equal scaling for x and y axes for true spatial representation

    # Save the plot
    plt.savefig(filename)
    print(f"Plot saved as {filename} in the current directory.")

def save_trajectory_to_json(trajectory_points, filename="trajectory_data.json"):
    """
    Saves a list of trajectory points to a JSON file.

    Args:
        trajectory_points (list): A list of 2D points (e.g., [[x1, y1], [x2, y2]]).
        filename (str): The name of the JSON file to save to.
    """
    try:
        with open(filename, 'w') as f:
            json.dump(trajectory_points, f, indent=4) # indent for pretty-printing
        print(f"Trajectory data saved to {filename} (JSON format).")
    except Exception as e:
        print(f"Error saving JSON file: {e}")


if __name__ == "__main__":
    # Set a random seed for reproducibility during testing
    np.random.seed(42)

    # This is the line where the change from 20 to 30 steps happened.
    # It has now been reverted to its default value (20 steps).
    pedestrian_trajectories = generate_pedestrian_dataset() 

    print(f"\nGenerated {len(pedestrian_trajectories)} pedestrian trajectories.")
    print(f"Each trajectory has a length of {len(pedestrian_trajectories[0])} time steps.")

    # You can also verify the length of each trajectory and its range
    for i, traj in enumerate(pedestrian_trajectories):
        if len(traj) != 20: # Updated check to reflect 20 steps
            print(f"Error: Trajectory {i} has length {len(traj)} instead of 20.")
        
        # Verify normalization
        x_min, x_max = min(p[0] for p in traj), max(p[0] for p in traj)
        y_min, y_max = min(p[1] for p in traj), max(p[1] for p in traj)
        # Check if ranges are approximately within [-1, 1] due to rounding errors
        if not (x_min >= -1.0001 and x_max <= 1.0001 and y_min >= -1.0001 and y_max <= 1.0001):
            print(f"Warning: Trajectory {i} out of expected [-1, 1] range: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]")

    for i, traj in enumerate(pedestrian_trajectories):
        traj_list = [[int(round((x + 1) * 50)), int(round((y + 1) * 50))] for x, y in traj]
        # plot_samples(traj_list, f"{i}.png")
        # print(traj_list)
        print(normalize_and_tokenize([[float(x), float(y)] for x,y in traj]))
        # print()
