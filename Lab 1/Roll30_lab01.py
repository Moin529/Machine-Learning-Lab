import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


SYNTHETIC_FILE = "lab01_data.csv"
REAL_DATA_FILE = "data_01.csv"

LEARNING_RATE = 0.005
ITERATIONS = 10000
RANDOM_SEED = 42


def generate_synthetic_data():
    rng = np.random.default_rng(RANDOM_SEED)

    x = np.arange(1, 101, dtype=float)
    noise = rng.normal(0, 1, size=len(x))
    y = 3 + 5 * x + noise

    return x, y


def save_data(filename, x, y):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])
        writer.writerows(zip(x, y))


def load_data(filename):
    if not Path(filename).exists():
        raise FileNotFoundError(f"Dataset not found: {filename}")

    x = []
    y = []

    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) < 2:
                continue

            try:
                x.append(float(row[0]))
                y.append(float(row[1]))
            except ValueError:
                continue

    if not x:
        raise ValueError(f"No valid data found in {filename}")

    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def process_data(x, y, feature_scaling=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("x and y must have the same number of samples.")

    mean = 0.0
    std = 1.0

    if feature_scaling:
        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            raise ValueError("Feature standard deviation cannot be zero.")

        x = (x - mean) / std

    X = np.column_stack((np.ones(len(x)), x))

    return X, y, mean, std


def compute_cost(X, y, theta):
    m = len(y)
    errors = X @ theta - y

    return np.sum(errors ** 2) / (2 * m)


def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        errors = X @ theta - y
        gradient = (X.T @ errors) / m
        theta -= learning_rate * gradient
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history


def train(X, y, learning_rate, iterations):
    theta = np.zeros(X.shape[1], dtype=float)

    return gradient_descent(
        X,
        y,
        theta,
        learning_rate,
        iterations
    )


def evaluate(X, y, theta):
    predictions = X @ theta
    errors = predictions - y

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)

    return predictions, mse, rmse


def convert_parameters(theta, mean, std):
    theta0 = theta[0] - (theta[1] * mean / std)
    theta1 = theta[1] / std

    return np.array([theta0, theta1])


def plot_data(x, y, title, filename):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="Data points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_cost(cost_history, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(1, len(cost_history) + 1),
        cost_history
    )
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost J(theta)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_regression(x, y, theta, title, filename):
    order = np.argsort(x)
    x_sorted = x[order]

    X_sorted = np.column_stack(
        (np.ones(len(x_sorted)), x_sorted)
    )

    y_pred = X_sorted @ theta

    plt.figure(figsize=(8, 5))
    plt.scatter(
        x,
        y,
        label="Data points"
    )
    plt.plot(
        x_sorted,
        y_pred,
        color="red",
        linewidth=2,
        label="Regression line"
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def run_experiment(filename, dataset_name, feature_scaling=True):
    x, y = load_data(filename)

    X, y, mean, std = process_data(
        x,
        y,
        feature_scaling
    )

    theta_scaled, cost_history = train(
        X,
        y,
        LEARNING_RATE,
        ITERATIONS
    )

    predictions, mse, rmse = evaluate(
        X,
        y,
        theta_scaled
    )

    theta = convert_parameters(
        theta_scaled,
        mean,
        std
    )

    print()
    print("=" * 60)
    print(dataset_name)
    print("=" * 60)
    print(f"Learning rate      : {LEARNING_RATE}")
    print(f"Iterations         : {ITERATIONS}")
    print(f"Feature scaling    : {feature_scaling}")
    print()
    print(f"theta0 = {theta[0]:.6f}")
    print(f"theta1 = {theta[1]:.6f}")
    print()
    print(f"MSE        = {mse:.6f}")
    print(f"RMSE       = {rmse:.6f}")
    print(f"Final cost = {cost_history[-1]:.6f}")

    prefix = dataset_name.lower().replace(" ", "_")

    plot_data(
        x,
        y,
        f"{dataset_name} - Data Points",
        f"{prefix}_data.png"
    )

    plot_cost(
        cost_history,
        f"{dataset_name} - Training Error",
        f"{prefix}_cost.png"
    )

    plot_regression(
        x,
        y,
        theta,
        f"{dataset_name} - Learned Regression Line",
        f"{prefix}_regression.png"
    )


def main():
    x, y = generate_synthetic_data()

    save_data(
        SYNTHETIC_FILE,
        x,
        y
    )

    print(f"Synthetic data saved to {SYNTHETIC_FILE}")

    run_experiment(
        SYNTHETIC_FILE,
        "Synthetic Dataset",
        feature_scaling=True
    )

    if Path(REAL_DATA_FILE).exists():
        run_experiment(
            REAL_DATA_FILE,
            "Real Dataset",
            feature_scaling=True
        )
    else:
        print(f"{REAL_DATA_FILE} not found.")


if __name__ == "__main__":
    main()