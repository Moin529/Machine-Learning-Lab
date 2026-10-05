import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile, os


def load_ccpp(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        xlsx_name = next(n for n in z.namelist() if n.endswith(".xlsx"))
        with z.open(xlsx_name) as f:
            df = pd.read_excel(f)
    return df


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=["x", "y"])
    return df


def train_val_split(X: np.ndarray, y: np.ndarray,
                    val_ratio: float = 0.2,
                    seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = int(len(y) * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def normalize(X_train: np.ndarray, X_val: np.ndarray):
    mu  = X_train.mean(axis=0)
    sig = X_train.std(axis=0)
    sig[sig == 0] = 1
    return (X_train - mu) / sig, (X_val - mu) / sig, mu, sig


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1)), X])


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([x ** d for d in range(1, degree + 1)])


def mse(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    m = len(y)
    residuals = X @ theta - y
    return float(residuals @ residuals) / (2 * m)


def gradient_descent(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray,   y_val: np.ndarray,
                      lr: float = 0.01, n_iter: int = 1000):
    m, n = X_train.shape
    theta = np.zeros(n)

    train_costs: list[float] = []
    val_costs:   list[float] = []

    for _ in range(n_iter):
        gradient  = (X_train.T @ (X_train @ theta - y_train)) / m
        theta    -= lr * gradient

        train_costs.append(mse(X_train, y_train, theta))
        val_costs.append(mse(X_val,   y_val,   theta))

    return theta, train_costs, val_costs


def plot_features_vs_target(X: np.ndarray, y: np.ndarray,
                             feature_names: list[str],
                             target_name: str = "PE (MW)"):
    n_features = X.shape[1]
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4))
    for i, ax in enumerate(axes):
        ax.scatter(X[:, i], y, s=5, alpha=0.4, color="steelblue")
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(target_name)
        ax.set_title(f"{feature_names[i]} vs {target_name}")
    plt.suptitle("Feature vs Target Plots", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("feature_vs_target.png", dpi=150)
    plt.show()


def plot_error_curves(train_costs: list[float], val_costs: list[float],
                      title: str = "Error Curves"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_costs, label="Training Error",   color="steelblue")
    ax.plot(val_costs,   label="Validation Error", color="coral")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_fitted_curves(x_raw: np.ndarray, y: np.ndarray,
                       thetas: dict[int, np.ndarray],
                       mu_dict: dict[int, np.ndarray],
                       sig_dict: dict[int, np.ndarray]):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x_raw, y, s=10, alpha=0.4, color="gray", label="Data", zorder=1)

    x_line = np.linspace(x_raw.min(), x_raw.max(), 400)
    colours = {1: "steelblue", 2: "coral", 3: "seagreen"}

    for d, theta in thetas.items():
        Phi    = polynomial_features(x_line, d)
        Phi_n  = (Phi - mu_dict[d]) / sig_dict[d]
        Phi_n  = add_bias(Phi_n)
        y_pred = Phi_n @ theta
        ax.plot(x_line, y_pred, linewidth=2,
                color=colours[d], label=f"d={d}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Fitted Polynomial Curves (d = 1, 2, 3)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("poly_fitted_curves.png", dpi=150)
    plt.show()


def plot_val_error_bar(val_errors: dict[int, float]):
    fig, ax = plt.subplots(figsize=(5, 4))
    degrees = list(val_errors.keys())
    errors  = [val_errors[d] for d in degrees]
    colours = ["steelblue", "coral", "seagreen"]
    bars = ax.bar([str(d) for d in degrees], errors, color=colours[:len(degrees)])
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_xlabel("Polynomial Degree (d)")
    ax.set_ylabel("Best Validation MSE")
    ax.set_title("Validation Error by Degree")
    plt.tight_layout()
    plt.savefig("poly_val_bar.png", dpi=150)
    plt.show()


def report_linear(label: str, theta: np.ndarray,
                  train_costs: list[float], val_costs: list[float],
                  feature_names: list[str]):
    best_train_iter = int(np.argmin(train_costs)) + 1
    best_val_iter   = int(np.argmin(val_costs))   + 1
    best_train      = train_costs[best_train_iter - 1]
    best_val        = val_costs[best_val_iter - 1]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Best Training   MSE : {best_train:.4f}  (iteration {best_train_iter})")
    print(f"  Best Validation MSE : {best_val:.4f}  (iteration {best_val_iter})")
    print(f"  Learnt parameters (theta):")
    print(f"    bias  = {theta[0]:.6f}")
    for i, name in enumerate(feature_names):
        print(f"    {name:4s}  = {theta[i+1]:.6f}")


def report_polynomial(d: int, theta: np.ndarray,
                       train_costs: list[float], val_costs: list[float]):
    best_train_iter = int(np.argmin(train_costs)) + 1
    best_val_iter   = int(np.argmin(val_costs))   + 1
    best_train      = train_costs[best_train_iter - 1]
    best_val        = val_costs[best_val_iter - 1]
    print(f"\n  -- d = {d} --")
    print(f"     Best Training MSE   : {best_train:.4f}  (iteration {best_train_iter})")
    print(f"     Best Validation MSE : {best_val:.4f}  (iteration {best_val_iter})")
    print(f"     Learnt parameters (theta):")
    print(f"       bias  = {theta[0]:.6f}")
    for p in range(1, d + 1):
        print(f"       x^{p}  = {theta[p]:.6f}")


def part_a(zip_path: str,
           lr: float = 0.1,
           n_iter: int = 1000,
           lr_raw: float = 1e-9):

    print("\n" + "="*60)
    print("  PART A — Linear Regression (Multiple Variables)")
    print("="*60)

    df            = load_ccpp(zip_path)
    feature_names = ["AT", "V", "AP", "RH"]
    X_raw         = df[feature_names].values
    y             = df["PE"].values

    plot_features_vs_target(X_raw, y, feature_names, target_name="PE (MW)")

    X_tr, X_val, y_tr, y_val = train_val_split(X_raw, y)

    X_tr_b  = add_bias(X_tr)
    X_val_b = add_bias(X_val)

    theta_raw, tc_raw, vc_raw = gradient_descent(
        X_tr_b, y_tr, X_val_b, y_val, lr=lr_raw, n_iter=n_iter
    )
    report_linear("Without Normalisation", theta_raw, tc_raw, vc_raw, feature_names)

    fig = plot_error_curves(tc_raw, vc_raw,
                            title="Error Curves — Without Normalisation")
    fig.savefig("linreg_error_raw.png", dpi=150)
    plt.show()

    X_tr_n, X_val_n, mu, sig = normalize(X_tr, X_val)
    X_tr_nb  = add_bias(X_tr_n)
    X_val_nb = add_bias(X_val_n)

    theta_norm, tc_norm, vc_norm = gradient_descent(
        X_tr_nb, y_tr, X_val_nb, y_val, lr=lr, n_iter=n_iter
    )
    report_linear("With Normalisation", theta_norm, tc_norm, vc_norm, feature_names)

    fig = plot_error_curves(tc_norm, vc_norm,
                            title="Error Curves — With Normalisation")
    fig.savefig("linreg_error_norm.png", dpi=150)
    plt.show()

    print("\n  ── Comparison: Raw vs Normalised ──")
    print(f"  {'Metric':<30} {'Raw':>12} {'Normalised':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Best Training MSE':<30} {min(tc_raw):>12.4f} {min(tc_norm):>12.4f}")
    print(f"  {'Best Validation MSE':<30} {min(vc_raw):>12.4f} {min(vc_norm):>12.4f}")


def part_b(csv_path: str,
           degrees: list[int] = [1, 2, 3],
           lr: float = 0.01,
           n_iter: int = 2000):

    print("\n" + "="*60)
    print("  PART B — Polynomial Regression")
    print("="*60)

    df    = load_csv(csv_path)
    x_raw = df["x"].values
    y     = df["y"].values

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x_raw, y, s=12, alpha=0.6, color="steelblue")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Data: x vs y")
    plt.tight_layout()
    plt.savefig("poly_data.png", dpi=150)
    plt.show()

    x_2d                      = x_raw.reshape(-1, 1)
    x_tr, x_val, y_tr, y_val = train_val_split(x_2d, y)
    x_tr                      = x_tr.ravel()
    x_val                     = x_val.ravel()

    results: dict[int, dict] = {}

    print()
    print("  Polynomial Regression Results:")
    for d in degrees:
        Phi_tr  = polynomial_features(x_tr,  d)
        Phi_val = polynomial_features(x_val, d)

        Phi_tr_n, Phi_val_n, mu_d, sig_d = normalize(Phi_tr, Phi_val)

        Phi_tr_b  = add_bias(Phi_tr_n)
        Phi_val_b = add_bias(Phi_val_n)

        theta, tc, vc = gradient_descent(
            Phi_tr_b, y_tr, Phi_val_b, y_val, lr=lr, n_iter=n_iter
        )

        results[d] = dict(theta=theta, tc=tc, vc=vc,
                          best_val=min(vc), mu=mu_d, sig=sig_d)

        report_polynomial(d, theta, tc, vc)

    best_d = min(results, key=lambda d: results[d]["best_val"])
    print(f"\n  ★  Best degree (lowest validation MSE): d = {best_d}")

    plot_fitted_curves(
        x_raw, y,
        thetas   = {d: results[d]["theta"] for d in degrees},
        mu_dict  = {d: results[d]["mu"]    for d in degrees},
        sig_dict = {d: results[d]["sig"]   for d in degrees},
    )

    plot_val_error_bar({d: results[d]["best_val"] for d in degrees})

    fig, axes = plt.subplots(1, len(degrees), figsize=(6 * len(degrees), 4))
    colours   = {1: "steelblue", 2: "coral", 3: "seagreen"}
    for ax, d in zip(axes, degrees):
        tc = results[d]["tc"]
        vc = results[d]["vc"]
        ax.plot(tc, label="Train",      color=colours[d])
        ax.plot(vc, label="Validation", color=colours[d], linestyle="--")
        ax.set_title(f"Error Curves (d={d})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MSE")
        ax.legend()
    plt.suptitle("Polynomial Regression — Error Curves per Degree",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("poly_error_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    CCPP_ZIP = "CCPP.zip"
    DATA_CSV = "data_02b.csv"

    part_a(CCPP_ZIP)
    part_b(DATA_CSV)