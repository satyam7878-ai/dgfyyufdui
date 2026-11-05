from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots without opening a window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def evaluate_regression(model, X_test, y_test):
    """Scores and plot for the Linear Regression model."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print("\n--- Linear Regression (predicts Vehicles) ---")
    print(f"MAE : {mae:.2f} vehicles")
    print(f"RMSE: {rmse:.2f} vehicles")
    print(f"R2  : {r2:.3f}")

    # Predicted vs actual for the first 7 days of one junction
    first_junction = X_test["Junction"].iloc[0]
    mask = (X_test["Junction"] == first_junction).to_numpy()
    actual = y_test.to_numpy()[mask][:168]
    predicted = pred[mask][:168]

    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title(f"Predicted vs actual vehicles (Junction {first_junction}, first 7 test days)")
    plt.xlabel("Hours")
    plt.ylabel("Vehicles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "linear_predicted_vs_actual.png")
    plt.close()

    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_classifier(clf, X_test, congested_test):
    """Scores and confusion matrix for the Logistic Regression alert model."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    pred = clf.predict(X_test)

    acc = accuracy_score(congested_test, pred)
    prec = precision_score(congested_test, pred, zero_division=0)
    rec = recall_score(congested_test, pred, zero_division=0)

    print("\n--- Logistic Regression (congestion alert) ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")

    ConfusionMatrixDisplay.from_predictions(
        congested_test, pred, display_labels=["Not congested", "Congested"])
    plt.title("Congestion alert: confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "logistic_confusion_matrix.png")
    plt.close()

    n_alerts = int(pred.sum())
    print(f"Alerts raised on the test data: {n_alerts} of {len(pred)} hours")
    return {"accuracy": acc, "precision": prec, "recall": rec}
