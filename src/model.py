from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _preprocess():
    """One-hot encode Junction, hour and day_of_week; scale month."""
    return ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"),
         ["Junction", "hour", "day_of_week"]),
        ("month", StandardScaler(), ["month"]),
    ])


def train_model(X_train, y_train):
    """Linear Regression: predicts the number of vehicles."""
    pipe = Pipeline([
        ("preprocess", _preprocess()),
        ("model", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_classifier(X_train, congested_train):
    """Logistic Regression: predicts congested (1) or not (0)."""
    pipe = Pipeline([
        ("preprocess", _preprocess()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, congested_train)
    return pipe
