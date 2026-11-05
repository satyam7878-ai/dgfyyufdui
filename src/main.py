from data import get_train_test, congestion_labels
from model import train_model, train_classifier
from evaluate import evaluate_regression, evaluate_classifier
import eda


def main():
    X_train, X_test, y_train, y_test = get_train_test()
    c_train, c_test = congestion_labels(X_train, y_train, X_test, y_test)
    print(f"Training rows: {len(X_train)}, test rows: {len(X_test)}")

    eda.run()                                   # charts -> outputs/

    reg = train_model(X_train, y_train)         # Linear Regression
    clf = train_classifier(X_train, c_train)    # Logistic Regression

    evaluate_regression(reg, X_test, y_test)
    evaluate_classifier(clf, X_test, c_test)


if __name__ == "__main__":
    main()
