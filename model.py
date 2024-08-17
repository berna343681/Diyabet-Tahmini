from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_model(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=46)
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
    print(f"Recall: {round(recall_score(y_pred, y_test), 3)}")
    print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
    print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
    print(f"AUC: {round(roc_auc_score(y_pred, y_test), 2)}")
