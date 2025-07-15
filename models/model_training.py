import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from sklearn.ensemble import RandomForestRegressor
import os
import logging

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a logistic regression model.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("=== Logistic Regression Results ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
    print("-" * 40)
    return model

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a random forest classifier.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("=== Random Forest Results ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
    print("-" * 40)
    # Feature importances
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print("Random Forest Feature Importances:\n", importances.sort_values(ascending=False))
    return model

#def main():
#    # Load the data
#    #df = pd.read_excel("Output.xlsx")
#    df = pd.read_csv("Output.csv")
#    print(f"Loaded {len(df)} rows from Output.xlsx")
#
#    # Drop columns not useful for modeling (if present)
#    drop_cols = ['reservation_date', 'reservation_time']
#    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
#
#    # Check if there are enough samples
#    if len(df) < 10:
#        print("Not enough data to train/test split. Please provide more samples.")
#        return
#
#    # Define features and target
#    X = df.drop(columns=['will_cancel'])
#    joblib.dump(list(X.columns), "feature_columns.pkl")
#    y = df['will_cancel']
#
#    # Split data
#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, random_state=42, stratify=y
#    )
#
#    # Train and evaluate both models, capturing the returned models
#    logreg_model = train_logistic_regression(X_train, y_train, X_test, y_test)
#    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
#
#    # Save the trained models
#    joblib.dump(logreg_model, "logreg_model.pkl")
#    joblib.dump(rf_model, "rf_model.pkl")
#
#if __name__ == "__main__":
#    main()

def train_waiting_time_regressor(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a random forest regressor for waiting time.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("=== Waiting Time Regression Results ===")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))
    print("-" * 40)
    return model

def main():
    # Load the data
    df = pd.read_csv("Output.csv")
    print(f"Loaded {len(df)} rows from Output.csv")

    # Drop columns not useful for modeling
    drop_cols = ['subscriber_ID']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Identify categorical columns
    categorical_cols = ['booking_channel', 'reservation_type', 'weather_condition', 'order_complexity', 'preferred_channel']

    # Only encode columns that exist in the dataframe
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    # One-hot encode
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Check if there are enough samples
    if len(df) < 10:
        print("Not enough data to train/test split. Please provide more samples.")
        return

    # Define features and target
    X = df.drop(columns=['will_cancel'])
    joblib.dump(list(X.columns), "feature_columns.pkl")
    y = df['will_cancel']

    # --- Classification: Predict will_cancel ---
    X_cls = df.drop(columns=['will_cancel'])
    y_cls = df['will_cancel']
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    logreg_model = train_logistic_regression(X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    rf_model = train_random_forest(X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    
    # --- Regression: Predict waiting_time_minutes ---
    X_reg = df.drop(columns=['waiting_time_minutes', 'will_cancel'])
    y_reg = df['waiting_time_minutes']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_model = train_waiting_time_regressor(X_train_reg, y_train_reg, X_test_reg, y_test_reg)
    
    # Save the trained models
    if not os.path.exists("trained_models"):
        os.makedirs("trained_models", exist_ok=True)
    joblib.dump(logreg_model, "trained_models/logreg_model.pkl")
    joblib.dump(rf_model, "trained_models/rf_model.pkl")
    joblib.dump(reg_model, "trained_models/waiting_time_regressor.pkl")

    # Identify categorical columns
    categorical_cols = ['booking_channel', 'reservation_type', 'weather_condition', 'order_complexity', 'preferred_channel']

    # Only encode columns that exist in the dataframe
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    # One-hot encode
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

if __name__ == "__main__":
    main()