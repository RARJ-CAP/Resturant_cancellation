import pandas as pd
import joblib


def predict_cancellation_probability(model, new_row, feature_columns):
    """
    Predict the probability of cancellation for a new reservation.
    - model: trained sklearn model
    - new_row: pd.DataFrame with one row, matching the training features
    - feature_columns: list of columns used for training (order matters)
    Returns: probability of cancellation (float)
    """
    # Ensure the new_row has the same columns and order as training data
    for col in feature_columns:
        if col not in new_row.columns:
            new_row[col] = 0
    new_row = new_row[feature_columns]
    prob = model.predict_proba(new_row)[:, 1][0]
    print(f"Predicted probability of cancellation: {prob:.2%}")
    return prob

# Load trained models
logreg_model = joblib.load("trained_models/logreg_model.pkl")
rf_model = joblib.load("trained_models/rf_model.pkl")

# Load the feature columns used during training
feature_columns = joblib.load("feature_columns.pkl")

# Prepare a new row (example)
example_row = pd.DataFrame([{
    'lead_time_days': 14,
    'party_size': 5,
    'no_show_rate': 0.15,
    'visit_frequency': 12,
    'table_assigned': True,
    'responded_to_confirmation': False,
    'occupancy_rate': 0.65,
    'reservation_hour': 19,
    'reservation_dayofweek': 4,
    'booking_channel_app': 1,
    'booking_channel_kiosk': 0,
    'booking_channel_other': 0,
    'booking_channel_phone': 0,
    'booking_channel_web': 0
}])

print("\n--- Predicting cancellation probability for a new reservation (Logistic Regression) ---")
predict_cancellation_probability(logreg_model, example_row, feature_columns)

print("\n--- Predicting cancellation probability for a new reservation (Random Forest) ---")
predict_cancellation_probability(rf_model, example_row, feature_columns)
