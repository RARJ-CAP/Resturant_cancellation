import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import OneHotEncoder

# Setup logging to track ETL steps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure reproducibility
fake = Faker()
np.random.seed(42)
random.seed(42)

def generate_synthetic_data(n_samples, n_customers=500):
    """
    Generates synthetic reservation data with customer and operational context.
    Ensures that subscriber_IDs repeat, simulating real customer history.
    """
    # Generate a pool of unique subscriber IDs
    subscriber_pool = [str(Faker().uuid4()) for _ in range(n_customers)]
    # Optionally, initialize customer history
    customer_history = {
        sid: {
            "no_show_rate": round(random.uniform(0, 1), 2),
            "visit_frequency": random.randint(0, 20)
        }
        for sid in subscriber_pool
    }

    data = []
    for _ in range(n_samples):
        subscriber_ID = random.choice(subscriber_pool)
        # Use and update customer history for more realism
        no_show_rate = customer_history[subscriber_ID]["no_show_rate"]
        visit_frequency = customer_history[subscriber_ID]["visit_frequency"]
        # Optionally, update visit frequency
        customer_history[subscriber_ID]["visit_frequency"] += 1

        reservation_datetime = fake.date_time_between(start_date='-30d', end_date='+30d')
        booking_channel = random.choice(['web', 'phone', 'kiosk', 'app', 'other'])
        reservation_type = random.choices(
            ['standard', 'VIP', 'event', 'walk-in', 'group'],
            weights=[0.7, 0.05, 0.1, 0.1, 0.05]
        )[0]
        vip_status = reservation_type == 'VIP' or random.random() < 0.05
        lead_time_days = random.randint(0, 30)
        party_size = random.randint(1, 10)
        table_assigned = random.choice([True, False])
        responded_to_confirmation = random.choice([True, False])
        occupancy_rate = round(random.uniform(0.3, 1.0), 2)
        special_request = random.choice([True, False])
        weather_condition = random.choice(['sunny', 'rainy', 'cloudy', 'stormy'])
        event_nearby = random.choice([True, False])
        staff_on_shift = random.randint(5, 20)
        order_complexity = random.choices(
            ['low', 'medium', 'high'],
            weights=[0.6, 0.3, 0.1]
        )[0]
        base_wait = 5
        complexity_factor = {'low': 0, 'medium': 5, 'high': 10}[order_complexity]
        wait = base_wait \
            + party_size * random.uniform(1, 3) \
            + (occupancy_rate * 20) \
            + (5 if reservation_type == 'event' else 0) \
            - (5 if vip_status else 0) \
            + (5 if special_request else 0) \
            + (5 if event_nearby else 0) \
            - (staff_on_shift * 0.5) \
            + complexity_factor \
            + random.gauss(0, 3)
        
        # Calculate waiting_time_minutes 
        waiting_time_minutes = max(0, round(wait, 1))
        # Adjust cancellation probability if waiting time is high
        cancel_prob = 0.1  # base probability
        if no_show_rate > 0.5 and not responded_to_confirmation:
            cancel_prob = 0.7  # high risk customer
        if waiting_time_minutes > 40:
            cancel_prob = max(cancel_prob, 0.5)  # at least 50% chance if wait is very long
            # Optionally, make it even higher for extreme waits:
            if waiting_time_minutes > 60:
                cancel_prob = max(cancel_prob, 0.8)
        will_cancel = int((no_show_rate > 0.5 and not responded_to_confirmation) or random.random() < 0.1)

        data.append([
            subscriber_ID, reservation_datetime, booking_channel, reservation_type, vip_status,
            lead_time_days, party_size, no_show_rate, visit_frequency, table_assigned,
            responded_to_confirmation, occupancy_rate, special_request, weather_condition,
            event_nearby, staff_on_shift, will_cancel, waiting_time_minutes, order_complexity
        ])

    columns = [
        'subscriber_ID', 'reservation_datetime', 'booking_channel', 'reservation_type', 'vip_status',
        'lead_time_days', 'party_size', 'no_show_rate', 'visit_frequency', 'table_assigned',
        'responded_to_confirmation', 'occupancy_rate', 'special_request', 'weather_condition',
        'event_nearby', 'staff_on_shift', 'will_cancel', 'waiting_time_minutes', 'order_complexity'
    ]
    return pd.DataFrame(data, columns=columns)

def transform_data(df):
    """
    Transforms raw data into a machine learning-ready format.
    Includes feature engineering and encoding of categorical variables.
    """
    logging.info("Transforming data...")

    # Extract hour and day of week from datetime
    df['reservation_hour'] = df['reservation_datetime'].dt.hour
    df['reservation_dayofweek'] = df['reservation_datetime'].dt.dayofweek

    # Drop the original datetime column
    df.drop(columns=['reservation_datetime'], inplace=True)

    # One-hot encode categorical variables
    categorical_cols = ['booking_channel', 'reservation_type', 'weather_condition']
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Combine encoded columns with the rest of the dataset
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    return df

def main():
    n_samples = 10000
    df = generate_synthetic_data(n_samples)
    df = transform_data(df)

    # Preview the cleaned dataset
    print(df.head())
    df.to_excel("Output.xlsx", index=False)
    df.to_csv("Output.csv", index=False)
    print("\nData types:\n", df.dtypes)
    print("\nClass balance (will_cancel):\n", df['will_cancel'].value_counts(normalize=True))

if __name__ == "__main__":
    main()

