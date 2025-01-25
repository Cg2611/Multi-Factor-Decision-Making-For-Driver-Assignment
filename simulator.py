# simulator.py
import numpy as np
import pandas as pd

def generate_simulated_data(num_drivers=5, num_orders=5, seed=42):
    """
    Generates:
      - drivers_df: 
          driver_id, loc_x, loc_y, rating, cost_factor, incentive_progress, 
          shift_start, shift_end, next_break_in_minutes
      - orders_df: 
          order_id, loc_x, loc_y, item_value, priority, required_sla, surge_zone
      - traffic_level: 
          a single float representing a city-wide traffic multiplier
    """
    np.random.seed(seed)

    # Simulate city-wide traffic level [1..3], 
    # e.g., 1 => low traffic, 3 => severe traffic
    traffic_level = np.random.uniform(1.0, 3.0)

    # For demonstration, center around Bangalore lat/lon ~ (12.9716, 77.5946)
    # We'll generate random offsets in a ~ 0.1 deg box to keep markers in the city
    # 1 deg of lat ~ 111 km, 0.1 deg ~ ~11 km
    # We'll store them in 'loc_lat' and 'loc_lon' for clarity.
    def random_bangalore_location():
        lat = 12.9716 + np.random.uniform(-0.05, 0.05) 
        lon = 77.5946 + np.random.uniform(-0.05, 0.05)
        return lat, lon

    # Drivers
    drivers_data = []
    for d_id in range(num_drivers):
        driver_id = f"D{d_id+1}"
        lat, lon = random_bangalore_location()
        rating = np.random.uniform(3.0, 5.0)            # rating between 3 and 5
        cost_factor = np.random.uniform(1.0, 2.0)       # cost multiplier
        incentive_progress = np.random.randint(0, 10)   # deliveries completed toward an incentive

        # Shift times: random 9-hr shift between 8am-8pm
        shift_start = np.random.randint(8, 14)          # e.g., 8AM to 2PM
        shift_end = shift_start + 9                     # 9-hour shift
        # Next break in minutes
        next_break_in_minutes = np.random.randint(0, 60)  # could be in 0..60 min

        drivers_data.append([
            driver_id, lat, lon, rating, cost_factor, 
            incentive_progress, shift_start, shift_end, 
            next_break_in_minutes
        ])

    drivers_df = pd.DataFrame(drivers_data, columns=[
        "driver_id", "loc_lat", "loc_lon", "rating", "cost_factor", 
        "incentive_progress", "shift_start", "shift_end", 
        "next_break_in_minutes"
    ])

    # Orders
    orders_data = []
    for o_id in range(num_orders):
        order_id = f"O{o_id+1}"
        lat, lon = random_bangalore_location()
        item_value = np.random.randint(100, 1000)
        priority = np.random.choice(["normal", "vip", "high_value"])
        required_sla = np.random.randint(10, 30)  # 10-30 minutes

        # Some fraction of orders are in "surge zones"
        surge_zone = np.random.choice([True, False], p=[0.3, 0.7])

        orders_data.append([order_id, lat, lon, item_value, priority, required_sla, surge_zone])

    orders_df = pd.DataFrame(orders_data, columns=[
        "order_id", "loc_lat", "loc_lon", "item_value", "priority", 
        "required_sla", "surge_zone"
    ])

    return drivers_df, orders_df, traffic_level
