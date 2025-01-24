# simulator.py
import numpy as np
import pandas as pd

def generate_simulated_data(num_drivers=5, num_orders=5, seed=42):
    np.random.seed(seed)

    # Simulate drivers
    drivers_data = []
    for d_id in range(num_drivers):
        driver_id = f"D{d_id+1}"
        location_x = np.random.uniform(0, 10)
        location_y = np.random.uniform(0, 10)
        rating = np.random.uniform(3, 5)  # rating between 3 and 5
        cost_factor = np.random.uniform(1, 2)  # cost multiplier
        incentive_progress = np.random.randint(0, 10)  # e.g., number of deliveries done today
        drivers_data.append([driver_id, location_x, location_y, rating, cost_factor, incentive_progress])

    drivers_df = pd.DataFrame(drivers_data, 
                              columns=["driver_id", "loc_x", "loc_y", 
                                       "rating", "cost_factor", "incentive_progress"])

    # Simulate orders
    orders_data = []
    for o_id in range(num_orders):
        order_id = f"O{o_id+1}"
        loc_x = np.random.uniform(0, 10)
        loc_y = np.random.uniform(0, 10)
        item_value = np.random.randint(100, 1000)  # random order value
        priority = np.random.choice(["normal", "vip", "high_value"])
        required_sla = np.random.randint(10, 30)  # e.g., 10–30 minutes
        orders_data.append([order_id, loc_x, loc_y, item_value, priority, required_sla])

    orders_df = pd.DataFrame(orders_data, 
                             columns=["order_id", "loc_x", "loc_y", 
                                      "item_value", "priority", "required_sla"])

    return drivers_df, orders_df
