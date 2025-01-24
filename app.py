# app.py
import streamlit as st
import pandas as pd
import numpy as np

from simulator import generate_simulated_data
from assignment import compute_pairwise_cost, assign_drivers_to_orders

def main():
    st.title("Pathbreaking Quick-Commerce Driver Assignment")

    st.sidebar.header("Simulation Controls")
    num_drivers = st.sidebar.slider("Number of Drivers", 1, 20, 5)
    num_orders = st.sidebar.slider("Number of Orders", 1, 20, 5)
    seed_value = st.sidebar.number_input("Random Seed", 0, 9999, 42)

    # Generate data
    drivers_df, orders_df = generate_simulated_data(num_drivers, num_orders, seed=seed_value)

    st.sidebar.header("Weight Controls")
    w_time = st.sidebar.slider("Weight: Time/Distance", 0.0, 5.0, 1.0, 0.1)
    w_cost = st.sidebar.slider("Weight: Driver Cost Factor", 0.0, 5.0, 1.0, 0.1)
    w_rating = st.sidebar.slider("Weight: Rating Penalty (VIP/High Value)", 0.0, 5.0, 1.0, 0.1)
    w_incentive = st.sidebar.slider("Weight: Incentive Alignment (negative => discount)", -2.0, 2.0, -0.5, 0.1)

    # Display the data in the UI
    st.subheader("Drivers")
    st.dataframe(drivers_df)
    st.subheader("Orders")
    st.dataframe(orders_df)

    # Compute pairwise costs
    costs = compute_pairwise_cost(drivers_df, orders_df, 
                                  w_time=w_time,
                                  w_cost=w_cost, 
                                  w_rating=w_rating, 
                                  w_incentive=w_incentive)
    
    # Solve assignment
    assignment_results = assign_drivers_to_orders(drivers_df, orders_df, costs)
    st.subheader("Assignment Results")

    if assignment_results:
        assigned_df = pd.DataFrame(assignment_results, columns=["driver_id", "order_id", "cost"])
        st.dataframe(assigned_df.style.format({"cost": "{:.2f}"}))
    else:
        st.write("No assignment found or data is insufficient.")

    # Show a quick map (using lat/long placeholders, but we can interpret x,y in a 'map' sense)
    # Note: Streamlit's map component expects lat/long in columns named "lat" and "lon".
    st.subheader("Driver & Order Locations (Approx.)")
    driver_locations = drivers_df.rename(columns={"loc_x": "lat", "loc_y": "lon"})
    driver_locations["type"] = "driver"
    order_locations = orders_df.rename(columns={"loc_x": "lat", "loc_y": "lon"})
    order_locations["type"] = "order"

    combined_locations = pd.concat([driver_locations[["lat","lon","type"]],
                                    order_locations[["lat","lon","type"]]])
    st.map(combined_locations)

if __name__ == "__main__":
    main()
