# app.py
import streamlit as st
import pandas as pd
import numpy as np

from simulator import generate_simulated_data
from assignment import compute_pairwise_cost, assign_drivers_to_orders

import pydeck as pdk

def main():
    st.title("Driver Assignment")

    st.sidebar.header("Simulation Controls")
    num_drivers = st.sidebar.slider("Number of Drivers", 1, 20, 5)
    num_orders = st.sidebar.slider("Number of Orders", 1, 50, 5)
    seed_value = st.sidebar.number_input("Random Seed", 0, 9999, 42)

    # Generate data
    drivers_df, orders_df, traffic_level = generate_simulated_data(
        num_drivers=num_drivers,
        num_orders=num_orders,
        seed=seed_value
    )

    # Additional UI
    st.sidebar.header("Time & Capacity")
    current_time = st.sidebar.slider("Current Time (Hour)", 0, 24, 10)
    driver_capacity = st.sidebar.slider("Driver Capacity (multi-stop routes)", 1, 5, 2)

    st.sidebar.header("Weight Controls")
    w_time = st.sidebar.slider("Weight: Time/Distance", 0.0, 5.0, 1.0, 0.1)
    w_cost = st.sidebar.slider("Weight: Driver Cost Factor", 0.0, 5.0, 1.0, 0.1)
    w_rating = st.sidebar.slider("Weight: Rating Penalty (VIP/High Value)", 0.0, 5.0, 1.0, 0.1)
    w_incentive = st.sidebar.slider("Weight: Incentive Alignment (negative => discount)", -2.0, 2.0, -0.5, 0.1)
    w_fairness = st.sidebar.slider("Weight: Fairness Penalty", 0.0, 3.0, 0.5, 0.1)
    w_surge = st.sidebar.slider("Weight: Surge Adjustment (negative => discount)", -5.0, 0.0, -2.0, 0.1)

    st.write("## Simulation Data")
    st.write(f"**Traffic Level** (city-wide): {traffic_level:.2f}")
    st.subheader("Drivers")
    st.dataframe(drivers_df)
    st.subheader("Orders")
    st.dataframe(orders_df)

    # Compute pairwise costs
    costs = compute_pairwise_cost(
        drivers_df, 
        orders_df,
        traffic_level=traffic_level,
        w_time=w_time,
        w_cost=w_cost,
        w_rating=w_rating,
        w_incentive=w_incentive,
        w_fairness=w_fairness,
        w_surge=w_surge,
        current_time=current_time
    )

    # Solve assignment
    assignment_results = assign_drivers_to_orders(
        drivers_df, 
        orders_df, 
        costs, 
        driver_capacity=driver_capacity
    )

    st.subheader("Assignment Results")
    if assignment_results:
        assigned_df = pd.DataFrame(assignment_results, columns=["driver_id", "order_id", "cost"])
        st.dataframe(assigned_df.style.format({"cost": "{:.2f}"}))
    else:
        st.write("No assignment found or data is insufficient (possibly all costs are too high).")

    # Prepare map data: color code driver vs. order
    # Let's create two separate layers in PyDeck:
    #   - Red for drivers
    #   - Green for orders

    # The lat/lon columns MUST be "latitude" and "longitude" for PyDeck,
    # or we specify custom get_position below.

    drivers_map_df = drivers_df[["driver_id", "loc_lat", "loc_lon"]].copy()
    drivers_map_df["type"] = "driver"
    drivers_map_df.rename(columns={"loc_lat": "latitude", "loc_lon": "longitude"}, inplace=True)

    orders_map_df = orders_df[["order_id", "loc_lat", "loc_lon"]].copy()
    orders_map_df["type"] = "order"
    orders_map_df.rename(columns={"loc_lat": "latitude", "loc_lon": "longitude"}, inplace=True)

    # PyDeck layers:
    driver_layer = pdk.Layer(
        "ScatterplotLayer",
        data=drivers_map_df,
        get_position="[longitude, latitude]",
        get_radius=150,
        get_fill_color=[255, 0, 0],  # Red
        pickable=True,
        tooltip="Driver"
    )

    order_layer = pdk.Layer(
        "ScatterplotLayer",
        data=orders_map_df,
        get_position="[longitude, latitude]",
        get_radius=150,
        get_fill_color=[0, 255, 0],  # Green
        pickable=True,
        tooltip="Order"
    )

    # View: Center at Bangalore with a good zoom level
    view_state = pdk.ViewState(
        latitude=12.9716,
        longitude=77.5946,
        zoom=11,
        pitch=0
    )

    # Combine layers
    r = pdk.Deck(
        layers=[driver_layer, order_layer],
        initial_view_state=view_state,
        tooltip={"text": "{type}"}
    )

    st.subheader("Map of Drivers (Red) & Orders (Green)")
    st.pydeck_chart(r)

if __name__ == "__main__":
    main()
