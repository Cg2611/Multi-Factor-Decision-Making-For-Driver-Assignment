# assignment.py
import numpy as np
import pulp

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Roughly calculates distance in km between two lat/lon points using Haversine.
    For short distances in a city, Euclidean could suffice, but let's do Haversine for demonstration.
    """
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * (np.sin(dlon/2)**2)
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def compute_pairwise_cost(
    drivers_df, orders_df, 
    traffic_level=1.0, 
    w_time=1.0, 
    w_cost=1.0, 
    w_rating=1.0, 
    w_incentive=-0.5, 
    w_fairness=0.5,
    w_surge=-2.0, 
    current_time=10
):
    """
    Returns a dictionary of costs for each (driver, order) pair.

    Additional factors:
    - traffic_level multiplies distance
    - w_fairness: a positive value => if driver is 'overused', we increase cost
    - w_surge: negative => reduce cost if order is in surge zone => encourages assignment
    - If driver is unavailable (shift or break), we can set cost = very large => effectively unassignable
    """
    costs = {}

    for _, driver in drivers_df.iterrows():
        d_id = driver["driver_id"]
        d_lat = driver["loc_lat"]
        d_lon = driver["loc_lon"]
        d_rating = driver["rating"]
        d_cost_factor = driver["cost_factor"]
        d_incentive = driver["incentive_progress"]
        shift_start = driver["shift_start"]
        shift_end = driver["shift_end"]
        next_break = driver["next_break_in_minutes"]

        # If current_time is outside shift or break is imminent => driver effectively unavailable
        # We'll use a very large cost to discourage assignment.
        if not (shift_start <= current_time <= shift_end):
            # Outside shift => set cost extremely high
            for _, order in orders_df.iterrows():
                costs[(d_id, order["order_id"])] = 999999  # effectively disqualifies
            continue

        # If next break is extremely soon, we can either disqualify or heavily penalize.
        # Let's penalize if < 10 minutes to break.
        break_penalty = 0
        if next_break < 10:
            break_penalty = 50  # some large penalty

        # Also incorporate a fairness penalty. 
        # For demonstration, let's say if incentive_progress is > 5, 
        # driver has already done many orders => slight penalty
        fairness_penalty = w_fairness * max(0, driver["incentive_progress"] - 5)

        for _, order in orders_df.iterrows():
            o_id = order["order_id"]
            o_lat = order["loc_lat"]
            o_lon = order["loc_lon"]
            o_priority = order["priority"]
            o_surge = order["surge_zone"]

            # timeCost => distance * traffic_level
            dist_km = haversine_distance(d_lat, d_lon, o_lat, o_lon)
            time_cost = w_time * dist_km * traffic_level

            # driverCost => cost_factor
            driver_cost = w_cost * d_cost_factor

            # ratingPenalty => if vip/high_value and rating < 4 => penalize
            rating_penalty = 0
            if o_priority in ["vip", "high_value"]:
                rating_penalty = w_rating * (5.0 - d_rating) if d_rating < 5 else 0

            # incentiveAdjustment => if d_incentive >= 8 => negative cost (discount)
            incentive_adj = w_incentive if d_incentive >= 8 else 0

            # surgeAdjustment => if o_surge is True => negative cost to encourage assignment
            surge_adjustment = w_surge if o_surge else 0

            # Sum up
            pair_cost = (
                time_cost 
                + driver_cost 
                + rating_penalty
                + incentive_adj
                + fairness_penalty
                + surge_adjustment
                + break_penalty
            )

            costs[(d_id, o_id)] = pair_cost

    return costs

def assign_drivers_to_orders(drivers_df, orders_df, costs, driver_capacity=2):
    """
    Solve the assignment problem using PuLP (linear optimization).
    Minimizes total cost of assignments subject to constraints:
      - Each order must be assigned to exactly one driver
      - Each driver can handle up to `driver_capacity` orders
    """
    driver_ids = drivers_df["driver_id"].tolist()
    order_ids = orders_df["order_id"].tolist()

    # Build the model
    model = pulp.LpProblem("DriverAssignmentExtended", pulp.LpMinimize)

    # Decision variables: x[d_id, o_id] ∈ {0,1}
    x = pulp.LpVariable.dicts(
        'assign',
        ((d_id, o_id) for d_id in driver_ids for o_id in order_ids),
        cat=pulp.LpBinary
    )

    # Objective: minimize sum of costs * x
    model += pulp.lpSum([costs[(d_id, o_id)] * x[(d_id, o_id)] 
                         for d_id in driver_ids 
                         for o_id in order_ids])

    # Constraint 1: Each order is assigned to exactly one driver
    for o_id in order_ids:
        model += pulp.lpSum([x[(d_id, o_id)] for d_id in driver_ids]) == 1

    # Constraint 2: Each driver can handle at most `driver_capacity` orders
    for d_id in driver_ids:
        model += pulp.lpSum([x[(d_id, o_id)] for o_id in order_ids]) <= driver_capacity

    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # Collect results
    assignment_results = []
    for d_id in driver_ids:
        for o_id in order_ids:
            if pulp.value(x[(d_id, o_id)]) == 1:
                cost_val = costs[(d_id, o_id)]
                assignment_results.append((d_id, o_id, cost_val))

    return assignment_results
