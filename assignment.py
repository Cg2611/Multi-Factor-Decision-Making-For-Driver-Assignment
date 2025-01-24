# assignment.py
import numpy as np
import pulp

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def compute_pairwise_cost(drivers_df, orders_df, 
                          w_time=1.0, w_cost=1.0, w_rating=1.0, w_incentive=-0.5):
    """
    Returns a dictionary of costs for each (driver, order) pair.
    
    w_incentive negative => If driver is near incentive threshold, it lowers cost
    so that algorithm is more likely to pick that driver.
    """
    costs = {}
    
    for _, driver in drivers_df.iterrows():
        d_id = driver["driver_id"]
        d_loc_x = driver["loc_x"]
        d_loc_y = driver["loc_y"]
        d_rating = driver["rating"]
        d_cost_factor = driver["cost_factor"]
        d_incentive = driver["incentive_progress"]
        
        for _, order in orders_df.iterrows():
            o_id = order["order_id"]
            o_loc_x = order["loc_x"]
            o_loc_y = order["loc_y"]
            o_priority = order["priority"]
            o_required_sla = order["required_sla"]
            
            # 1) timeCost => distance
            dist = euclidean_distance(d_loc_x, d_loc_y, o_loc_x, o_loc_y)
            time_cost = w_time * dist
            
            # 2) driverCost => cost_factor
            driver_cost = w_cost * d_cost_factor
            
            # 3) ratingCost => if order is high_value/vip, 
            #    penalize low-rating drivers
            #    e.g., cost increases for low rating
            if o_priority in ["vip", "high_value"]:
                # simple approach: cost = w_rating*(5 - driver_rating)
                rating_penalty = w_rating * (5.0 - d_rating)
            else:
                rating_penalty = 0
            
            # 4) incentiveAdjustment => if driver_incentive is close to 10, 
            #    we reduce cost slightly
            #    e.g., if d_incentive >= 8, it’s a better candidate
            if d_incentive >= 8:
                incentive_adj = w_incentive  # negative => reduce cost
            else:
                incentive_adj = 0
            
            # Summation
            pair_cost = time_cost + driver_cost + rating_penalty + incentive_adj
            
            # Store in dictionary
            costs[(d_id, o_id)] = pair_cost

    return costs

def assign_drivers_to_orders(drivers_df, orders_df, costs):
    """
    Solve the assignment problem using PuLP (linear optimization).
    Minimizes total cost of assignments subject to constraints:
      - Each order is assigned to exactly one driver
      - Each driver can handle at most 1 order (in this simple example)
    """
    driver_ids = drivers_df["driver_id"].tolist()
    order_ids = orders_df["order_id"].tolist()

    # Build the model
    model = pulp.LpProblem("DriverAssignment", pulp.LpMinimize)

    # Decision variables: x[d_id, o_id] ∈ {0,1}
    x = pulp.LpVariable.dicts('assign',
                              ((d_id, o_id) for d_id in driver_ids for o_id in order_ids),
                              cat=pulp.LpBinary)

    # Objective: minimize sum of costs * x
    model += pulp.lpSum([costs[(d_id, o_id)] * x[(d_id, o_id)] 
                         for d_id in driver_ids 
                         for o_id in order_ids])

    # Constraint 1: Each order is assigned to exactly one driver
    for o_id in order_ids:
        model += pulp.lpSum([x[(d_id, o_id)] for d_id in driver_ids]) == 1

    # Constraint 2: Each driver can handle at most one order (in this demo)
    # For real quick commerce, you might allow capacity > 1
    for d_id in driver_ids:
        model += pulp.lpSum([x[(d_id, o_id)] for o_id in order_ids]) <= 1

    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # Collect results
    assignment_results = []
    for d_id in driver_ids:
        for o_id in order_ids:
            if pulp.value(x[(d_id, o_id)]) == 1:
                assignment_results.append((d_id, o_id, costs[(d_id, o_id)]))

    return assignment_results
