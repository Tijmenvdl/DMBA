# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 08:37:24 2024

@author: stijn
"""

from gurobipy import Model, GRB, quicksum
import numpy as np

def generate_hard_knapsack_data(num_items, weight_range, value_range, capacity_factor=0.5):
    """
    Generate challenging instances of the knapsack problem.

    Parameters:
        num_items (int): Number of items.
        weight_range (tuple): (min_weight, max_weight) range for generating weights.
        value_range (tuple): (min_value, max_value) range for generating values.
        capacity_factor (float): Fraction of the total weight that defines the knapsack capacity.

    Returns:
        weights (list of int): Weights of the items.
        values (list of int): Values of the items.
        capacity (int): Capacity of the knapsack.
    """

    # Generate weights and values that are very close to each other
    weights = np.random.randint(weight_range[0], weight_range[1], size=num_items)
    values = np.random.randint(value_range[0], value_range[1], size=num_items)
    
    # Calculate the total weight of all items
    total_weight = np.sum(weights)
    
    # Set the knapsack capacity to be a challenging fraction of the total weight
    capacity = int(total_weight * capacity_factor)
    
    return list(weights), list(values), capacity

# Example usage
num_items = 50  # Number of items in the knapsack problem
weight_range = (90, 110)  # Weights are close in range, making decisions difficult
value_range = (80, 120)  # Values are also close in range
capacity_factor = 0.5  # Knapsack can carry half of the total weight of all items

weights, values, capacity = generate_hard_knapsack_data(num_items, weight_range, value_range, capacity_factor)

print("Weights:", weights)
print("Values:", values)
print("Knapsack Capacity:", capacity)




def knapsackSolver(num_items, capacity, values, weights):
    # Create a Gurobi model
    model = Model("Knapsack")

    # Suppress Gurobi output for cleaner display (you can remove this if you want detailed solver information)
    model.setParam('OutputFlag', 0)

    # Define decision variables, x[i] is 1 if item i is included, otherwise 0
    x = model.addVars(num_items, vtype=GRB.BINARY, name="x")

    # Define the objective function: Maximize sum of values of the included items
    model.setObjective(quicksum(values[i] * x[i] for i in range(num_items)), GRB.MAXIMIZE)

    # Define the weight constraint: Sum of weights should not exceed the capacity
    model.addConstr(quicksum(weights[i] * x[i] for i in range(num_items)) <= capacity, "Weight")

    # Optimize the model
    model.optimize()

    # Extracting the selected items
    selected_items = [i for i in range(num_items) if x[i].X > 0.5]
    total_value = sum(values[i] for i in selected_items)
    total_weight = sum(weights[i] for i in selected_items)

    # Output the results
    print("Status:", model.Status)
    print("Selected Items:", selected_items)
    print("Total Value:", total_value)
    print("Total Weight:", total_weight)

    return selected_items

# Example usage with generated data
num_items = 10
max_weight = 20
max_value = 50

# Generate data using the function provided earlier
weights, values, capacity = generate_knapsack_data(num_items, max_weight, max_value)
knapsackSolver(num_items, capacity, values, weights)




def knapsackSolver(num_items, capacity, values, weights):
    # Create a 2D list to store the maximum value for each subproblem
    dp = [[0 for _ in range(capacity + 1)] for _ in range(num_items + 1)]

    # Fill the DP table
    for i in range(1, num_items + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                # Option to include the current item
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                # Can't include the current item, so the value remains the same as without it
                dp[i][w] = dp[i - 1][w]

    # Backtrack to find the items that make up the optimal solution
    w = capacity
    selected_items = []
    for i in range(num_items, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= weights[i - 1]

    # Output the results
    print("Selected Items:", selected_items)
    print("Total Value:", dp[num_items][capacity])
    print("Total Weight:", sum(weights[i] for i in selected_items))

    return selected_items

# Example usage with generated data
num_items = 10
max_weight = 20
max_value = 50

# Generate data using the function provided earlier
weights, values, capacity = generate_knapsack_data(num_items, max_weight, max_value)
knapsackSolver(num_items, capacity, values, weights)




def knapsackSolver(num_items, capacity, values, weights):
    # Calculate value-to-weight ratio for each item
    ratios = [(values[i] / weights[i], i) for i in range(num_items)]

    # Sort items by value-to-weight ratio in descending order
    ratios.sort(key=lambda x: x[0], reverse=True)

    total_value = 0
    total_weight = 0
    selected_items = []

    # Iterate through the sorted items and add them to the knapsack if possible
    for ratio, i in ratios:
        if total_weight + weights[i] <= capacity:
            selected_items.append(i)
            total_weight += weights[i]
            total_value += values[i]

    # Output the results
    print("Selected Items:", selected_items)
    print("Total Value:", total_value)
    print("Total Weight:", total_weight)

    return selected_items

# Example usage with generated data
num_items = 10
max_weight = 20
max_value = 50

# Generate data using the function provided earlier
weights, values, capacity = generate_knapsack_data(num_items, max_weight, max_value)
knapsackSolver(num_items, capacity, values, weights)
