from itertools import product
import gurobipy as gp
from gurobipy import GRB

def loader():
    parent_path = './data/'

    # Load state information
    state_factors = {}
    sys_techs = []
    # Read the state factors file
    with open(parent_path + 'state_factors.txt', 'r') as file:
        lines = file.readlines()
    # Remove newline characters and split lines into two lists
    state_factor_names = lines[0].strip().split(',')
    # Create the dictionary using a loop
    for i, label in enumerate(state_factor_names):
        state_factors[label] = lines[i+2].strip().split(',')

    for _, vals in state_factors.items():
        for val in vals:
            sys_techs.append(val)
        
    # Load action information
    action_factors = {}
    # Read the action factors file
    with open(parent_path + 'action_factors.txt', 'r') as file:
        lines = file.readlines()
    # Remove newline characters and split lines into two lists
    action_factor_names = lines[0].strip().split(',')
    # Create the dictionary using a loop
    for i, label in enumerate(action_factor_names):
        action_factors[label] = lines[i+2].strip().split(',')
    
    switching_costs = []
    with open(parent_path + 'switching_costs.txt', 'r') as file:
        for line in file:
            values = line.strip().split(',')
            row = [float(val) for val in values]
            switching_costs.append(row)

    return state_factors, action_factors, switching_costs


def get_states_from_state_factors(state_factors):
    # Get the values from the factors
    values = list(state_factors.values())
    # Generate all possible combinations as tuples
    possible_states = list(product(*values))
    return possible_states

def state_to_state_one_hot_encoding(state_vector, state_factors):
    ohc = []
    i = 0
    for _, values in state_factors.items():
        for value in values:
            if value == state_vector[i]:
                ohc.append(1)
            else:
                ohc.append(0)
        i+=1
    return tuple(ohc)
            

def state_one_hot_encoding_to_state(state_ohc_vector, state_factors):
    state = []
    i = 0
    for _, values in state_factors.items():
        for value in values:
            if state_ohc_vector[i] == 1:
                state.append(value)
            i+=1
    return tuple(state)

def get_actions_from_action_factors(action_factors):
    # Get the values from the factors
    values = list(action_factors.values())
    # Generate all possible combinations as tuples
    possible_actions = list(product(*values))
    return possible_actions

def action_to_action_one_hot_encoding(action_vector, action_factors):
    ohc = []
    i = 0
    for _, values in action_factors.items():
        for value in values:
            if value == action_vector[i]:
                ohc.append(1)
            else:
                ohc.append(0)
        i+=1
    return tuple(ohc)

def action_one_hot_encoding_to_action(action_ohc_vector, action_factors):
    action = []
    i = 0
    for _, values in action_factors.items():
        for value in values:
            if action_ohc_vector[i] == 1:
                action.append(value)
            i+=1
    return tuple(action)


def lp_solve(coefficients, num_constr, const_num_coeff):
    # Create a new Gurobi model
    model = gp.Model("LP_Minimization")
    model.setParam('OutputFlag', 0)

    # Variables
    w1 = model.addVar(name="w1")
    w2 = model.addVar(name="w2")
    w3 = model.addVar(name="w3")
    w4 = model.addVar(name="w4")

    model.setObjective(w1 + w2 + w3 + w4, GRB.MINIMIZE)

    for i in range(num_constr):
        constraint = gp.LinExpr()
        for j in range(const_num_coeff):
            constraint += coefficients[i][j] * [w1, w2, w3, w4, 1][j]
        model.addConstr(constraint <= 0, name=f"constraint_{i + 1}")

    # Optimize the model
    model.optimize()

    # Print results
    if model.status == GRB.OPTIMAL:
        weights = [w1.x, w2.x, w3.x, w4.x]
        # Dispose of the model
        model.dispose()
        return weights
    else:
        # print("Optimal solution not found.")
        return []
