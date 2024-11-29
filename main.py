import sys
import os
import copy
from datetime import datetime
import pickle
import random
import time
import itertools

from gurobipy import *
import networkx as nx
import numpy as np

from classes import Tree, Data

silence_all = 1

env = Env()

list_np_seeds = [64117468, 61426413, 92232177, 37229339, 84464803, 48242438, 92911353, 39071652, 22879523, 29557695,
                 77869602, 72482311, 30346251, 36743271, 56698155, 56542032, 77156023, 15059166, 71254118, 24486443]
only_1_core = True


def make_data(no_scenarios, size_graph, seed, budget_type, method, time_limit, time_limit2, param_budget):
    """
    Parameter:
        - no_scenarios (int)
        - size_graph (int)
        - seed (int)
        - budget_type (str)
        - method (str)
        - time_limit (int)
        - time_limit2 (int)
        - param_budget (float)

    Returns:
        - data (Data): Data for the run
    """
    # make costs
    np.random.seed(seed)
    no_edges = (size_graph - 1) * size_graph * 2
    all_costs = np.zeros((1250, 144))
    midpoints = []
    deviations = []
    list_j = []
    for e in range(144):
        t_temp_m = []
        t_temp_d = []
        for i in range(3):
            t_temp_m.append(np.random.random_sample() * 20 + 10)
            t_temp_d.append(np.random.random_sample() * 0.25)
        midpoints.append(t_temp_m)
        deviations.append(t_temp_d)
    for i in range(1250):
        t = np.random.randint(0, 3)
        for e in range(144):
            ran = np.random.random_sample()
            ub = (1 + deviations[e][t]) * midpoints[e][t]
            lb = (1 - deviations[e][t]) * midpoints[e][t]
            all_costs[i][e] = ran * (ub - lb) + lb
    for i in range(no_scenarios):
        list_j.append([0])
    dict_costs = {range(no_scenarios)[i]: range(no_scenarios)[i] for i in range(no_scenarios)}
    all_costs = np.round(all_costs, decimals=5)
    c_train = all_costs[:no_scenarios, :no_edges]
    c_test = all_costs[-1000:, :no_edges]

    # make graph
    list_edges = []  # list of edges for the use in the dictionary
    graph = nx.DiGraph()
    for node in range(size_graph * size_graph):
        graph.add_node(node)
    for node in range(size_graph * size_graph - size_graph):  # if node is not in the highest row
        if not ((node + 1) / size_graph).is_integer():  # if node is not in the last column
            graph.add_edge(node, node + 1)  # add edge to the right
            list_edges.append((node, node + 1))  # add edge to the dictionary
        graph.add_edge(node, node + size_graph)  # add edge to the top
        list_edges.append((node, node + size_graph))  # add edge to the dictionary
    for node in range(size_graph * size_graph - size_graph,
                      size_graph * size_graph - 1):  # if node is in the highest row and not in the last column
        graph.add_edge(node, node + 1)  # add edge to the right
        list_edges.append((node, node + 1))  # add edge to the dictionary
    ind = -1
    for edge in graph.edges:
        ind += 1
        graph.edges[edge]["edg_ind"] = ind
    dict_edges = {k: v for v, k in enumerate(list_edges)}  # create dictionary for the cost matrix

    # determine maximum difference
    difs = []
    for e_ind in range(len(graph.edges)):
        difs.append(max(c_train[:, e_ind])-min(c_train[:, e_ind]))
    max_dif = max(difs)

    # calculate budget
    if budget_type == "l":
        budget = param_budget * 2 * max_dif
    else:
        budget = param_budget * 2 * no_scenarios * max_dif

    # store in Data object
    data = Data(seed, size_graph, no_scenarios, c_train, c_test, list_j, budget, budget_type, param_budget, graph,
                dict_edges, dict_costs, 0, size_graph * size_graph - 1, method, time_limit, time_limit2)

    return data


def calc_theta(costs):
    """
    Returns a list of candidates for branching-thresholds based on a given cost-matrix.

    Parameter:
        - costs (ndarray): Cost-matrix

    Returns:
        - theta_list (list) - Contains candidates for branching-thresholds
    """
    # sorts the possible costs of each edge and returns the midpoint between following values for the use as possible
    # value for branching
    theta_list = []
    for j in range(len(costs[0])):
        theta_edge = []
        cost_temp = np.unique(costs[:, j])
        cost_temp = np.sort(cost_temp)
        for i in range(len(cost_temp) - 1):
            current_value = cost_temp[i]
            next_value = cost_temp[i + 1]
            theta = round((current_value + next_value) / 2, 7)
            theta_edge.append(theta)
        theta_list.append(theta_edge)
    return theta_list


def evaluate2(data, problem, compare):
    """
    Parameter:
        - data (Data)
        - problem (str)
        - compare (bool): if True return bool, if False return objective

    Return:
        - coherent (bool) OR objective (float)
    """
    costs = data.costs
    if problem == "en":
        ind_obs = data.get_last_obs()
        observations = data.obs[ind_obs, :]
        if compare:
            ref_obj = data.rob_objs[-1]
    else:
        ind_obs = range(len(costs))
        observations = costs
        if compare:
            ref_obj = data.nom_objs[-1]
    edges = range(len(costs[0]))

    sols = data.tree_last.get_sols(observations)
    objective = 0
    for ind, obs in enumerate(ind_obs):
        for edge in edges:
            base_scenario = data.dict_c[obs]
            objective += sols[ind][edge] * costs[base_scenario][edge]

    if compare:
        if ref_obj * 0.999 <= objective <= ref_obj * 1.001:
            coherent = True
        else:
            coherent = False

        return coherent
    else:
        return objective


def master_problem2(data):
    """
    Returns the solution and its parameters for the master-problem.

    Parameter:
        - data_run (DataRun): Data for the current run

    Returns:
        - data_run (DataRun) - Updated data of the current run
    """
    start_time = time.time()
    time_limit = max(10, data.time_left)
    graph, source, sink, costs, obs = data.graph, data.s, data.t, data.costs, data.obs
    dict_edges, dict_costs, list_j = data.dict_e, data.dict_c, data.mapping

    # definition and params of the main model
    m = Model(name="master", env=env)
    m.Params.OutputFlag = silence_all
    if only_1_core:
        m.Params.Threads = 1
    m.setParam('TimeLimit', time_limit)

    # definition of the params depending on the actual instance
    F = range(len(graph.edges))  # set of features (one feature for every edge)
    I = range(len(obs))  # set of scenarios (datapoints)
    L = [3, 4, 5, 6]  # set of leaves of the DT
    Nstrich = [0, 1, 2]  # set of branching nodes for the case of a symmetric DT
    Theta = calc_theta(costs)  # calculating possible values for branching
    Kanten_int = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]  # edges of the DT
    edges = graph.edges  # set of edges

    M = []  # set of values for M
    for i in range(len(costs)):  # Only for every cost-scenario (NOT every observation)
        # calculate M_i for every scenario by calculating the longest path in the grid given scenario i
        m_lp = Model(name="long_path", env=env)
        m_lp.Params.OutputFlag = silence_all
        if only_1_core:
            m_lp.Params.Threads = 1
        y_lp = m_lp.addVars(range(len(edges)), vtype=GRB.BINARY, name="y_lp")
        m_lp.setObjective(quicksum(costs[i][dict_edges[e]] * y_lp[dict_edges[e]] for e in edges), GRB.MAXIMIZE)
        m_lp.addConstr((quicksum(y_lp[dict_edges[(source, j)]] for j in graph.neighbors(source)) == 1),
                       name="lp_source")
        m_lp.addConstrs((quicksum(y_lp[dict_edges[e]] for e in graph.in_edges(nbunch=i)) - quicksum(y_lp[dict_edges[(i, j)]]
                         for j in graph.neighbors(i)) == 0 for i in graph.nodes if i not in {source, sink}), name="lp_flow")
        m_lp.optimize()
        M.append(m_lp.getObjective().getValue())

    # construction of the main optimization problem
    u = m.addVar(vtype=GRB.CONTINUOUS, name="u")
    z = m.addVars(I, Kanten_int, vtype=GRB.BINARY, name="z")
    z_2 = m.addVars(I, vtype=GRB.CONTINUOUS, name="zstrich")
    indices = [(i, j, k) for i in Nstrich for j, K in zip(F, Theta) for k in K]
    b = m.addVars(indices, vtype=GRB.BINARY, name="b")

    # if the solutions shouldn't be part of the problem use them as parameters
    if data.method == "s" or data.method == "a":
        y = np.asarray(data.start_sols)
        y = np.r_[np.zeros((3,len(edges))), y]
    else:
        y = m.addVars(L, range(len(edges)), vtype=GRB.BINARY, name="y")

    # objective
    m.setObjective(u, GRB.MINIMIZE)

    # constraints for objective
    m.addConstrs((u >= quicksum(z_2[i] for i in I if j in list_j[i]) for j in range(max(max(list_j)) + 1)),
                 name="epigraph")

    # constraints for the construction of the tree
    m.addConstrs((quicksum(b[n, f, theta] for f in F for theta in Theta[f]) == 1 for n in Nstrich),
                 name="tree_branching")

    # constraints for the measurement of the flow through the tree
    m.addConstrs(quicksum(b[0, f, theta] for f in F for theta in Theta[f] if obs[i][f] <= theta) >= z[i, 0, 1] for i in I)  # branching on node 0, left
    m.addConstrs(quicksum(b[0, f, theta] for f in F for theta in Theta[f] if obs[i][f] > theta) >= z[i, 0, 2] for i in I)  # branching on node 0, right
    m.addConstrs(quicksum(b[1, f, theta] for f in F for theta in Theta[f] if obs[i][f] <= theta) >= z[i, 1, 3] for i in I)  # branching on node 1, left
    m.addConstrs(quicksum(b[1, f, theta] for f in F for theta in Theta[f] if obs[i][f] > theta) >= z[i, 1, 4] for i in I)  # branching on node 1, right
    m.addConstrs(quicksum(b[2, f, theta] for f in F for theta in Theta[f] if obs[i][f] <= theta) >= z[i, 2, 5] for i in I)  # branching on node 2, left
    m.addConstrs(quicksum(b[2, f, theta] for f in F for theta in Theta[f] if obs[i][f] > theta) >= z[i, 2, 6] for i in I)  # branching on node 2, right
    m.addConstrs(z[i, 0, 1] == z[i, 1, 3] + z[i, 1, 4] for i in I)  # flow conservation, node 1
    m.addConstrs(z[i, 0, 2] == z[i, 2, 5] + z[i, 2, 6] for i in I)  # flow conservation, node 2
    m.addConstrs(z[i, 0, 1] + z[i, 0, 2] >= 1 for i in I)  # outgoing flow of node 0 has to be 1

    # constraints for the shortest path problems (only needed if the solutions are also optimized)
    if not data.method == "s" or not data.method == "a":
        m.addConstrs((quicksum(y[n, dict_edges[(source, j)]] for j in graph.neighbors(source)) == 1 for n in L), name="sp_source")
        m.addConstrs((quicksum(y[n, dict_edges[e]] for e in graph.in_edges(nbunch=i)) - quicksum(
            y[n, dict_edges[(i, j)]] for j in graph.neighbors(i)) == 0 for i in graph.nodes if i not in {source, sink} for n in L), name="sp_flow")
    # constraints for the calculation of objective
    m.addConstrs((z_2[i] + M[dict_costs[i]] * (1 - z[i, 1, 3]) >= quicksum(
        costs[dict_costs[i]][dict_edges[e]] * y[3, dict_edges[e]] for e in edges) for i in I), name="costs1")
    m.addConstrs((z_2[i] + M[dict_costs[i]] * (1 - z[i, 1, 4]) >= quicksum(
        costs[dict_costs[i]][dict_edges[e]] * y[4, dict_edges[e]] for e in edges) for i in I), name="costs2")
    m.addConstrs((z_2[i] + M[dict_costs[i]] * (1 - z[i, 2, 5]) >= quicksum(
        costs[dict_costs[i]][dict_edges[e]] * y[5, dict_edges[e]] for e in edges) for i in I), name="costs3")
    m.addConstrs((z_2[i] + M[dict_costs[i]] * (1 - z[i, 2, 6]) >= quicksum(
        costs[dict_costs[i]][dict_edges[e]] * y[6, dict_edges[e]] for e in edges) for i in I), name="costs4")

    # if we are not in the 0th iteration, read warm start solution
    if data.iteration != 0 or data.method == "a":
        m.update()
        m.read("last_sol_"+str(data.warm_start_id)+".sol")

    # solve model
    m.optimize()

    # save solutions
    tree = Tree(2)
    if data.method == "s" or data.method == "a":
        for l in L:
            tree.add_sol(l, y[l])
    else:
        for l in L:
            sol_vector = []
            for f in F:
                sol_vector.append(y[l, f].x)
            tree.add_sol(l, sol_vector)

    # save branching decisions
    for (i), v in b.items():
        if v.X > 0.5:
            tree.add_branching(i)

    # save the tree generated
    tree.nom_obj = m.ObjVal
    data.add_tree(tree)

    # optimize thresholds
    if data.method != "nom":
        data = opt_thresholds(data)

    # save (partial) solution for warm start
    start_time_sol = time.time()
    m.write("last_sol_"+str(data.warm_start_id)+".sol")
    with open("last_sol_"+str(data.warm_start_id)+".sol", 'r') as file:
        lines = file.readlines()
    # for optimal approach save solutions in the leaves and branching decisions
    if data.method == "o" or "nom":
        filtered_lines = [line for line in lines if not line.startswith('z') and not line.startswith('u')]
    # for alternating heu only save solutions in the leaves
    else:
        filtered_lines = [line for line in lines if not line.startswith('z') and not line.startswith('u') and not
        line.startswith('b')]
    with open("last_sol_"+str(data.warm_start_id)+".sol", 'w') as file:
        file.writelines(filtered_lines)
    data.warmstarttime = time.time() - start_time_sol

    # save time related information
    runtime = time.time() - start_time
    data.time_mp.append(runtime)
    data.time_left -= time.time() - start_time

    return data


def adversary2(data, use_test=False, dif_gamma=None, dif_tree=None, only_obj=False, best_tree=False):
    """
    Returns the solution and its parameters for the adversary problem (global AND local budget).

    Parameter:
        - data_run (DataRun): Data of the current run
        - use_test (bool): If True use test data
        - dif_gamma (float): Use for evaluation with dif gamma
        - dif_tree (Tree): Use for evaluation with dif tree
        - only_obj (bool): NOT USED
        - best_tree (bool): If True use best instead of last tree

    Returns:
        - data_run (DataRun) - Updated data of the current run
    """
    start_time = time.time()
    obs = data.obs
    if use_test:
        costs = data.costs_test
    else:
        costs = data.costs
    dict_costs = data.dict_c
    list_j = data.mapping
    if dif_gamma is None:
        budget = data.budget
    else:
        budget = dif_gamma
    budget_type = data.budget_type
    iteration = data.iteration
    if best_tree == False:
        if dif_tree is None:
            tree = data.tree_last
        else:
            tree = dif_tree
    else:
        tree = data.tree_rob

    # preprocessing
    delta, p = tree.pre_adv(costs, budget)

    # START OPTIMIZATION
    leaves = [0, 1, 2, 3]
    I = range(len(costs))
    m = Model(name="enemy_problem", env=env)
    m.Params.OutputFlag = silence_all
    m.setParam('TimeLimit', max(5, data.time_left))
    if only_1_core:
        m.Params.Threads = 1

    xi = m.addVars(I, leaves, vtype=GRB.BINARY, name="xi")
    m.setObjective(quicksum(delta[i][n] * xi[i, n] for i in I for n in leaves), GRB.MAXIMIZE)
    if budget_type == "g":
        m.addConstr(quicksum(p[i][n] * xi[i, n] for i in I for n in leaves) <= budget, name="budget_g")
    elif budget_type == "l":
        m.addConstrs((quicksum(p[i][n] * xi[i, n] for n in leaves) <= budget for i in I), name="budget_l")
    else:
        print("error")
    m.addConstrs(quicksum(xi[i, n] for n in leaves) == 1 for i in I)
    m.optimize()

    if only_obj:
        return m.ObjVal

    # postprocessing
    xi_list = []
    for (i), v in xi.items():
        if v.X > 0.5:
            xi_list.append(i)
    list_j, observations, dict_costs = tree.post_adv(xi_list, list_j, obs, iteration, dict_costs)

    runtime = time.time() - start_time
    data.time_left -= runtime

    # save data and check results
    data.add_scens(dict_costs, list_j, observations, m.ObjVal, runtime)

    return data


def opt_thresholds(data):
    start_time = time.time()
    nom_scenarios = data.obs[:data.no_train, :]
    # get existing queries
    f1, f2, f3 = data.tree_last.f1, data.tree_last.f2, data.tree_last.f3
    t1, t2, t3 = data.tree_last.t1, data.tree_last.t2, data.tree_last.t3
    # find the closest observations
    ubs = [min([m for m in nom_scenarios[:, f1] if m > t1]), min([m for m in nom_scenarios[:, f2] if m > t2]),
           min([m for m in nom_scenarios[:, f3] if m > t3])]
    lbs = [max([m for m in nom_scenarios[:, f1] if m < t1]), max([m for m in nom_scenarios[:, f2] if m < t2]),
           max([m for m in nom_scenarios[:, f3] if m < t3])]
    # divide these intervals in steps of 10 %
    poss_ts = [[], [], []]
    for ind_query in range(3):
        for factor in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            poss_ts[ind_query].append(lbs[ind_query] + factor * (ubs[ind_query] - lbs[ind_query]))

    # iterate over them to find best splits
    first_obj = adversary2(data, only_obj=True)
    best_obj = copy.deepcopy(first_obj)
    best_tree = data.tree_last

    for t_new1 in poss_ts[0]:
        for t_new2 in poss_ts[1]:
            for t_new3 in poss_ts[2]:
                new_tree = data.tree_last
                new_tree.reset_nom()
                new_tree.t1, new_tree.t2, new_tree.t3 = t_new1, t_new2, t_new3
                obj_new_tree = adversary2(data, dif_tree=new_tree, only_obj=True)
                # if better splits are found update
                if obj_new_tree < best_obj:
                    best_obj = copy.deepcopy(obj_new_tree)
                    best_tree = copy.deepcopy(new_tree)

    data.time_opt_thrshlds.append(time.time() - start_time)
    data.tree_last = copy.deepcopy(best_tree)

    if best_obj < first_obj:
        data.marker_opt_thrshld = True

    return data


def iterative(data):  # for optimal and alternating approach
    """
    Returns the data of a run with the given parameters using the optimal iterative approach.

    Parameter:
        - - obj_run (DataRun) - Instance

    Returns:
        - obj_run (DataRun) - Information regarding the performed run
    """
    # while time limit is not violated
    st = time.time()
    time_limit = data.time_left
    while time.time() - st <= time_limit:
        data = master_problem2(data)
        data = adversary2(data)
        if data.check_convergence():
            break

    data.rtime = time.time() - st
    if data.method == "a":
        data.start_for_alternating_heu = data.tree_last

    return data


def heu_fix_tree(data, no_of_trys=10**8, opt_time_limit=None):
    """
    Return a set of decision trees which are generated by applying the heuristic where trees are sampled and solutions
    are optimized.

    Parameter:
        - data (Data): Object containing the instance
        - no_of_trys (int): Number of trees to sample
        - opt_time_limit (int): Use if a different time limit should be used

    Returns:
        - data (Data) - Updated object containing the instance
    """
    st = time.time()
    if opt_time_limit is None:
        time_limit = max(10, data.time_left)
    else:
        time_limit = max(10,opt_time_limit)
    budget, graph, source, sink, costs = data.budget, data.graph, data.s, data.t, data.costs
    dict_edges = data.dict_e
    edges = graph.edges
    I = len(costs)
    L = [3, 4, 5, 6]
    Theta = calc_theta(costs)
    stop = False

    # lists for saving data
    best = None
    best_obj = 100000000

    # generate solution candidates
    for _ in range(no_of_trys):
        if data.method == "a" and data.start_for_alternating_heu is not None:
            tree = data.start_for_alternating_heu
        # sample tree randomly
        else:
            tree = Tree(2)
            tree.f1, tree.f2, tree.f3 = random.choice(range(len(costs[0]))), random.choice(range(len(costs[0]))), random.choice(range(len(costs[0])))
            tree.t1, tree.t2, tree.t3 = random.choice(Theta[tree.f1]), random.choice(Theta[tree.f2]), random.choice(Theta[tree.f3])
            tree.s1, tree.s2, tree.s3, tree.s4 = [0] * len(edges), [0] * len(edges), [0] * len(edges), [0] * len(edges)

        # preprocessing
        d, p = tree.pre_adv(costs, budget)

        # local budget
        if data.budget_type == "l":
            l_list = []
            for row in p:
                indices = [i for i, value in enumerate(row) if value <= budget]
                l_list.append(indices)

            m_loc = Model(env=env)
            m_loc.Params.OutputFlag = silence_all
            if only_1_core:
                m_loc.Params.Threads = 1
            m_loc.setParam('TimeLimit', time_limit)
            q = m_loc.addVars(I, vtype=GRB.CONTINUOUS, name="q")
            y = m_loc.addVars(L, range(len(edges)), vtype=GRB.BINARY, name="y")
            m_loc.setObjective(quicksum(q[i] for i in range(I)))
            m_loc.addConstrs(q[i] >= quicksum(costs[i][dict_edges[e]] * y[n+3, dict_edges[e]] for e in edges) for i in
                             range(I) for n in l_list[i])
            m_loc.addConstrs((quicksum(y[n, dict_edges[(source, j)]] for j in graph.neighbors(source)) == 1 for n in L), name="sp_source")
            m_loc.addConstrs((quicksum(y[n, dict_edges[e]] for e in graph.in_edges(nbunch=i)) - quicksum(y[n, dict_edges[(i, j)]] for j in graph.neighbors(i))
                              == 0 for i in graph.nodes if i not in {source, sink} for n in L), name="sp_flow")

            # if using the alternating heuristic use warm start
            if data.method == "a" and data.counter_int_iteration > 1:
                m_loc.update()
                m_loc.read("last_sol_" + str(data.warm_start_id) + ".sol")
            m_loc.optimize()

            # if current tree is the best, save it
            if m_loc.ObjVal <= best_obj:
                best_obj = m_loc.ObjVal
                tree.rob_obj = m_loc.ObjVal
                for l in L:
                    sol_vector = []
                    for e in range(len(edges)):
                        sol_vector.append(y[l, e].x)
                    tree.add_sol(l, sol_vector)
                best = tree

                if time.time() - st > time_limit:
                    stop = True

        else:
            # global budget
            iteration = 0
            convergence = False
            while not convergence and time.time() - st < time_limit:
                # master problem
                iteration += 1
                if iteration == 1:
                    n_list = []
                    for row in p:
                        indices = [i for i, value in enumerate(row) if value <= 0.000001]
                        n_list.append(indices)
                    n_list = [[element for sublist in n_list for element in sublist]]

                m_glob_mp = Model(env=env)
                m_glob_mp.Params.OutputFlag = silence_all
                if only_1_core:
                    m_glob_mp.Params.Threads = 1
                m_glob_mp.setParam('TimeLimit', time_limit)
                q = m_glob_mp.addVar(vtype=GRB.CONTINUOUS)
                y = m_glob_mp.addVars(L, range(len(edges)), vtype=GRB.BINARY, name="y")
                m_glob_mp.setObjective(q)
                m_glob_mp.addConstrs(q >= quicksum(costs[i][dict_edges[e]] * y[n_list[j][i] + 3, dict_edges[e]]
                                                   for e in edges for i in range(I)) for j in range(iteration))
                m_glob_mp.addConstrs((quicksum(y[n, dict_edges[(source, j)]] for j in graph.neighbors(source)) == 1
                                      for n in L), name="sp_source")
                m_glob_mp.addConstrs((quicksum(y[n, dict_edges[e]] for e in graph.in_edges(nbunch=i))
                                      - quicksum(y[n, dict_edges[(i, j)]] for j in graph.neighbors(i))
                                      == 0 for i in graph.nodes if i not in {source, sink} for n in L), name="sp_flow")

                # if alternating heu use warm start
                if data.method == "a" and data.counter_int_iteration > 1:
                    m_glob_mp.update()
                    m_glob_mp.read("last_sol_" + str(data.warm_start_id) + ".sol")
                m_glob_mp.optimize()

                # save solutions in a list
                y_en = []
                for l in L:
                    sol_vector = []
                    for e in range(len(edges)):
                        sol_vector.append(y[l, e].x)
                    y_en.append(sol_vector)

                # ADVERSARY PROBLEM
                m_glob_ad = Model(env=env)
                m_glob_ad.Params.OutputFlag = silence_all
                if only_1_core:
                    m_glob_ad.Params.Threads = 1
                m_glob_ad.setParam('TimeLimit', time_limit)
                xi = m_glob_ad.addVars(I, L, vtype=GRB.BINARY, name="xi")
                m_glob_ad.setObjective(quicksum(
                    costs[i][dict_edges[e]] * y_en[n - 3][dict_edges[e]] * xi[i, n] for e in edges for i in range(I) for
                    n in L), GRB.MAXIMIZE)
                m_glob_ad.addConstr(quicksum(xi[i, n] * p[i][n - 3] for n in L for i in range(I)) <= budget)
                m_glob_ad.addConstrs(quicksum(xi[i, n] for n in L) <= 1 for i in range(I))
                m_glob_ad.optimize()
                # extract assignment
                n_list_temp = []
                for (scenario, leaf), v in xi.items():
                    if v.x >= 0.1:
                        n_list_temp.append(leaf - 3)
                n_list.append(n_list_temp)

                if round(m_glob_mp.ObjVal, 2) == round(m_glob_ad.ObjVal, 2):
                    # save sols
                    tree.s1, tree.s2, tree.s3, tree.s4 = y_en[0], y_en[1], y_en[2], y_en[3]
                    tree.rob_obj = m_glob_ad.ObjVal
                    if m_glob_ad.ObjVal < best_obj:
                        best_obj = m_glob_ad.ObjVal
                        best = tree
                    convergence = True
                    if time.time() - st > time_limit:
                        stop = True
            if time.time() - st > time_limit:
                stop = True
        if stop:
            pass
            break

    # calc nominal objective and save tree
    data.tree_last = copy.deepcopy(best)
    nom_obj = evaluate2(data, "mp", False)
    data.iteration = 1
    data.reset_obs()
    data = adversary2(data)
    data.tree_last.nom_obj = nom_obj
    data.nom_objs.append(nom_obj)

    # if the alternating heuristic is used, create file for a warm start
    if data.method == "a":
        true_vars = [(0, best.f1, best.t1), (1, best.f2, best.t2), (2, best.f3, best.t3)]
        content = []
        for q in [0, 1, 2]:
            for f in range(len(costs[0])):
                for theta in Theta[f]:
                    if (q, f, theta) in true_vars:
                        content.append("b[" + str(q) + "," + str(f) + "," + str(theta) + "] 1\n")
                    else:
                        content.append("b[" + str(q) + "," + str(f) + "," + str(theta) + "] 0\n")
        with open("last_sol_" + str(data.warm_start_id) + ".sol", 'w') as file:
            file.writelines(content)

    if data.method == "a":
        data.start_for_alternating_heu = data.tree_last
    data.rtime = time.time() - st
    data.time_left -= time.time() - st
    return data


def heu_fix_sols(data, no_of_trys=10**8, opt_time_limit=None):
    """
    Return a set of decision trees which are generated by applying the heuristic where solutions are sampled and the
    tree structures are optimized.

    Parameter:
        - data (Data): Object containing the instance
        - no_of_trys (int): Number of solution sets to sample
        - opt_time_limit (int): Use if a different time limit should be used

    Returns:
        - data (Data) - Updated object containing the instance
    """
    st = time.time()
    if opt_time_limit is None:
        time_limit = max(10, data.time_left)
    else:
        time_limit = max(10, opt_time_limit)
    graph, source, sink, costs = data.graph, data.s, data.t, data.costs
    dict_edges = data.dict_e
    edges = graph.edges
    I = len(costs)

    # generate solution candidates
    if not data.method == "a":
        all_sols = []
        for i in range(I):
            # solve every scenario to optimality
            m_opt = Model(env=env)
            m_opt.Params.OutputFlag = silence_all
            if only_1_core:
                m_opt.Params.Threads = 1
            y = m_opt.addVars(range(len(edges)), vtype=GRB.BINARY, name="y")

            m_opt.setObjective(quicksum(costs[i, dict_edges[e]] * y[dict_edges[e]] for e in edges))
            m_opt.addConstr((quicksum(y[dict_edges[(source, j)]] for j in graph.neighbors(source)) == 1), name="source")
            m_opt.addConstrs((quicksum(y[dict_edges[e]] for e in graph.in_edges(nbunch=i)) - quicksum(y[dict_edges[(i, j)]]
                              for j in graph.neighbors(i)) == 0 for i in graph.nodes if i not in {source, sink}), name="flow")
            m_opt.optimize()

            sol = []
            for v in m_opt.getVars():
                if v.VarName.startswith("y"):
                    sol.append(round(v.x))
            all_sols.append(sol)
        # only consider unique solutions
        all_sols = [list(t) for t in set(tuple(l) for l in all_sols)]

    best = None
    best_rob_obj = 1000000000

    if data.method != "a":
        all_comb = list(itertools.product(all_sols, repeat=4))
        all_comb = [list(combo) for combo in all_comb]
        random.shuffle(all_comb)
        no_of_trys = len(all_comb)

    for ind_run in range(no_of_trys):
        if data.method == "a":
            data.start_sols = [data.start_for_alternating_heu.s1, data.start_for_alternating_heu.s2,
                               data.start_for_alternating_heu.s3, data.start_for_alternating_heu.s4]

        else:
            data.start_sols = all_comb[ind_run]

        data_temp = copy.deepcopy(data)

        data_temp = iterative(data_temp)
        best_rob_after_it = min(data_temp.rob_objs)
        if best_rob_after_it < best_rob_obj:
            best_rob_obj = copy.deepcopy(best_rob_after_it)
            best = copy.deepcopy(data_temp)
        if time.time() - st >= time_limit:
            break

    if data.method == "a":
        best.start_for_alternating_heu = best.tree_rob
    data.rtime = time.time() - st
    data.time_left -= time.time() - st
    return best


def heu_alternate(data):
    best = None
    best_obj = 10**8
    st = time.time()
    time_limit = data.t_lim
    counter_ext = 0
    for _ in range(10**8):
        counter_ext += 1
        convergence = False
        copy_data = copy.deepcopy(data)
        copy_data.time_left = time_limit - (time.time() - st)
        while time.time() - st <= time_limit and not convergence:
            copy_data.counter_int_iteration += 1
            copy_data = heu_fix_tree(copy_data, 1)
            copy_data = heu_fix_sols(copy_data, 1)
            convergence = copy_data.check_convergence_alternating()
        data.rtime = time.time() - st
        best_rob_obj_run = min(copy_data.rob_objs)
        if best_rob_obj_run < best_obj:
            copy_data.counter_ext_iteration = counter_ext
            best_obj = copy.deepcopy(best_rob_obj_run)
            best = copy.deepcopy(copy_data)
        if time.time() - st > time_limit:
            break
    data.time_left -= time.time() - st
    return best


def one_for_all(data):
    st = time.time()

    # params
    graph, source, sink, costs = data.graph, data.s, data.t, data.costs
    dict_edges = data.dict_e
    F = range(len(graph.edges))  # set of features (one feature for every edge)
    I = range(len(costs))  # set of scenarios (datapoints)

    # optimization model
    m = Model(name="one", env=env)
    m.Params.OutputFlag = silence_all
    if only_1_core:
        m.Params.Threads = 1
    m.setParam('TimeLimit', data.t_lim)
    x = m.addVars(F, vtype=GRB.BINARY, name="x")

    # objective
    m.setObjective(quicksum(costs[s_ind, e_ind] * x[e_ind] for e_ind in F for s_ind in I))
    # constraints
    m.addConstr((quicksum(x[dict_edges[(source, j)]] for j in graph.neighbors(source)) == 1),
                name="sp_source")
    m.addConstrs((quicksum(x[dict_edges[e]] for e in graph.in_edges(nbunch=i)) - quicksum(x[dict_edges[(i, j)]]
                  for j in graph.neighbors(i)) == 0 for i in graph.nodes if i not in {source, sink}), name="sp_flow")
    # solve and save objective + solution
    m.optimize()
    sol = []
    for i, v in x.items():
        sol.append(v.x)
    data.one_sol = sol
    data.best_rob_obj = m.ObjVal

    data.rtime = time.time() - st
    return data


def all_methods(no_scens, grid_size, np_s, tl1, tl2, gamma_p):
    # one solution
    data_1 = make_data(no_scens, grid_size, np_s, "g", "one", tl1, tl2, gamma_p)
    res_1 = one_for_all(data_1)

    # GLOBAL
    # optimal approach
    data_g_o = make_data(no_scens, grid_size, np_s, "g", "o", tl1, tl2, gamma_p)
    res_g_o = iterative(data_g_o)
    # heuristic fix tree
    data_g_t = make_data(no_scens, grid_size, np_s, "g", "t", tl1, tl2, gamma_p)
    res_g_t = heu_fix_tree(data_g_t)
    # heuristic fix solutions
    data_g_s = make_data(no_scens, grid_size, np_s, "g", "s", tl1, tl2, gamma_p)
    res_g_s = heu_fix_sols(data_g_s)
    # alternating heuristic
    data_g_a = make_data(no_scens, grid_size, np_s, "g", "a", tl1, tl2, gamma_p)
    res_g_a = heu_alternate(data_g_a)

    # LOCAL
    # optimal approach
    data_l_o = make_data(no_scens, grid_size, np_s, "l", "o", tl1, tl2, gamma_p)
    res_l_o = iterative(data_l_o)
    # heuristic fix tree
    data_l_t = make_data(no_scens, grid_size, np_s, "l", "t", tl1, tl2, gamma_p)
    res_l_t = heu_fix_tree(data_l_t)
    # # heuristic fix solutions
    data_l_s = make_data(no_scens, grid_size, np_s, "l", "s", tl1, tl2, gamma_p)
    res_l_s = heu_fix_sols(data_l_s)
    # # alternating heuristic
    data_l_a = make_data(no_scens, grid_size, np_s, "l", "a", tl1, tl2, gamma_p)
    res_l_a = heu_alternate(data_l_a)
    return [res_1, res_g_o, res_g_t, res_g_s, res_g_a,
            res_l_o, res_l_t, res_l_s, res_l_a]


def only_ftree(no_scens, grid_size, np_s, tl1, tl2, g_glob):
    data_l_f = make_data(no_scens, grid_size, np_s, "l", "t", tl1, tl2, g_glob)
    res_l_f = heu_fix_tree(data_l_f)
    data_g_f = make_data(no_scens, grid_size, np_s, "g", "t", tl1, tl2, g_glob)
    res_g_f = heu_fix_tree(data_g_f)
    return [res_l_f, res_g_f]


def nom_and_one(no_scens, grid_size, np_s, tl1, tl2, g_glob):
    # one solution for all scenarios
    data_1 = make_data(no_scens, grid_size, np_s, "g", "one", tl1, tl2, g_glob)
    res_1 = one_for_all(data_1)
    # nom
    data_nom = make_data(no_scens, grid_size, np_s, "l", "nom", tl1, tl2, g_glob)
    data_nom = master_problem2(data_nom)
    return [res_1, data_nom]


def run_exp(no_scens, grid_size, methods, time_limit, gamma, run_id):
    seed = list_np_seeds[run_id]
    # execute scenario generation + all heuristics except one solution
    if methods == "all":
        results = all_methods(no_scens, grid_size, seed, time_limit, time_limit, gamma)
    # execute both benchmark approaches
    elif methods == "nom_and_one":
        results = nom_and_one(no_scens, grid_size, seed, time_limit, time_limit, gamma)
    # execute only the fix tree heuristic
    elif methods == "ftree":
        results = only_ftree(no_scens, grid_size, seed, time_limit, time_limit, gamma)
    else:
        results = None
    time.sleep(2)
    # store the data in a pickle file
    with open("data-" + str(no_scens) + "-" + str(grid_size) + "-" + str(methods) + "-" + str(time_limit) + "-"
              + str(gamma) + "-" + str(run_id) + "-" + (
              datetime.now().isoformat()) + ".pickle", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pass


param_scens = int(sys.argv[1])
param_grid = int(sys.argv[2])
param_methods = sys.argv[3]
param_time = int(sys.argv[4])
param_gamma = float(sys.argv[5])
param_id = int(sys.argv[6])

run_exp(param_scens, param_grid, param_methods, param_time, param_gamma, param_id)
