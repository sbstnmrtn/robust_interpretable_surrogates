import numpy as np
import copy
import time

epsilon_glob = 0.001


class Tree:
    def __init__(self, depth):
        """Decision tree object."""
        # branching nodes
        self.depth = depth
        # thresholds
        self.t1 = None
        self.t2 = None
        self.t3 = None
        # features
        self.f1 = None
        self.f2 = None
        self.f3 = None
        # solutions
        self.s1 = None
        self.s2 = None
        self.s3 = None
        self.s4 = None
        # quality measures
        self.rob_obj = None
        self.nom_obj = None
        # helper for postprocessing the adversary problem
        self.true_assignment = None
        self.scenarios_post = None

    def get_sol(self, datapoint):
        """returns sol for a given data point"""
        if self.depth == 1:
            if datapoint[self.f1] <= self.t1:
                return self.s1
            else:
                return self.s2
        elif self.depth == 2:
            if datapoint[self.f1] <= self.t1:
                split1 = True
            else:
                split1 = False
            if datapoint[self.f2] <= self.t2:
                split2 = True
            else:
                split2 = False
            if datapoint[self.f3] <= self.t3:
                split3 = True
            else:
                split3 = False
            if split1 and split2:
                return self.s1
            elif split1 and not split2:
                return self.s2
            elif not split1 and split3:
                return self.s3
            elif not split1 and not split3:
                return self.s4
        else:
            print("error")
            return None

    def get_sols(self, datapoints):
        """returns sols for given data points"""
        sols = []
        for datapoint in datapoints:
            sols.append(self.get_sol(datapoint))
        return sols

    def reset_nom(self):
        self.nom_obj = None

    def calc_scheduling_objective(self, cost_vector, solution):
        """calculate the objective value for a given combination of cost vector and solution"""
        (no_machines, no_jobs) = solution.shape
        machines, jobs = range(no_machines), range(no_jobs)
        obj_machines = []
        for machine in machines:
            obj_machines.append(sum(solution[machine, job] * cost_vector[job] for job in jobs))
        objective = max(obj_machines)
        return objective

    def pre_adv(self, costs, budget, scheduling=False):
        """preprocessing for adversary problem"""
        scenarios = range(len(costs))
        edges = range(len(costs[0]))
        d = np.zeros((len(costs), self.depth * 2))  # costs of combination of solution and scenario
        p = np.zeros((len(costs), self.depth * 2))  # price of shifting a scenario into a specific leaf
        obs_list = [copy.deepcopy(costs), copy.deepcopy(costs), copy.deepcopy(costs), copy.deepcopy(costs)]
        if budget == 0.0:
            upper_bound_p = 1.0
        else:
            upper_bound_p = budget * 2
        self.true_assignment = []

        if self.t1 is None or self.t2 is None or self.t3 is None:
            print("e")

        temp_array = np.zeros(6)
        for scenario in scenarios:
            # calc delta
            if not scheduling:
                for edge in edges:
                    d[scenario][0] = round(d[scenario][0] + self.s1[edge] * costs[scenario][edge], 5)
                    d[scenario][1] = round(d[scenario][1] + self.s2[edge] * costs[scenario][edge], 5)
                    d[scenario][2] = round(d[scenario][2] + self.s3[edge] * costs[scenario][edge], 5)
                    d[scenario][3] = round(d[scenario][3] + self.s4[edge] * costs[scenario][edge], 5)
            else:
                d[scenario][0] = self.calc_scheduling_objective(costs[scenario], self.s1)
                d[scenario][1] = self.calc_scheduling_objective(costs[scenario], self.s2)
                d[scenario][2] = self.calc_scheduling_objective(costs[scenario], self.s3)
                d[scenario][3] = self.calc_scheduling_objective(costs[scenario], self.s4)

            # calc p
            # check queries
            if costs[scenario][self.f1] <= self.t1:
                q1 = True
                temp_array[0] = 0
                temp_array[1] = round(self.t1 - costs[scenario][self.f1] + epsilon_glob, 5)
            else:
                q1 = False
                temp_array[0] = round(costs[scenario][self.f1] - self.t1 + epsilon_glob, 5)
                temp_array[1] = 0
            if costs[scenario][self.f2] <= self.t2:
                q2 = True
                temp_array[2] = 0
                temp_array[3] = round(self.t2 - costs[scenario][self.f2] + epsilon_glob, 5)
            else:
                q2 = False
                temp_array[2] = round(costs[scenario][self.f2] - self.t2 + epsilon_glob, 5)
                temp_array[3] = 0
            if costs[scenario][self.f3] <= self.t3:
                q3 = True
                temp_array[4] = 0
                temp_array[5] = round(self.t3 - costs[scenario][self.f3] + epsilon_glob, 5)
            else:
                q3 = False
                temp_array[4] = round(costs[scenario][self.f3] - self.t3 + epsilon_glob, 5)
                temp_array[5] = 0

            # calc p values
            if self.f1 == self.f2:
                p[scenario][0] = max(abs(temp_array[0]), abs(temp_array[2]))
                obs_list[0][scenario][self.f1] = round(obs_list[0][scenario][self.f1] - p[scenario][0], 5)
                if self.t2 > self.t1:
                    p[scenario][1] = round(upper_bound_p, 5)
                else:
                    if q1 and q2:
                        if self.t1 - 0.000001 <= self.t2 <= self.t1 + 0.000001:
                            p[scenario][1] = round(upper_bound_p, 5)
                        else:
                            p[scenario][1] = abs(temp_array[3])
                            obs_list[1][scenario][self.f1] = round(obs_list[1][scenario][self.f1] + temp_array[3], 5)
                    if q1 and not q2:
                        p[scenario][1] = 0
                    if not q1 and not q2:
                        if self.t1 - 0.000001 <= self.t2 <= self.t1 + 0.000001:
                            p[scenario][1] = round(upper_bound_p, 5)
                        else:
                            p[scenario][1] = abs(temp_array[0])
                            obs_list[1][scenario][self.f1] = round(obs_list[1][scenario][self.f1] - temp_array[0], 5)
            else:
                p[scenario][0] = round(abs(temp_array[0]) + abs(temp_array[2]), 5)
                p[scenario][1] = round(abs(temp_array[0]) + abs(temp_array[3]), 5)
                obs_list[0][scenario][self.f1] = round(obs_list[0][scenario][self.f1] - temp_array[0], 5)
                obs_list[0][scenario][self.f2] = round(obs_list[0][scenario][self.f2] - temp_array[2], 5)
                obs_list[1][scenario][self.f1] = round(obs_list[1][scenario][self.f1] - temp_array[0], 5)
                obs_list[1][scenario][self.f2] = round(obs_list[1][scenario][self.f2] + temp_array[3], 5)
            if self.f1 == self.f3:
                p[scenario][3] = max(abs(temp_array[1]), abs(temp_array[5]))
                obs_list[3][scenario][self.f1] = round(obs_list[3][scenario][self.f1] + p[scenario][3], 5)
                if self.t1 > self.t3:
                    p[scenario][2] = round(upper_bound_p, 5)
                else:
                    if not q1 and q3:
                        p[scenario][2] = 0
                    if q1 and q3:
                        if self.t1 - 0.000001 <= self.t3 <= self.t1 + 0.000001:
                            p[scenario][2] = round(upper_bound_p, 5)
                        else:
                            p[scenario][2] = abs(temp_array[1])
                            obs_list[2][scenario][self.f1] = round(obs_list[2][scenario][self.f1] + temp_array[1], 5)
                    if not q1 and not q3:
                        if self.t1 - 0.000001 <= self.t3 <= self.t1 + 0.000001:
                            p[scenario][2] = round(upper_bound_p, 5)
                        else:
                            p[scenario][2] = abs(temp_array[4])
                            obs_list[2][scenario][self.f1] = round(obs_list[2][scenario][self.f1] - temp_array[4], 5)
            else:
                p[scenario][2] = round(abs(temp_array[1]) + abs(temp_array[4]), 5)
                p[scenario][3] = round(abs(temp_array[1]) + abs(temp_array[5]), 5)
                obs_list[2][scenario][self.f1] = round(obs_list[2][scenario][self.f1] + temp_array[1], 5)
                obs_list[2][scenario][self.f3] = round(obs_list[2][scenario][self.f3] - temp_array[4], 5)
                obs_list[3][scenario][self.f1] = round(obs_list[3][scenario][self.f1] + temp_array[1], 5)
                obs_list[3][scenario][self.f3] = round(obs_list[3][scenario][self.f3] + temp_array[5], 5)

            # save true assignment
            if q1:
                if q2:
                    self.true_assignment.append(0)
                else:
                    self.true_assignment.append(1)
            else:
                if q3:
                    self.true_assignment.append(2)
                else:
                    self.true_assignment.append(3)

        self.scenarios_post = obs_list

        return d, p

    def post_adv(self, xi_vars, list_j, observations, iteration, dict_costs):
        """postprocessing for adversary problem"""
        for xi in xi_vars:
            scenario = xi[0]
            leaf = xi[1]
            if leaf == self.true_assignment[scenario]:
                list_j[scenario].append(iteration)
            else:
                # check if this scenario was already observed in the past
                if np.any(np.array(
                        [np.array_equal(row, self.scenarios_post[leaf][scenario, :]) for row in observations])):
                    ind = np.where(np.all(observations == self.scenarios_post[leaf][scenario, :], axis=1))
                    list_j[ind[0][0]].append(iteration)
                # else append entry
                else:
                    list_j.append([iteration])
                    observations = np.append(observations,
                                             self.scenarios_post[leaf][scenario, :].reshape(1, len(observations[0, :])),
                                             axis=0)
                    dict_costs.update({len(dict_costs): scenario})
        return list_j, observations, dict_costs

    def add_branching(self, values):
        """values: triple (node, feat, threshold)"""
        if values[0] == 0:
            self.f1 = values[1]
            self.t1 = values[2]
        elif values[0] == 1:
            self.f2 = values[1]
            self.t2 = values[2]
        elif values[0] == 2:
            self.f3 = values[1]
            self.t3 = values[2]

    def add_sol(self, index_leaf, sol_vector):
        """index_leaf: int, sol_vector: list"""
        if index_leaf == 3:
            self.s1 = sol_vector
        elif index_leaf == 4:
            self.s2 = sol_vector
        elif index_leaf == 5:
            self.s3 = sol_vector
        elif index_leaf == 6:
            self.s4 = sol_vector


class Data:
    def __init__(self, seed, instance_size, no_train, costs_tr, costs_te, mapping, budget, budget_type, param_budget,
                 graph, dict_e, dict_c, s, t, method, time_limit, time_limit2):
        """Contains Data for one instance and one method."""
        self.seed = seed
        self.instance_size = instance_size
        self.no_train = no_train
        self.costs = costs_tr
        self.costs_test = costs_te
        self.obs = copy.deepcopy(costs_tr)
        self.mapping = mapping
        self.start_mapping = copy.deepcopy(mapping)
        self.budget = budget
        self.budget_type = budget_type
        self.budget_param = param_budget
        self.graph = graph
        self.dict_e = dict_e
        self.dict_c = dict_c
        self.s = s
        self.t = t
        self.method = method

        # trees and their metrics
        self.tree_nom = None
        self.best_nom_obj = 10 ** 8
        self.nom_objs = []
        self.tree_rob = None
        self.best_rob_obj = 10 ** 8
        self.rob_objs = []
        self.tree_rob_theta = None
        self.best_rob_obj_theta = 10 ** 8
        self.tree_last = None
        self.one_sol = None

        # time related things
        self.time_mp = []
        self.time_opt_thrshlds = []
        self.time_ad = []
        self.rtime = 0
        self.warmstarttime = 0
        self.tl_all = time_limit
        self.tl_grb = time_limit2
        self.start_time = time.time()
        self.end_time = None
        self.time_vs_obj = []

        # misc
        self.iteration = 0
        self.gap = None
        self.start_sols = None
        self.start_for_alternating_heu = None
        self.last_assgn_sols = []
        self.org_dict_c = copy.deepcopy(dict_c)
        self.marker_opt_thrshld = False
        dict_methods = {"nom": 0, "one": 1, "o": 2, "s": 3, "t": 4, "a": 5}
        dict_budget_type = {"l": 0, "g": 1}
        if type(instance_size) == tuple:
            self.warm_start_id = int(str(seed) + str(instance_size[0]) + str(instance_size[1]) + str(no_train)
                                     + str(int(param_budget * 1000)) + str(time_limit) + str(dict_methods[method])
                                     + str(dict_budget_type[budget_type]))
        else:
            self.warm_start_id = int(str(seed)+str(instance_size)+str(no_train)+str(int(param_budget*1000))
                                     +str(time_limit)+str(dict_methods[method])+str(dict_budget_type[budget_type]))
        self.counter_int_iteration = 0
        self.counter_ext_iteration = 0
        self.list_time_obj = []

    def add_tree(self, tree):
        if self.iteration == 0:
            self.best_nom_obj = tree.nom_obj
            if self.method == "o":
                self.tree_nom = copy.deepcopy(tree)
        self.tree_last = copy.deepcopy(tree)
        self.nom_objs.append(tree.nom_obj)
        self.iteration += 1

    def override_last_tree(self, tree):
        self.tree_last = copy.deepcopy(tree)

    def add_scens(self, dict_c, mapping, obs, obj, runtime):
        self.dict_c = dict_c
        self.mapping = mapping
        self.obs = obs
        self.rob_objs.append(obj)
        self.time_ad.append(runtime)
        if obj < self.best_rob_obj:
            self.best_rob_obj = obj
            self.tree_rob = copy.deepcopy(self.tree_last)

    def check_convergence(self):
        gap = (min(self.rob_objs) - max(self.nom_objs)) / min(self.rob_objs)
        self.gap = gap
        if gap < 0.001:
            return True
        else:
            return False

    def check_convergence_alternating(self):
        abs_gap = (self.rob_objs[-1] - self.rob_objs[-2])
        if abs_gap < 0.001:
            return True
        else:
            return False

    def get_last_obs(self):
        list_obs_ind = []
        for ind, sublist in enumerate(self.mapping):
            if self.iteration in sublist:
                list_obs_ind.append(ind)
        return list_obs_ind

    def reset_obs(self):
        self.obs = copy.deepcopy(self.costs)
        self.iteration = 1
        self.mapping = copy.deepcopy(self.start_mapping)
        self.dict_c = copy.deepcopy(self.org_dict_c)

    def return_best_rob(self):
        if self.rob_objs:
            return min(min(self.rob_objs), self.best_rob_obj_theta)
        else:
            return self.best_rob_obj_theta

