from typing import List

import numpy as np


class Optimizer:
    def __init__(self, id_: int, solutions, criterion, wages: List[float] = None):
        self.id_ = id_
        self.neighbours = {}
        self.solutions = solutions
        self.decisions = {}
        self.feedback = {}
        self.criterion = criterion
        self.wages = wages

    def add_neighbour(self, neighbour: 'Optimizer', connections: np.ndarray, feedback=None):
        self.neighbours[neighbour.id_] = connections
        if feedback:
            self.feedback[neighbour.id_] = feedback

    def _get_non_dominated(self):
        non_dominated = {}
        for neighbour, solution in self.neighbours.items():
            ni, no = solution.shape
            idx_s = np.zeros(ni)
            res_temp = np.zeros((ni - 1, 3))
            counter = -1
            if ni == 1:
                idx_s = 1
                break

            for j in range(ni):
                res_tmp = np.zeros(solution.shape)
                this_solution = solution[j, :]
                for k in range(no):
                    res_tmp[:, k] = this_solution[k] - solution[:, k]
                res_tmp = np.delete(res_tmp, j, 0)
                res_tmp = np.sign(res_tmp)
                res_temp[:, 0] = sum(res_tmp.T < 0)
                res_temp[:, 1] = sum(res_tmp.T == 0)
                res_temp[:, 2] = sum(res_tmp.T > 0)
                if np.min(res_temp[:, 0]) > 0:
                    counter = counter + 1
                    idx_s[counter] = j + 1
                else:
                    res_temp_z = res_temp[res_temp[:, 1] == 0, :]
                    idx_z = np.argwhere(res_temp_z[:, 2] == no)
                    if len(idx_z) == res_temp_z.shape[1]:
                        counter = counter + 1
                        idx_s[counter] = j
            idx_s = idx_s[idx_s != 0]
            idx_s = [int(idx - 1) for idx in idx_s]
            non_dominated[neighbour] = idx_s
        return non_dominated

    def remove_dominated(self):
        non_dominated = self._get_non_dominated()
        result = {}
        for neighbour, solution in self.neighbours.items():
            result[neighbour] = []
            for i in range(solution.shape[0]):
                temp = []
                if i in non_dominated[neighbour]:
                    for j in range(solution.shape[1]):
                        if neighbour in list(self.feedback.keys()):
                            if j in self.feedback[neighbour]:
                                if solution[i][j] == 1:
                                    temp.append((i, j))
                        else:
                            if solution[i][j] == 1:
                                temp.append((i, j))
                    result[neighbour].extend(temp)
        self.decisions = result

