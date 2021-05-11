from typing import List

from src.optimizer import Optimizer

import numpy as np


class Net:
    def __init__(self, optimizers: List[Optimizer], first_point_id: List[int], last_point_id: List[int]):
        self.optimizers = optimizers
        self.first_point_id = first_point_id
        self.last_point_id = last_point_id
        self.check_optimizers()
        self.result, self.maximum = self.get_best_solution()

    def check_optimizers(self):
        for optimizer in self.optimizers:
            optimizer.remove_dominated()

    def get_optimizer_by_id(self, id_) -> Optimizer:
        for optimizer in self.optimizers:
            if optimizer.id_ == id_:
                return optimizer

    def _begin_point(self):
        result = []
        for first_point_id in self.first_point_id:
            first_point = self.get_optimizer_by_id(first_point_id)
            for neighbour_id, decisions in first_point.decisions.items():
                for decision in decisions:
                    one_path = [(first_point.id_, decision[0])]
                    self._find_path(decision, neighbour_id, one_path, result)

        return result

    def _find_path(self, prev_decision, neighbour_id, one_path, result):
        one_path.append((neighbour_id, prev_decision[1]))
        if neighbour_id not in self.last_point_id:
            neighbour = self.get_optimizer_by_id(neighbour_id)
            for next_neighbour_id, decisions in neighbour.decisions.items():
                for decision in decisions:
                    if prev_decision[1] == decision[0]:
                        next_decision = decision
                        one_path = self._find_path(next_decision, next_neighbour_id, one_path, result)
            return one_path[:-1]

        else:
            result.append(one_path)
            return one_path[:-1]

    def get_best_solution(self):
        results = self._begin_point()
        best_paths = []
        maximum = 0
        for ind, result in enumerate(results):
            for point in result[:-1]:
                optimizer = self.get_optimizer_by_id(point[0])
                criterion = optimizer.criterion[point[1]]
                wages = optimizer.wages
                sum_result = sum(np.array(criterion)*np.array(wages))
                if sum_result > maximum:
                    best_paths.clear()
                    best_paths.append(result)
                    maximum = sum_result
                if sum_result == maximum:
                    best_paths.append(result)

        return best_paths, maximum
