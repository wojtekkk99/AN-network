import numpy as np

from src.net import Net
from data.data import U0, U1, U2, U3, U4, U5, U6

if __name__ == "__main__":
    network = Net([U0, U1, U2, U3, U4, U5, U6], first_point_id=[0], last_point_id=[5, 6])
    result_representation = ""
    for ind, result in enumerate(network.result):
        result_representation += "Rozwiązanie " + str(ind + 1) + ":\n"
        for point in result:
            result_representation += "\tOptymalizator " + str(point[0]) + " -> decyzja: " + str(point[1]) + "\n"

    result_representation += "\nSuma kryterium: " + str(network.maximum)

    print(result_representation)
