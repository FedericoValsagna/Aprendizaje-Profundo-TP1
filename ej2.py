import numpy as np
from HopfieldNetwork import HopfieldNetwork
from ej1 import add_noise, recall_pattern

N = 1000
PMAXOVERN = 0.185
PMAX = 150
NOISE_PERCENTAGE = 3

def generar_patrones(largo ,n):
    return [generar_patron(largo) for _ in range(n)]

def generar_patron(largo):
    return np.random.choice([1, -1], size=largo)

def ej2():
    xs = generar_patrones(N, PMAX)
    hopfield_network = HopfieldNetwork(xs)
    hopfield_network.train()
    recall_pattern(xs,NOISE_PERCENTAGE, hopfield_network)
