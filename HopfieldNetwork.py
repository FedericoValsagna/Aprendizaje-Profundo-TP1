from time import time

import numpy as np

CUTTING_CONDITION = 2000


class HopfieldNetwork:
    def __init__(self, xs: list, verbose=False):
        self.Xs = xs
        self.N = len(xs)
        self.l = len(xs[0])
        self.W = np.zeros((self.l, self.l))
        self.verbose = verbose

    def train(self):
        if self.verbose:
            print("Training Hopfield Network...")
            time_start = time()

        X = np.array(self.Xs)
        self.W = (X.T @ X) / self.l
        np.fill_diagonal(self.W, 0)

        if self.verbose:
            time_end = time()
            print("Training complete.")
            print(f"Training time: {time_end - time_start} seconds")

    def recall(self, x):
        x = np.array(x, dtype=float)
        current_fixed = 0
        while True:
            index = np.random.randint(0, self.l)
            fixed = self.update_cell(x, index)
            if fixed:
                current_fixed += 1
                if current_fixed > CUTTING_CONDITION:
                    break
                continue
            else:
                current_fixed = 0
        return x

    def update_cell(self, x, index):
        cell_before = x[index]
        x[index] = self.sign(self.W[index] @ x)
        return x[index] == cell_before

    def sign(self, number):
        if number >= 0:
            return 1
        else:
            return -1

    def recall_con_energia(
        self, W: np.ndarray, x, cutting_condition: int = 2000, max_iter: int = 200000
    ):
        x = np.array(x, dtype=float).copy()
        energias = [self.energia(W, x)]
        current_fixed = 0

        while True:
            index = np.random.randint(0, len(x))
            before = x[index]
            activacion = W[index] @ x
            x[index] = 1 if activacion >= 0 else -1
            energias.append(self.energia(W, x))
            if x[index] == before:
                current_fixed += 1
                if current_fixed > cutting_condition:
                    break
            else:
                current_fixed = 0

        return x, energias

    def energia(self, W: np.ndarray, x: np.ndarray) -> float:
        return -0.5 * x @ W @ x
