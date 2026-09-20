from time import time

import numpy as np

CUTTING_CONDITION = 1200


class HopfieldNetwork:
    def __init__(self, xs):
        self.Xs = xs
        self.N = len(xs)
        self.l = len(xs[0])
        self.W = np.zeros((self.l, self.l))
        # print("W:", self.W)
        # self.train()
        
    def train(self):
        
        print("Training Hopfield Network...")
        time_start = time()
        for i in range(self.l):
            for j in range(self.l):
                if i != j:
                    self.W[i][j] = sum([x[i] * x[j] for x in self.Xs])
        
        time_end = time()
        print("Training complete.")
        print(f"Training time: {time_end - time_start} seconds")   
        # print("W:", self.W)
        
    
    def recall(self, x):

        # print("Recalling image...")
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
        x[index] = self.sign(sum(self.W[index][j] * x[j] for j in range(self.l) if j != index))
        return x[index] == cell_before
    
    
    def sign(self, number):
        if number >= 0:
            return 1
        else:
            return -1