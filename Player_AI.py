"""
Author Evimar Principal
The class Player_AI plays 2048, choose the best move in 0.1 seconds
based on Adversarial Games, using the algorithm
MiniMax search with alpha beta pruning
jue, mar 2 2017 15:51:26
"""

from random import randint
from BaseAI_3 import BaseAI
import time
import numpy as np
import csv


class PlayerAI(BaseAI):
    log2 = {0: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8, 512: 9, 1024: 10, 2048: 11, 4096: 12, 8192: 13,
            16384: 14}
    INFINITE = 1000000
    maxTime = 0.1
    prueba = 11
    # w = np.array([ -55.17806677,  178.8909986,   114.45168342]) #100 114 90 116
    w = np.array([2, 1, -0.08])  # free tiles, monocity, smooth

    def listTuple(l):
        """Transform a list of lists in a tuple"""
        r = tuple()
        for v in l:
            r += tuple(v)
        return r

    def listArray(l):
        """Transform a list of lists in a numpy array"""
        return np.array(l)

    def getMove(self, grid):
        moves = grid.getAvailableMoves()
        if not moves: return None

        self.n = "X_" + str(PlayerAI.prueba) + ".csv"
        # Dict of Grid -> eval function. Grid is a tuple form
        self.dicEvalGrid = dict()
        self.Monotonic, self.Politonic = set(), set()

        self.eval_utility(grid, True)  # Save the heuristic train X

        self.pmin = 0
        self.pmax = 0
        self.start = time.clock()
        maxDeep = 1

        while time.clock() - self.start < PlayerAI.maxTime:
            move = self.best_move(grid, maxDeep)
            if move is not None:
                bestMove = move
            else:
                break
            maxDeep += 1

        return bestMove

    def generateChildren(self, grid):
        """Return a list of tuples (grid, move)"""
        moves = grid.getAvailableMoves()
        children = []
        for move in moves:
            gridCopy = grid.clone()
            gridCopy.move(move)
            children.append((gridCopy, move, self.eval_utility(gridCopy)))
        children = sorted(children, key=lambda x: x[2], reverse=True)

        return children

    def generatePCChildren(self, grid, valor):
        """Return a list of tuples (grid, move)"""
        cells = grid.getAvailableCells()
        children = []
        for cell in cells:
            gridCopy = grid.clone()
            gridCopy.setCellValue(cell, valor)
            children.append((gridCopy, cell, self.eval_utility(gridCopy)))
        children = sorted(children, key=lambda x: x[2], reverse=True)

        return children

    def best_move(self, grid, maxDeep):
        move, utility = self.maxi(grid, 0, -PlayerAI.INFINITE, PlayerAI.INFINITE, maxDeep)
        return move

    def maxi(self, grid, deep, alpha, beta, maxDeep):
        if time.clock() - self.start > PlayerAI.maxTime:
            return None, -PlayerAI.INFINITE
        if deep == maxDeep:
            return None, self.eval_utility(grid)

        bestMove, maxUtility = None, -PlayerAI.INFINITE

        children = self.generateChildren(grid)
        self.pmax += len(children)

        for node in children:
            grid = node[0]
            move = node[1]
            self.pmax -= 1
            utility = self.expectingMin(grid, deep + 1, alpha, beta, maxDeep)
            if utility == PlayerAI.INFINITE:
                return None, -PlayerAI.INFINITE

            if utility > maxUtility:
                bestMove, maxUtility = move, utility

            if maxUtility >= beta:
                break  # the prune, cut the search

            if maxUtility > alpha:
                alpha = maxUtility
        return (bestMove, maxUtility)

    def expectingMin(self, grid, deep, alpha, beta, maxDeep):
        if time.clock() - self.start > PlayerAI.maxTime:
            return PlayerAI.INFINITE

        v2 = self.mini(grid, deep, alpha, beta, maxDeep, 2) * 0.9
        if v2 == PlayerAI.INFINITE:
            return PlayerAI.INFINITE
        v4 = self.mini(grid, deep, alpha, beta, maxDeep, 4) * 0.1
        if v4 == PlayerAI.INFINITE:
            return PlayerAI.INFINITE
        return v2 + v4

    def mini(self, grid, deep, alpha, beta, maxDeep, valor):
        if time.clock() - self.start > PlayerAI.maxTime:
            return PlayerAI.INFINITE

        if deep == maxDeep:
            return self.eval_utility(grid)

        worstMove, minUtility = None, PlayerAI.INFINITE

        children = self.generatePCChildren(grid, valor)

        self.pmin += len(children)
        for node in children:
            grid = node[0]
            move = node[1]
            self.pmin -= 1
            child, utility = self.maxi(grid, deep + 1, alpha, beta, maxDeep)
            if utility == -PlayerAI.INFINITE:
                return PlayerAI.INFINITE

            if utility < minUtility:
                worstMove, minUtility = move, utility

            if minUtility <= alpha:
                break  # the prune, cut the search

            if minUtility < beta:
                beta = minUtility

        return minUtility

    def monotonicity(self, l):
        """
        Return a bool value true is the array has monotonicity
        """
        if l in self.Monotonic:
            return True
        if l in self.Politonic:
            return False
        if all(x >= y for x, y in zip(l, l[1:])):
            self.Monotonic.add(l)
            return True
        else:

            if all(x <= y for x, y in zip(l, l[1:])):
                self.Monotonic.add(l)
                return True
        self.Politonic.add(l)
        return False

    def monoValue(self, a):
        """
        Return the sum of rows and cols that are monotonicity
        Arg
        array a numpy array integer size 4x4
        """
        valor = 0
        # generate the column list

        f = np.fliplr(a)

        for i in range(4):
            if self.monotonicity(tuple(a[i])): valor += 1  # row i
            if self.monotonicity(tuple(a[:, i])): valor += 1  # col i

        # Diagonals
        for i in range(-1, 2):
            if self.monotonicity(tuple(a.diagonal(i))): valor += 1  # Diagonal
            if self.monotonicity(tuple(f.diagonal(i))): valor += 1  # Diagonal Inversa

        return valor

    def smoothVal(self, v1, v2):
        """
        It must delete empty cells, because the big tiles will be left in the middle
        and tiny-value tiles could be in between randomly
        """
        return abs(v1 - v2)

    def smooth(a):
        valor = 0

        for i in range(4):
            for j in range(4):
                fijo = PlayerAI.log2[a[i, j]]
                if i > 0:
                    valor += PlayerAI.smoothVal(fijo, PlayerAI.log2[a[i - 1, j]])
                if i < 3:
                    valor += PlayerAI.smoothVal(fijo, PlayerAI.log2[a[i + 1, j]])
                if j > 0:
                    valor += PlayerAI.smoothVal(fijo, PlayerAI.log2[a[i, j - 1]])
                if j < 3:
                    valor += PlayerAI.smoothVal(fijo, PlayerAI.log2[a[i, j + 1]])
        return valor

    def eval_utility(self, grid, grabar=False):
        # transform the map into a tuple
        tuplaGrid = PlayerAI.listTuple(grid.map)

        if tuplaGrid in self.dicEvalGrid:
            return self.dicEvalGrid[tuplaGrid]

        arreglo = np.array(grid.map)
        # Qty of empty Tiles
        numEmptyTiles = len(grid.getAvailableCells())
        # Monotonicity
        numMonotocity = self.monoValue(arreglo)
        # Smoothness
        numSmooth = PlayerAI.smooth(arreglo)

        linea = [numEmptyTiles, numMonotocity, numSmooth]
        x0 = np.array(linea)

        utility = x0.dot(PlayerAI.w)

        if grabar:
            with open(self.n, 'a', newline='') as fp:
                a = csv.writer(fp, delimiter=',')
                a.writerow(linea)

        self.dicEvalGrid[tuplaGrid] = utility

        return utility

