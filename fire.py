import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from multiprocessing import Pool

# Constants
EMPTY = 0
TREE = 1
BURNING = 2

# Parameters as doc samples
probTree = 0.8
probBurning = 0.01
probImmune = 0.3
probLightning = 0.001

# Helper functions
def initialize_grid(n):
    grid = np.random.choice([EMPTY, TREE], size=(n, n), p=[1 - probTree, probTree])
    grid[grid == TREE] = np.random.choice([TREE, BURNING], size=grid[grid == TREE].shape, p=[1 - probBurning, probBurning])
    return grid

def spread(grid, i, j):
    neighbors = get_moore_neighbors(grid, i, j)
    if grid[i, j] == EMPTY:
        return EMPTY
    elif grid[i, j] == TREE:
        if np.random.rand() < probImmune:
            return TREE
        elif BURNING in neighbors or np.random.rand() < probLightning:
            return BURNING
        else:
            return TREE
    else:
        return EMPTY

def get_moore_neighbors(grid, i, j):
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == dj == 0:
                continue
            ni, nj = (i + di) % grid.shape[0], (j + dj) % grid.shape[1]
            neighbors.append(grid[ni, nj])
    return neighbors

def apply_spread_parallel(grid):
    n = grid.shape[0]
    extended_grid = np.pad(grid, 1, mode='wrap')
    with Pool() as pool:
        result = pool.starmap(spread, [(extended_grid, i, j) for i in range(1, n + 1) for j in range(1, n + 1)])
    result = np.array(result).reshape(n, n)
    return result

def visualize(grid):
    cmap = colors.ListedColormap(['white', 'green', 'red'])
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap=cmap)
    plt.show()

# Main function
def main():
    n = 100
    grid = initialize_grid(n)

    # Sequential implementation
    start_time = time.time()
    for i in range(10):
        grid = np.array([[spread(grid, i, j) for j in range(n)] for i in range(n)])
    seq_time = time.time() - start_time
    print(f"Sequential implementation time: {seq_time:.6f} seconds")
    visualize(grid)

    # Parallel implementation
    grid = initialize_grid(n)
    start_time = time.time()
    for i in range(10):
        grid = apply_spread_parallel(grid)
    parallel_time = time.time() - start_time
    print(f"Parallel implementation time: {parallel_time:.6f} seconds")
    visualize(grid)

if __name__ == "__main__":
    main()