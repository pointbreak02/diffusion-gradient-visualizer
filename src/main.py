import numpy as np
import matplotlib.pyplot as plt

def initialize_grid(size=50, center_value=100):
    grid = np.zeros((size, size))
    center = size // 2
    grid[center, center] = center_value
    return grid

def diffuse(grid, steps=100, diffusion_rate=0.1):
    for _ in range(steps):
        new_grid = grid.copy()
        for i in range(1, grid.shape[0]-1):
            for j in range(1, grid.shape[1]-1):
                new_grid[i, j] = grid[i, j] + diffusion_rate * (
                    grid[i+1, j] + grid[i-1, j] + grid[i, j+1] + grid[i, j-1] - 4 * grid[i, j]
                )
        grid = new_grid
    return grid

def plot_grid(grid):
    plt.imshow(grid, cmap='viridis')
    plt.colorbar(label="Concentration")
    plt.title("2D Diffusion Gradient")
    plt.show()

if __name__ == "__main__":
    grid = initialize_grid()
    final_grid = diffuse(grid)
    plot_grid(final_grid)
