import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=-1))

def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Direction matrix: 0=diagonal, 1=up, 2=left
    direction_matrix = np.zeros((n + 1, m + 1), dtype=int)
    
    # Convert lists to numpy arrays for vectorized operations
    s = np.array(s)
    t = np.array(t)
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs = [dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]]
            min_cost = min(costs)
            dtw_matrix[i, j] = np.linalg.norm(s[i-1] - t[j-1]) + min_cost
            
            # Record the direction
            direction_matrix[i, j] = np.argmin(costs)
    
    # Backtrack to find path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        if direction_matrix[i, j] == 0:
            i, j = i-1, j-1
        elif direction_matrix[i, j] == 1:
            i = i-1
        else:
            j = j-1
    
    return  dtw_matrix[n, m], path[::-1]

if __name__ == "__main__":
    t = np.linspace(0,1, 20)
    x = np.stack([t, t**2], axis=-1)
    y = np.stack([t, t**2 + 0.1 + np.random.rand(*t.shape)*0.1 ], axis=-1)
    distance,path = dtw_matrix, cost = dtw(x, y)
    print(distance)
    print(path)

    fig = plt.figure()
    ax = plt.scatter(x[:,0], x[:,1], c='r')
    ax = plt.scatter(y[:,0], y[:,1], c='b')
    for i,j in path:
        plt.plot([x[i,0], y[j,0]], [x[i,1], y[j,1]], 'k-')
    fig.savefig('dtw.png')