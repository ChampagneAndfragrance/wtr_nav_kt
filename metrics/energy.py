import numpy as np

def compute_energy(positions, m=1.0,I=1.0, dt = 1.0, ):


    velocities = np.diff(positions, axis=0) / dt

    angles = np.arctan2(velocities[:, 1], velocities[:, 0])
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    angular_velocities = np.diff(angles) / dt

    N = velocities.shape[0]

    trans_ke = 0.5 * m * np.sum(velocities**2)/N
    rotational_ke = 0.5 * I * np.sum(angular_velocities**2)/N

    return trans_ke + rotational_ke

if __name__ == '__main__':
    positions = np.array([[0, 0], [1, 1], [2, 0], [3, -1], [4, 0]])
    average_total_ke = compute_energy(positions)
    print(average_total_ke)