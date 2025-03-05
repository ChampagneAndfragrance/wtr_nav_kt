import numpy as np


def compute_jerk(positions, dt=1.0):

    velocities = np.gradient(positions, dt, axis=0)
    accelerations = np.gradient(velocities, dt, axis=0)
    
    # Compute third derivative (jerk)
    jerks = np.gradient(accelerations, dt, axis=0)
    
    return (jerks**2).sum(axis=1).mean()

if __name__ == '__main__':
    t = np.linspace(0, 10, 100)
    positions = np.stack([t, np.sin(t)], axis=-1)

    jerk = compute_jerk(positions)
    print('jerk from smooth data', jerk)

    positions = positions + np.random.rand(*positions.shape) *2.0

    jerk = compute_jerk(positions)
    print('jerk from noisy data', jerk)