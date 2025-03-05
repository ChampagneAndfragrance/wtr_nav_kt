import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

def best_fit_transform(A, B):

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):


    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=1e-6):



    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i

def icp_error(x,y):
    if x.shape[1] == 2:
        x = np.hstack((x, np.zeros((x.shape[0],1))))
    if y.shape[1] == 2:
        y = np.hstack((y, np.zeros((y.shape[0],1))))
    T,_,_ = icp(x,y)
    C = np.dot(x, T[:3,:3].T) + T[:3,3]
    return np.mean(np.linalg.norm(C-y, axis=1))

def rotz(theta):
    """ Rotation matrix around z-axis. """
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

if __name__ == '__main__':
    # Example usage
    visualize = True
    x = np.linspace(0.0, 2.0, 30)
    y = x**2

    A = np.array([x, y, np.zeros_like(x)]).T  # 3D curve
    R = rotz(np.pi / 4)  # Rotation matrix
    t = np.array([1.0, 2.0, 3.0])  # Translation vector

    B = np.dot(A, R.T) + t  # Transformed curve


    T,_,_ = icp(A, B)
    
    print(f"R = {T[:3,:3]}")
    print(f"t = {T[:3,3]}")
    
    print(f"Error = {icp_error(A,B)}")
    print(f"Error 2D= {icp_error(A[:,:2],B[:,:2])}")
    if visualize:
        C = np.dot(A, T[:3,:3].T) + T[:3,3]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(A[:,0], A[:,1], A[:,2], c='r', label='A')
        ax.scatter(B[:,0], B[:,1], B[:,2], c='b', label='B')
        ax.scatter(C[:,0], C[:,1], C[:,2], c='g', label='Transform A->B')
        ax.legend()
        fig.savefig("test_icp.png")


