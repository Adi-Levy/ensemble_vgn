import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from src.vgn.utils.transform import Transform, Rotation
import os

# plot grasp quality for each round from grasp.npy files with color coded score
def plot_grasp_qual(npy_dir):
    file_list = os.listdir(npy_dir)
    file_list.sort()
    for file in file_list:
        # if file.endswith(".npy"):
            # file_name = file.split(".")[0]
            # _, round_id, attempt = file_name.split("_")
        # for ech round plot attempts on the same scale
        npy_data = np.load(os.path.join(npy_dir, file), allow_pickle=True)
        grasp_qual = npy_data[0]
        grasp_rot = npy_data[1]
        grasp_width = npy_data[2]
        positions = []
        for i in range(0, grasp_qual.shape[0]):
            for j in range(0, grasp_qual.shape[1]):
                for k in range(0, grasp_qual.shape[2]):
                    if grasp_qual[i,j,k] > 0:
                        pos = np.array([i,j,k], dtype=np.float32) * (6*0.05/40)
                        positions.append(np.concatenate([pos, grasp_qual[i,j,k].copy().reshape(1)]))

        positions = np.array(positions)
        # print("attempt: ", attempt)
        # print("locations: ", npy_data[:,0:3])
        # print("scores: ", npy_data[:,7])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(positions[:,0], positions[:,1], positions[:,2], c=positions[:,3], cmap=cm.coolwarm, linewidth=0.5)
        ax.set_xlim3d(0, 0.3)
        ax.set_ylim3d(0, 0.3)
        ax.set_zlim3d(0, 0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.colorbar(scatter)
        # plt.title(f'Round {round_id} Attempt {attempt}')
        plt.show()


if __name__ == "__main__":
    base_dir = "/home/user_119/docker/grasp_repos/vgn_catkin_ws/src/vgn/data/grasps"
    plot_grasp_qual(base_dir)
