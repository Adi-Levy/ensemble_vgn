# load npy file data with grasps data and 3d plot the location of the grasps and the point color should be the score of the grasp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_grasps_w_score(npy_dir, req_round_id):
    file_list = os.listdir(npy_dir)
    file_list.sort()
    for file in file_list:
        if file.endswith(".npy"):
            file_name = file.split(".")[0]
            _, round_id, attempt = file_name.split("_")
            # for ech round plot attempts on the same scale
            if int(round_id) == req_round_id:
                npy_data = np.load(os.path.join(npy_dir, file), allow_pickle=True)
                print("attempt: ", attempt)
                print("locations: ", npy_data[:,0:3])
                print("scores: ", npy_data[:,7])
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(npy_data[:,0], npy_data[:,1], npy_data[:,2], c=npy_data[:,7], cmap=cm.coolwarm, linewidth=0.5)
                ax.set_xlim3d(0, 0.21)
                ax.set_ylim3d(0, 0.21)
                ax.set_zlim3d(0, 0.21)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                fig.colorbar(scatter)
                plt.title(f'Round {round_id} Attempt {attempt}')
                plt.show()


if __name__ == "__main__":
    base_dir = "/home/user_119/docker/grasp_repos/vgn_catkin_ws/src/vgn/sim_res"
    runs = os.listdir(base_dir)
    runs.sort()
    last_run = os.path.join(base_dir, runs[-1])
    # npy_dir = "/home/alexander/Documents/Research/GraspNet/GraspNetAPI/GraspNetAPI/sim_res/2021-05-14_16-30-11"
    plot_grasps_w_score(last_run, 0)
