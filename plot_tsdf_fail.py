import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

if __name__ == "__main__":
    date_time = "23-01-17-19-53-11"
    data_dir = f"data/experiments/{date_time}/scenes/"
    scene_id = "e5829b60535848b69e1db747027319e1"
    # load scene data from npz file
    scene_data = np.load(data_dir + scene_id + ".npz")
    tsdf_vol = scene_data["grid"].squeeze()
    points = scene_data["points"]
    
    # Assume that the TSDF is stored in a NumPy ndarray called "tsdf"

    # Create a figure and 3D axes
    fig = plt.figure()
    ax1 = fig.add_subplot(122, projection='3d')
    ax2 = fig.add_subplot(121, projection='3d')

    # Extract the x, y, and z coordinates of the points in the TSDF where the value is greater than 0
    x1, y1, z1 = np.where(np.isfinite(tsdf_vol) & (tsdf_vol > 0))
    indxs = np.where(z1 > 9)
    x1 = x1[indxs]
    y1 = y1[indxs]
    z1 = z1[indxs]    

    # Extract the values of the TSDF at each point
    v = tsdf_vol[np.isfinite(tsdf_vol) & (tsdf_vol > 0)]
    v= v[indxs]

    points = points[points[:,2] > 0.051]
    
    # Plot the points as a scatter plot
    ax1.scatter(x1, y1, z1, c=v, cmap='coolwarm')
    ax2.scatter(points[:,0], points[:,1], points[:,2], c='r')
    ax1.set_xlim3d(0, 40)
    ax1.set_ylim3d(0, 40)
    ax1.set_zlim3d(0, 40)
    ax2.set_xlim3d(0, 0.3)
    ax2.set_ylim3d(0, 0.3)
    ax2.set_zlim3d(0, 0.3)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Show the plot
    plt.show()
   
    print("test") 
   

