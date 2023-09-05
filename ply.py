import open3d as o3d
import numpy as np

def save_ply(point1, point2, file_path, color1, color2):
    
    colors1 = [color1 for _ in point1]
    colors2 = [color2 for _ in point2]

    combined_points = np.vstack((point1, point2))
    combined_colors = np.vstack((colors1, colors2))


    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(combined_points)
    point_cloud.colors = o3d.utility.Vector3dVector(combined_colors)
    o3d.io.write_point_cloud(file_path, point_cloud)