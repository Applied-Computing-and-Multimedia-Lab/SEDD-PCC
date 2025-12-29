import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull

def calculate_color_gamut_volume(ply_file):
    # Step 1: Read the PLY file
    pcd = o3d.io.read_point_cloud(ply_file)
    colors = np.asarray(pcd.colors)

    if len(colors) == 0:
        return 0  # Return 0 volume if no color information is found

    # Step 2: Convert the colors to the [0, 255] range
    colors = (colors * 255).astype(int)

    # Step 3: Calculate the volume of the RGB color space using the convex hull algorithm
    try:
        hull = ConvexHull(colors)
        color_gamut_volume = hull.volume
    except Exception as e:
        color_gamut_volume = 0
        print(f"Unable to calculate convex hull: {e}")

    return color_gamut_volume

def calculate_color_gamut_volume_normalized(ply_file):
    # Calculate the color gamut volume
    gamut_volume = calculate_color_gamut_volume(ply_file)

    # Maximum color gamut volume: The volume of the RGB color space (convex hull of the RGB cube)
    max_gamut_points = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                                 [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255]])
    max_gamut_volume = ConvexHull(max_gamut_points).volume

    # Normalize the volume (convert the result into a percentage)
    normalized_volume = (gamut_volume / max_gamut_volume) * 100
    return normalized_volume

# Test
ply_file = '/home/user/Desktop/Ryan/ANFPCAC_context_0725/testing_data/testdata/Owlii/dancer.ply'
normalized_gamut_volume = calculate_color_gamut_volume_normalized(ply_file)
print(f"Normalized color gamut volume: {normalized_gamut_volume:.2f}%")
