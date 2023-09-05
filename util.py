import numpy as np
import laspy

# points (n, 8) numpy -> x, y, z, intensity, point_index, red, green, blue,
def create_las(path:str, points)->None:
    n, c = points.shape
    
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.offsets = np.min(points[:,:3], axis=0)
    header.scales = np.array([0.01, 0.01, 0.01])
    # header.vlrs
    
    las = laspy.LasData(header)
    las.x = points[:,0]
    las.y = points[:,1]
    las.z = points[:,2]
    las.intensity = points[:,3].astype(np.uint8)
    las.red = points[:, 5].astype(np.uint8)
    las.green = points[:, 6].astype(np.uint8)
    las.blue = points[:, 7].astype(np.uint8)
    if c > 8:
        las.classification = points[:, 8]
    
    las.write(path)
    print(f'{n} points were saved on {path}.')