import cv2
import numpy as np
from plyfile import PlyData


def load(path, separator=','):
    extension = path.split('.')[-1]
    if extension == 'npy':
        pcl = np.load(path, allow_pickle=True)
    elif extension == 'npz':
        pcl = np.load(path)
        pcl = pcl['pred']
    elif extension == 'ply':
        ply = PlyData.read(path)
        vertex = ply['vertex']
        (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
        pcl = np.column_stack((x, y, z))
    elif extension == 'txt':
        f = open(path, 'r')
        line = f.readline()
        data = []
        while line:
            x, y, z = line.split(separator)[:3]
            data.append([float(x), float(y), float(z)])
            line = f.readline()
        f.close()
        pcl = np.array(data)
    elif extension == 'pth':
        import torch
        pcl = torch.load(path, map_location='cpu')
        pcl = pcl.detach().numpy()
    else:
        print('unsupported file format.')
        raise FileNotFoundError

    print(f'point cloud shape: {pcl.shape}')
    assert pcl.shape[-1] == 3 or pcl.shape[-1] == 6

    if len(pcl.shape) == 3:
        pcl = pcl[0]
        print("the dimension is 3, we select the first element in the batch.")
    return pcl


def load_self_colormap(value_path):
    vec = load(path=value_path)  # load the value of each point, and the shape is (N)
    vec = np.power(vec, 2)  # You can adjust the Level Curve with gamma transformation
    vec = 255 - 255 * (vec - np.min(vec)) / (np.max(vec) - np.min(vec))  # normalize to [0, 255]
    vec = vec.reshape(1, -1).astype(np.uint8)
    vec = cv2.applyColorMap(vec, cv2.COLORMAP_JET)  # apply colormap
    color = vec.reshape(-1, 3) / 255  # normalize to [0, 1]
    color = 0.5 * color + 0.5 * 0.5
    return color


def generate_pos_colormap(x, y, z, config, knn_center=[]):
    vec = np.array([x, y, z])
    if knn_center != []:
        dis = np.linalg.norm(knn_center - vec, axis=1)
        index = np.argmin(dis)
        vec = knn_center[index]

    vec = np.clip(vec, config.contrast, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm

    return [vec[0], vec[1], vec[2]]


def standardize_bbox(data):
    pcl = data[:, :3]
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    print("Center: {}, Scale: {}".format(center, scale))
    
    if data.shape[1] == 6:
        color = data[:, 3:]
        color[color < 0] = 0
        color[color > 1] = 1
        result = np.concatenate((result, color), axis=1)
        
    return result


def fps(data, k):
    N, C = data.shape
    sample_data = np.zeros((k, C))
    points = data[:, :3]
    color = data[:, 3:]
    barycenter = np.sum(points, axis=0) / points.shape[0]
    distance = np.full((points.shape[0]), np.nan)
    point = barycenter

    for i in range(k):
        distance = np.minimum(distance, np.sum((points - point) ** 2, axis=1))
        index = np.argmax(distance)
        point = points[index]
        sample_data[i] = np.concatenate((point, color[index]), axis=0)
        mask = np.ones((points.shape[0]), dtype=bool)
        mask[index] = False
        points = points[mask]
        distance = distance[mask]
    return sample_data


def rotation(rotation_angle):
    x, y, z = rotation_angle
    x, y, z = int(x), int(y), int(z)
    print(f'rotation angle: {x}, {y}, {z}')
    x_rad, y_rad, z_rad = np.radians(x), np.radians(y), np.radians(z)

    rot_x = np.array([[1, 0, 0], [0, np.cos(x_rad), -np.sin(x_rad)], [0, np.sin(x_rad), np.cos(x_rad)]])
    rot_y = np.array([[np.cos(y_rad), 0, np.sin(y_rad)], [0, 1, 0], [-np.sin(y_rad), 0, np.cos(y_rad)]])
    rot_z = np.array([[np.cos(z_rad), -np.sin(z_rad), 0], [np.sin(z_rad), np.cos(z_rad), 0], [0, 0, 1]])

    rot_matrix = np.dot(np.dot(rot_z, rot_y), rot_x)
    return rot_matrix


def get_xml(resolution=[1920, 1080], radius=0.025, object_type="point"):
    width, height = int(resolution[0]), int(resolution[1])
    if width / height > 4 / 3:
        position = "3,3,3"
    else:
        position = "2.5,2,2"
    xml_head = \
        f"""
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="{position}" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>
            <sampler type="independent">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="{width}"/>
                <integer name="height" value="{height}"/>
                <rfilter type="gaussian"/>
            </film>
        </sensor>

        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>

    """

    xml_ball_segment = \
        """
        <shape type="sphere">
            <float name="radius" value="%s"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """ % radius
    
    xml_cube_segment = \
        """
        <shape type="cube">
        <transform name="toWorld">
            <scale x="%s" y="%s" z="%s" />
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
    """ % (radius, radius, radius)

    xml_tail = \
        """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="10" y="10" z="2"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
        </shape>

        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="6,6,6"/>
            </emitter>
        </shape>
    </scene>
    """

    assert object_type == "point" or object_type == "voxel"
    xml_object_segment = xml_ball_segment if object_type == "point" else xml_cube_segment
    return xml_head, xml_object_segment, xml_tail
