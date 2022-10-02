import numpy as np
from plyfile import PlyData


def load(path, Separator=';'):
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
        f = open(path,'r') 
        line = f.readline() 
        data = []
        while line:
            x,y,z = line.split(Separator)
            data.append([float(x), float(y), float(z)])
            line = f.readline()
        f.close() 
        pcl = np.array(data)
    else:
        print('unsupported file format.')
        raise FileNotFoundError
    return pcl

def colormap(x, y, z, config, knn_center=[]):

    if config.white:
        return [0.6, 0.6, 0.6]

    vec = np.array([x, y, z])
    if knn_center != []:
        temp = abs(knn_center[:, 0] - x) + abs(knn_center[:, 1] - y) + abs(knn_center[:, 2] - z)
        index = np.argmin(temp)
        vec = knn_center[index]

    vec = np.clip(vec, config.contrast, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm

    return [vec[0], vec[1], vec[2]]


def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


def fps(points, k):
    sample_points = np.zeros((k, 3))
    
    # 用离重心最远的初始化
    barycenter = np.sum(points, axis=0)/points.shape[0]
    print(barycenter)
    distance = np.full((points.shape[0]), np.nan)
    point = barycenter
    
    # # 随机初始化
    # point = points[0]
    # points = points[:-1,:]
    # sample_points[0,:] = point
    # distance = np.sum((points - point)**2, axis=1)
    for i in range(k):
        distance = np.minimum(distance, np.sum((points - point)**2, axis=1)) # 前缀思想更新最小值
        index = np.argmax(distance)
        point = points[index]
        sample_points[i,:] = point
        mask = np.ones((points.shape[0]), dtype=bool)
        mask[index] = False
        points = points[mask]
        distance = distance[mask]
    return sample_points


def get_xml(resolution=[1920, 1080], radius=0.025):
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
                <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>
            <sampler type="independent">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="{resolution[0]}"/>
                <integer name="height" value="{resolution[1]}"/>
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
    f"""
        <shape type="sphere">
            <float name="radius" value="{radius}"/>
            <transform name="toWorld">
                <translate x="{"{}"}" y="{"{}"}" z="{"{}"}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{"{}"},{"{}"},{"{}"}"/>
            </bsdf>
        </shape>
    """

    xml_tail = \
    """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
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
    return xml_head, xml_ball_segment, xml_tail