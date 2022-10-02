import os
import time
import mitsuba as mi
from utils import load, standardize_bbox, colormap, get_xml, fps
import simple3d
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Point Cloud Visualizer')
    parser.add_argument('--path', type=str, help='the input file path', default='plane.ply')
    parser.add_argument('--render', help='using mitsuba3 create beautiful image', action='store_true')
    parser.add_argument('--tool', help='using real time point cloud visualization tools', action='store_true')
    parser.add_argument('--knn', help='using knn',  action='store_true')
    parser.add_argument('--white', help='white color',  action='store_true')
    parser.add_argument('--RGB', nargs='+', help='RGB color', default=[])
    parser.add_argument('--num', type=int, help='downsample point num', default=1024)
    parser.add_argument('--center_num', type=int, help='knn center num', default=16)
    parser.add_argument('--workdir', type=str, help='workdir', default='workdir')
    parser.add_argument('--output', type=str, help='output file name', default='result.jpg')
    parser.add_argument('--resolution', nargs='+', help='output file resolution', default=[1920, 1080])
    parser.add_argument('--radius', type=float, help='radius', default=0.025)
    parser.add_argument('--contrast', type=float, help='radius', default=0.0004)

    args = parser.parse_args()
    return args

def render(config, pcl):

    file_name = config.path.split('.')[0]
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125

    if config.knn:
        knn_center = fps(pcl, config.center_num)
        knn_center += 0.5
        knn_center[:, 2] -= 0.0125
    else:
        knn_center = []

    xml_head, xml_ball_segment, xml_tail = get_xml(config.resolution, config.radius)
    xml_segments = [xml_head]
    for i in range(pcl.shape[0]):
        color = colormap(pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125, config, knn_center)
        xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    os.makedirs(config.workdir, exist_ok=True)
    xmlFile = f'{config.workdir}/{file_name}.xml'
    with open(xmlFile, 'w') as f:
        f.write(xml_content)
    f.close()

    mi.set_variant("scalar_rgb")
    scene = mi.load_file(xmlFile)
    image = mi.render(scene, spp=256)
    mi.util.write_bitmap(config.output, image)
    # To prevent errors in the output image, we delay two seconds
    time.sleep(2)
    os.remove(xmlFile)

def real_time_tool(config, pcl):
    simple3d.showpoints(pcl, config)

def main():
    config = parse_args()
    if config.render and config.tool:
        raise RuntimeWarning('both render and real time tool are selected')
    if config.render is False and config.tool is False:
        raise RuntimeWarning('you need to choose one of render or real time tool')
    

    path = config.path
    pcl = load(path)
    pcl = standardize_bbox(pcl, config.num)

    if config.render:
        render(config, pcl)
    else:
        real_time_tool(config, pcl)

if __name__ == '__main__':
    main()
