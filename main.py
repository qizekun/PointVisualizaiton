import argparse
import numpy as np
from utils import load, standardize_bbox, color_map, rotation
from render import render, render_part
from simple3d import showpoints as real_time_tool


def parse_args():
    parser = argparse.ArgumentParser('Point Cloud Visualizer')
    parser.add_argument('--path', type=str, help='the input file path', default='plane.ply')
    parser.add_argument('--render', help='using mitsuba to create beautiful image with shadow', action='store_true')
    parser.add_argument('--tool', help='using real time point cloud visualization tools', action='store_true')
    parser.add_argument('--num', type=int, help='downsample point num', default=np.inf)
    parser.add_argument('--knn', help='using KNN color map', action='store_true')
    parser.add_argument('--center_num', type=int, help='KNN center num', default=24)
    parser.add_argument('--part', help='perform KNN clustering on the objects and render each segment separately', action='store_true')
    parser.add_argument('--white', help='render white object', action='store_true')
    parser.add_argument('--RGB', nargs='+', help='render object with specific RGB value', default=[])
    parser.add_argument('--rot', nargs='+', help='rotation angle from x,y,z', default=[])
    parser.add_argument('--workdir', type=str, help='workdir', default='workdir')
    parser.add_argument('--output', type=str, help='output file name', default='result.jpg')
    parser.add_argument('--res', nargs='+', help='output file resolution', default=[800, 800])
    parser.add_argument('--radius', type=float, help='radius', default=0.025)
    parser.add_argument('--contrast', type=float, help='contrast', default=0.0004)
    parser.add_argument('--separator', type=str, help='text separator', default=",")
    parser.add_argument('--type', type=str, help='render type, include point and voxel', default="point")
    parser.add_argument('--mask', help='mask the point cloud', action='store_true')
    parser.add_argument('--view', nargs='+', help='the x,y,z position of camera view point', default=[2.75, 2.75, 2.75])
    parser.add_argument('--translate', nargs='+', help='the x,y,z position of object translate', default=[0, 0, 0])
    parser.add_argument('--scale', nargs='+', help='the x,y,z scale of object', default=[1, 1, 1])
    parser.add_argument('--median', help='using median filter', action='store_true')
    parser.add_argument('--bbox', type=str, help='realtime tools bbox visualization', default='none')

    args = parser.parse_args()
    return args


def main():
    config = parse_args()
    if config.render and config.tool:
        raise RuntimeWarning('both render and real time tool are selected')
    if config.render is False and config.tool is False:
        raise RuntimeWarning('you need to choose one of render or real time tool')

    # load the point cloud
    pcl = load(config.path, config.separator)

    if config.tool:
        bbox = None if config.bbox == 'none' else np.load(config.bbox)
        real_time_tool(pcl, config, bbox)
    else:
        # standardize the point cloud
        pcl = standardize_bbox(config, pcl)

        # rotate the point
        if len(config.rot) != 0:
            assert len(config.rot) == 3
            rot_matrix = rotation(config.rot)
            pcl[:, :3] = np.matmul(pcl[:, :3], rot_matrix)

        # color the point cloud
        pcl = color_map(config, pcl)

        if config.part:
            render_part(config, pcl)
        elif config.render:
            render(config, pcl)


if __name__ == '__main__':
    main()
