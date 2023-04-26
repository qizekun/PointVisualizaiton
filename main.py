import argparse
import numpy as np
from utils import load, standardize_bbox, fps, load_self_colormap
from render import render, render_part, real_time_tool


def parse_args():
    parser = argparse.ArgumentParser('Point Cloud Visualizer')
    parser.add_argument('--path', type=str, help='the input file path', default='plane.ply')
    parser.add_argument('--render', help='using mitsuba3 create beautiful image', action='store_true')
    parser.add_argument('--tool', help='using real time point cloud visualization tools', action='store_true')
    parser.add_argument('--knn', help='using knn color map', action='store_true')
    parser.add_argument('--part', help='spilt object to parts and visualization', action='store_true')
    parser.add_argument('--white', help='white color', action='store_true')
    parser.add_argument('--value_path', help='point heat map value', default='')
    parser.add_argument('--RGB', nargs='+', help='RGB color', default=[])
    parser.add_argument('--num', type=int, help='downsample point num', default=np.inf)
    parser.add_argument('--center_num', type=int, help='knn center num', default=16)
    parser.add_argument('--workdir', type=str, help='workdir', default='workdir')
    parser.add_argument('--output', type=str, help='output file name', default='result.jpg')
    parser.add_argument('--resolution', nargs='+', help='output file resolution', default=[800, 800])
    parser.add_argument('--radius', type=float, help='radius', default=0.025)
    parser.add_argument('--contrast', type=float, help='contrast', default=0.0004)
    parser.add_argument('--separator', type=str, help='text separator', default=",")

    args = parser.parse_args()
    return args


def main():
    config = parse_args()
    if config.render and config.tool:
        raise RuntimeWarning('both render and real time tool are selected')
    if config.render is False and config.tool is False:
        raise RuntimeWarning('you need to choose one of render or real time tool')

    pcl = load(config.path, config.separator)
    print(f'point cloud shape: {pcl.shape}')

    # if rander the point with self colormap
    if config.value_path != "":
        color = load_self_colormap(config.value_path)
        pcl = np.concatenate((pcl, color), axis=1)  # add color to point cloud

    if config.num < pcl.shape[0]:
        print(f'downsample to {config.num} points')
        pt_indices = np.random.choice(pcl.shape[0], config.num, replace=False)
        np.random.shuffle(pt_indices)
        pcl = pcl[pt_indices]

    pcl = standardize_bbox(pcl)

    if config.part:
        render_part(config, pcl)
    elif config.render:
        render(config, pcl)
    else:
        real_time_tool(config, pcl)


if __name__ == '__main__':
    main()
