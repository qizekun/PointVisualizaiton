import os
import time
import mitsuba as mi
from utils import load, standardize_bbox, colormap, get_xml, fps
import simple3d
import argparse
import numpy as np
import torch
import cv2


def parse_args():
    parser = argparse.ArgumentParser('Point Cloud Visualizer')
    parser.add_argument('--path', type=str, help='the input file path', default='plane.ply')
    parser.add_argument('--render', help='using mitsuba3 create beautiful image', action='store_true')
    parser.add_argument('--tool', help='using real time point cloud visualization tools', action='store_true')
    parser.add_argument('--knn', help='using knn',  action='store_true')
    parser.add_argument('--part', help='part visualization',  action='store_true')
    parser.add_argument('--white', help='white color', action='store_true')
    parser.add_argument('--value_path', help='point value', default='')
    parser.add_argument('--RGB', nargs='+', help='RGB color', default=[])
    parser.add_argument('--num', type=int, help='downsample point num', default=1024)
    parser.add_argument('--center_num', type=int, help='knn center num', default=16)
    parser.add_argument('--workdir', type=str, help='workdir', default='workdir')
    parser.add_argument('--output', type=str, help='output file name', default='result.jpg')
    parser.add_argument('--resolution', nargs='+', help='output file resolution', default=[800, 800])
    parser.add_argument('--radius', type=float, help='radius', default=0.025)
    parser.add_argument('--contrast', type=float, help='radius', default=0.0004)
    parser.add_argument('--separator', type=str, help='text separator', default=",")

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

    # if rander the point with self colormap
    if config.value_path != "":
        vec = load(path=config.value_path)#[0]
        vec = 255 - 255 * (vec - np.min(vec))/(np.max(vec) - np.min(vec))
        L = len(vec)
        vec = vec.reshape(1, L).astype(np.uint8)
        vec = cv2.applyColorMap(vec, cv2.COLORMAP_JET)
        vec = vec.reshape(L, 3) / 255
        for i in range(pcl.shape[0]):
            color = 0.5 * vec[i] + 0.5 * 0.5
            xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
    # rander the point with position colormap
    else:
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
    # To prevent errors in the output image, we delay some seconds
    time.sleep(config.resolution[0] / 1000)
    os.remove(xmlFile)

def render_part(config, pcl):
    file_name = config.path.split('.')[0]
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125

    knn_center = fps(pcl, config.center_num)
    knn_center += 0.5
    knn_center[:, 2] -= 0.0125

    # config.resolution[0] /= 2
    # config.resolution[1] /= 2
    config.radius *= 2

    pcl_list = [[] for i in range(config.center_num)]
    for i in range(pcl.shape[0]):
        x, y, z = pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125
        temp = abs(knn_center[:, 0] - x) + abs(knn_center[:, 1] - y) + abs(knn_center[:, 2] - z)
        index = np.argmin(temp)
        pcl_list[index].append(pcl[i])

    for i in range(config.center_num):
        knn_patch = np.array(pcl_list[i])
        xml_head, xml_ball_segment, xml_tail = get_xml(config.resolution, config.radius)
        xml_segments = [xml_head]

        knn_patch = standardize_bbox(knn_patch, len(knn_patch))
        for j in range(len(knn_patch)):
            color = colormap(knn_patch[j, 0] + 0.5, knn_patch[j, 1] + 0.5, knn_patch[j, 2] + 0.5 - 0.0125, config, [])
            xml_segments.append(xml_ball_segment.format(knn_patch[j, 0], knn_patch[j, 1], knn_patch[j, 2], *color))
        
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

        output_file = config.output.split('.')[0] + f'_{str(i)}.' + config.output.split('.')[1]
        mi.util.write_bitmap(output_file, image)
        # To prevent errors in the output image, we delay some seconds
        time.sleep(config.resolution[0] / 1000)
        os.remove(xmlFile)

def real_time_tool(config, pcl):
    simple3d.showpoints(pcl, config)

def main():
    config = parse_args()
    if config.render and config.tool:
        raise RuntimeWarning('both render and real time tool are selected')
    if config.render is False and config.tool is False:
        raise RuntimeWarning('you need to choose one of render or real time tool')
    

    pcl = load(config.path, config.separator)
    if config.part:
        config.num = min(pcl.shape[0], config.num * 4)
    pcl = standardize_bbox(pcl, config.num, config)

    if config.part:
        render_part(config, pcl)
    elif config.render:
        render(config, pcl)
    else:
        real_time_tool(config, pcl)

if __name__ == '__main__':
    main()
