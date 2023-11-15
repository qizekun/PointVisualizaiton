import os
import time
import numpy as np
import mitsuba as mi
from utils import standardize_bbox, generate_pos_colormap, get_xml, fps, mask_point


def render(config, pcl):
    file_name = config.path.split('.')[0]
    pcl[:, 2] *= -1
    pcl[:, 1] -= min(pcl[:, 1]) + 0.25

    if config.mask:
        pcl = mask_point(pcl)

    xml_head, xml_object_segment, xml_tail = get_xml(config.res, config.view, config.radius, config.type)
    xml_segments = [xml_head]

    translate = list(map(float, config.translate))
    scale = list(map(float, config.scale))

    for i in range(pcl.shape[0]):
        x, y, z = np.roll(pcl[i, :3], 1) * scale + translate
        color = pcl[i, 3:]
        xml_segments.append(xml_object_segment.format(x, y, z, *color))

    xml_segments.append(xml_tail)
    xml_content = str.join('', xml_segments)

    os.makedirs(config.workdir, exist_ok=True)
    xmlFile = f'{config.workdir}/{file_name.split("/")[-1]}.xml'
    with open(xmlFile, 'w') as f:
        f.write(xml_content)
    f.close()

    mi.set_variant("scalar_rgb")
    scene = mi.load_file(xmlFile)
    image = mi.render(scene, spp=256)
    mi.util.write_bitmap(config.output, image)
    # To prevent errors in the output image, we delay some seconds
    time.sleep(int(config.res[0]) / 1000)
    os.remove(xmlFile)


def render_part(config, pcl):
    file_name = config.path.split('.')[0]
    pcl = pcl[:, [2, 0, 1]]
    pcl[:, 0] *= -1
    pcl[:, 2] += 0.0125

    knn_center = fps(pcl, config.center_num)
    knn_center += 0.5
    knn_center[:, 2] -= 0.0125

    # config.res[0] /= 2
    # config.res[1] /= 2
    config.radius *= 2

    pcl_list = [[] for i in range(config.center_num)]
    for i in range(pcl.shape[0]):
        x, y, z = pcl[i, 0] + 0.5, pcl[i, 1] + 0.5, pcl[i, 2] + 0.5 - 0.0125
        temp = abs(knn_center[:, 0] - x) + abs(knn_center[:, 1] - y) + abs(knn_center[:, 2] - z)
        index = np.argmin(temp)
        pcl_list[index].append(pcl[i])

    for i in range(config.center_num):
        knn_patch = np.array(pcl_list[i])
        xml_head, xml_object_segment, xml_tail = get_xml(config.res, config.view, config.radius, config.type)
        xml_segments = [xml_head]

        knn_patch = standardize_bbox(knn_patch)
        for j in range(len(knn_patch)):
            color = generate_pos_colormap(knn_patch[j] + 0.5, config)
            xml_segments.append(xml_object_segment.format(knn_patch[j, 0], knn_patch[j, 1], knn_patch[j, 2], *color))

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
        time.sleep(int(config.res[0]) / 1000)
        os.remove(xmlFile)
