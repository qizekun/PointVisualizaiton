import sys
import cv2
import numpy as np
from plyfile import PlyData, PlyElement


def write_ply(save_path, points, text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)


showsz = 800
mousex, mousey = 0.5, 0.5
zoom = 1.0
changed = True


def onmouse(*args):
    global mousex, mousey, changed
    y = args[1]
    x = args[2]
    mousex = x / float(showsz) * 2
    mousey = y / float(showsz) * 2
    changed = True


cv2.namedWindow('show3d')
cv2.moveWindow('show3d', 0, 0)
cv2.setMouseCallback('show3d', onmouse)


def showpoints(pts, center, scale, config, bbox=None, waittime=0, showrot=False, magnifyBlue=0,
               freezerot=False, background=(255, 255, 255)):
    global showsz, mousex, mousey, zoom, changed

    # bbox = np.array(
    #     [[-0.53, -0.07, 0.01], [-0.57, -0.05, 0.01], [-0.57, -0.05, -0.03], [-0.53, -0.07, -0.03], [-0.58, -0.21, 0.01],
    #      [-0.62, -0.19, 0.01], [-0.62, -0.19, -0.03], [-0.58, -0.21, -0.03]]
    # )

    xyz = pts[:, :3]
    if pts.shape[-1] == 6:
        if pts[:, 3:].max() <= 1 + 1e-2:
            rgb = pts[:, 3:] * 255.0
        else:
            rgb = pts[:, 3:]
    else:
        rgb = np.zeros_like(xyz)

    if bbox is not None:
        bbox = (bbox - center) / scale
        bbox *= showsz / 1.5
    xyz *= showsz / 1.5

    show = np.zeros((showsz, showsz, 3), dtype='uint8')

    def render(bbox):
        rotmat = np.eye(3)
        if not freezerot:
            xangle = (mousey - 0.5) * np.pi * 1.2
        else:
            xangle = 0
        rotmat = rotmat.dot(np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(xangle), -np.sin(xangle)],
            [0.0, np.sin(xangle), np.cos(xangle)],
        ]))
        if not freezerot:
            yangle = (mousex - 0.5) * np.pi * 1.2
        else:
            yangle = 0
        rotmat = rotmat.dot(np.array([
            [np.cos(yangle), 0.0, -np.sin(yangle)],
            [0.0, 1.0, 0.0],
            [np.sin(yangle), 0.0, np.cos(yangle)],
        ]))
        rotmat *= zoom
        nxyz = xyz.dot(rotmat)
        result = nxyz.copy()
        nz = nxyz[:, 1].argsort()
        nxyz = nxyz[nz]
        nrgb = rgb[nz]
        nxyz = (nxyz[:, :2] + [showsz / 2, showsz / 2]).astype('int32')
        p = nxyz[:, 0] * showsz + nxyz[:, 1]
        show[:] = background
        m = (nxyz[:, 0] >= 0) * (nxyz[:, 0] < showsz) * (nxyz[:, 1] >= 0) * (nxyz[:, 1] < showsz)
        if not config.bgr2rgb:
            nrgb = nrgb[:, [2, 1, 0]]
        show.reshape((showsz * showsz, 3))[p[m]] = nrgb[m]

        if bbox is not None:
            interpolated_list = [bbox,
                                 np.linspace(bbox[0], bbox[1], 100),
                                 np.linspace(bbox[1], bbox[2], 100),
                                 np.linspace(bbox[2], bbox[3], 100),
                                 np.linspace(bbox[3], bbox[0], 100),
                                 np.linspace(bbox[4], bbox[5], 100),
                                 np.linspace(bbox[5], bbox[6], 100),
                                 np.linspace(bbox[6], bbox[7], 100),
                                 np.linspace(bbox[7], bbox[4], 100),
                                 np.linspace(bbox[0], bbox[4], 100),
                                 np.linspace(bbox[1], bbox[5], 100),
                                 np.linspace(bbox[2], bbox[6], 100),
                                 np.linspace(bbox[3], bbox[7], 100)]
            bbox = np.concatenate(interpolated_list, axis=0)

            bbox1 = bbox.copy()
            bbox1[:, 0] = bbox1[:, 0] + 1
            bbox2 = bbox.copy()
            bbox2[:, 1] = bbox2[:, 1] + 1
            bbox3 = bbox.copy()
            bbox3[:, 2] = bbox3[:, 2] + 1
            bbox = np.concatenate([bbox, bbox1, bbox2, bbox3], axis=0)

            nbbox = bbox.dot(rotmat)
            nbbox = (nbbox[:, :2] + [showsz / 2, showsz / 2]).astype('int32')
            p = nbbox[:, 0] * showsz + nbbox[:, 1]
            m = (nbbox[:, 0] >= 0) * (nbbox[:, 0] < showsz) * (nbbox[:, 1] >= 0) * (nbbox[:, 1] < showsz)
            show.reshape((showsz * showsz, 3))[p[m]] = [0, 0, 255]

        if magnifyBlue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=0))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=1))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=1))
        if showrot:
            cv2.putText(show, 'xangle %d' % (int(xangle / np.pi * 180)), (30, showsz - 30), 0, 0.5,
                        cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, 'yangle %d' % (int(yangle / np.pi * 180)), (30, showsz - 50), 0, 0.5,
                        cv2.cv.CV_RGB(255, 0, 0))
            cv2.putText(show, 'zoom %d%%' % (int(zoom * 100)), (30, showsz - 70), 0, 0.5, cv2.cv.CV_RGB(255, 0, 0))
        return result

    changed = True
    while True:
        if changed:
            result = render(bbox)
            changed = False
        cv2.imshow('show3d', show)
        if waittime == 0:
            cmd = cv2.waitKey(10) % 256
        else:
            cmd = cv2.waitKey(waittime) % 256
        if cmd == ord('q'):
            break
        elif cmd == ord('Q'):
            sys.exit(0)
        if cmd == ord('n'):
            zoom *= 1.1
            changed = True
        elif cmd == ord('m'):
            zoom /= 1.1
            changed = True
        elif cmd == ord('r'):
            zoom = 1.0
            changed = True
        elif cmd == ord('s'):
            img = np.array(show.data, dtype=np.uint8)
            image_with_alpha = np.zeros((showsz, showsz, 4), dtype=np.uint8)
            image_with_alpha[:, :, :3] = img
            image_with_alpha[:, :, 3] = 255
            white_areas = (image_with_alpha[:, :, 0] == 255) & (image_with_alpha[:, :, 1] == 255) & (
                    image_with_alpha[:, :, 2] == 255)
            image_with_alpha[white_areas, 3] = 0
            cv2.imwrite(config.output, image_with_alpha)
        elif cmd == ord('p'):
            write_ply(config.output.split(".")[0] + ".ply", result)
        if waittime != 0:
            break
    return cmd
