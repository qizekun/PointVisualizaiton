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


def showpoints(pts, config, bbox=None, c0=None, c1=None, c2=None, waittime=0, showrot=False, magnifyBlue=0,
               freezerot=False,
               background=(255, 255, 255), normalizecolor=True):
    global showsz, mousex, mousey, zoom, changed

    xyz = pts[:, :3]
    if pts.shape[-1] == 6:
        rgb = pts[:, 3:] * 255.0
    else:
        rgb = np.zeros_like(xyz)

    if bbox is not None:
        bbox = bbox - xyz.mean(axis=0)
        xyz = xyz - xyz.mean(axis=0)
        radius = ((xyz ** 2).sum(axis=-1) ** 0.5).max()
        bbox /= (radius * 2.2) / showsz
        xyz /= (radius * 2.2) / showsz
    else:
        xyz = xyz - xyz.mean(axis=0)
        radius = ((xyz ** 2).sum(axis=-1) ** 0.5).max()
        xyz /= (radius * 2.2) / showsz

    if c0 is None:
        c0 = np.zeros((len(xyz),), dtype='float32')
    if c1 is None:
        c1 = c0
    if c2 is None:
        c2 = c0
    if normalizecolor:
        c0 /= (c0.max() + 1e-14) / 255.0
        c1 /= (c1.max() + 1e-14) / 255.0
        c2 /= (c2.max() + 1e-14) / 255.0

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
        nxyz = (nxyz[:, :2] + [showsz / 2, showsz / 2]).astype('int32')
        p = nxyz[:, 0] * showsz + nxyz[:, 1]
        show[:] = background
        m = (nxyz[:, 0] >= 0) * (nxyz[:, 0] < showsz) * (nxyz[:, 1] >= 0) * (nxyz[:, 1] < showsz)
        nrgb = rgb[:, [2, 1, 0]]
        show.reshape((showsz * showsz, 3))[p[m]] = nrgb[m]

        if bbox is not None:
            interpolated_list = [bbox,
                                 np.linspace(bbox[0], bbox[1], 50),
                                 np.linspace(bbox[1], bbox[2], 50),
                                 np.linspace(bbox[2], bbox[3], 50),
                                 np.linspace(bbox[3], bbox[0], 50),
                                 np.linspace(bbox[4], bbox[5], 50),
                                 np.linspace(bbox[5], bbox[6], 50),
                                 np.linspace(bbox[6], bbox[7], 50),
                                 np.linspace(bbox[7], bbox[4], 50),
                                 np.linspace(bbox[0], bbox[4], 50),
                                 np.linspace(bbox[1], bbox[5], 50),
                                 np.linspace(bbox[2], bbox[6], 50),
                                 np.linspace(bbox[3], bbox[7], 50)]
            bbox = np.concatenate(interpolated_list, axis=0)

            nxyz = bbox.dot(rotmat)
            nz = nxyz[:, 1].argsort()
            nxyz = nxyz[nz]
            nxyz = (nxyz[:, :2] + [showsz / 2, showsz / 2]).astype('int32')
            p = nxyz[:, 0] * showsz + nxyz[:, 1]
            m = (nxyz[:, 0] >= 0) * (nxyz[:, 0] < showsz) * (nxyz[:, 1] >= 0) * (nxyz[:, 1] < showsz)
            c00 = np.zeros((len(bbox),), dtype='float32')
            c01 = np.ones((len(bbox),), dtype='float32') + 255.0
            c02 = np.zeros((len(bbox),), dtype='float32')
            c00 /= (c00.max() + 1e-14) / 255.0
            c01 /= (c01.max() + 1e-14) / 255.0
            c02 /= (c02.max() + 1e-14) / 255.0
            show.reshape((showsz * showsz, 3))[p[m], 1] = c00[nz][m]
            show.reshape((showsz * showsz, 3))[p[m], 2] = c01[nz][m]
            show.reshape((showsz * showsz, 3))[p[m], 0] = c02[nz][m]

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
            cv2.imwrite(config.output, show)
        elif cmd == ord('p'):
            write_ply(config.output.split(".")[0] + ".ply", result)
        if waittime != 0:
            break
    return cmd
