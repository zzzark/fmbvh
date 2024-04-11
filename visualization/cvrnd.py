import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def _perspective_fov(fov, aspect_ratio, near_plane, far_plane):
    f, n, asp = far_plane, near_plane, aspect_ratio

    t = n * np.tan(fov / 2.0)
    b = -t
    r = asp * t
    l = -r

    f, n = -f, -n

    scale = np.array([
        [2/(r-l),       0,       0, 0],
        [      0, 2/(t-b),       0, 0],
        [      0,       0, 2/(n-f), 0],
        [      0,       0,       0, 1]
    ])
    trans = np.array([
        [1, 0, 0, -(l+r)/2],
        [0, 1, 0, -(b+t)/2],
        [0, 0, 1, -(n+f)/2],
        [0, 0, 0,        1],
    ])
    persp = np.array([
        [1, 0,       0,  0],
        [0, 1,       0,  0],
        [0, 0, (n+f)/n, -f],
        [0, 0,     1/n,  0],
    ])
    return scale @ trans @ persp


def _look_at(eye, at):
    up = np.array([0., 1., 0.])

    z_axis = eye - at
    z_axis = z_axis / np.linalg.norm(z_axis)
    phi = np.dot(z_axis, up)
    if phi == -1 or phi == 1:
        up = np.array([0., 0., -phi])

    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    return np.array([
        [           x_axis[0],            y_axis[0],            z_axis[0],  0.0],
        [           x_axis[1],            y_axis[1],            z_axis[1],  0.0],
        [           x_axis[2],            y_axis[2],            z_axis[2],  0.0],
        [-np.dot(x_axis, eye), -np.dot(y_axis, eye), -np.dot(z_axis, eye),  1.0]
    ]).T


def _draw_point(canvas: np.ndarray, xyz: list, mvp_mtx: np.ndarray, radius=4, color=(0,0,255), thickness=-1):
    raise NotImplementedError
    xyzw = np.array([*xyz, 1.0], dtype=float).reshape((4, 1))
    xyzw = mvp_mtx @ xyzw
    x = int(xyzw[0].item())
    y = int(xyzw[1].item())
    cv.circle(canvas, (x, y), radius, color, thickness)
    return canvas


def _draw_line(canvas: np.ndarray, xyz1: list, xyz2: list, mvp_mtx: np.ndarray, color=(0, 255, 0), linewidth=4):
    xyzw1 = np.array([*xyz1, 1.0], dtype=float)[:, None]
    xyzw2 = np.array([*xyz2, 1.0], dtype=float)[:, None]
    txyzw1 = mvp_mtx @ xyzw1
    txyzw2 = mvp_mtx @ xyzw2
    w1, w2 = txyzw1[3, 0], txyzw2[3, 0]

    # perspective projection
    x1, y1, z1 = txyzw1[:3, 0] / w1
    x2, y2, z2 = txyzw2[:3, 0] / w2

    # illegal points (outside the frustum)
    if abs(w1) < 1e-5 or abs(w2) < 1e-5:
        return canvas

    # simplified frustum clipping (z)
    if z1 < -1 or z1 > 1: return canvas
    if z2 < -1 or z2 > 1: return canvas

    # view port transform
    H, W = canvas.shape[0], canvas.shape[1]
    def _port_view(x, y):
        return int((x + 1.0)*0.5 * (W-1)), int((y + 1.0)*0.5 * (H-1))

    x1, y1 = _port_view(x1, y1)
    x2, y2 = _port_view(x2, y2)
    cv.line(canvas, (x1, y1), (x2, y2), color, linewidth)
    return canvas


class Renderer:
    def __init__(self, H=800, W=800):
        self.P = _perspective_fov(30.0, 1.0, 0.1, 100.0)  # degree, not rad
        self._eye = np.array([0.0, 3.0, 3.0])
        self._at = np.array([0.0, 0.0, 0.0])
        self.V = _look_at(self._eye, self._at)
        self.M = np.eye(4)

        self.H = H
        self.W = W
        self.canvas = np.ones((self.H, self.W, 3), dtype=np.uint8) * 255
    
    @property
    def VP(self):
        return self.P @ self.V
    
    @property
    def MVP(self):
        return self.P @ self.V @ self.M

    def begin(self):
        self.canvas[:] = 255

    def rotate_model(self, y_deg):
        y_deg = (y_deg + 1) % 360
        self.M[:3, :3] = R.from_euler('xyz', [0.0, y_deg, 0.0], degrees=True).as_matrix()
    
    def move_model(self, x, y, z):
        self.M[0, 3] = x
        self.M[1, 3] = y
        self.M[2, 3] = z
    
    def camera_follow(self, target, radius=None, theta=0/180*3.14, cam_y=2.0):
        if radius is not None:
            self._eye = np.array([np.cos(theta), cam_y/radius, np.sin(theta)]) * radius
        self.V = _look_at(self._eye + target, self._at + target)
    
    def draw_origin(self):
        mvp = self.MVP
        _draw_line(self.canvas, [+1,  0,  0], [0, 0, 0], mvp, (000, 000, 255), 1)
        _draw_line(self.canvas, [ 0, +1,  0], [0, 0, 0], mvp, (000, 255, 000), 1)
        _draw_line(self.canvas, [ 0,  0, +1], [0, 0, 0], mvp, (255, 000, 000), 1) 

    def draw_grid(self, count=9, unit=0.2):
        mvp = self.MVP
        for x in range(-count//2, count//2+1):
            for z in range(-count//2, count//2+1):
                a = [x*unit, 0.0, unit*(z-0.5)]
                b = [x*unit, 0.0, unit*(z+0.5)]
                _draw_line(self.canvas, a, b, mvp, (200, 200, 200), 1)

        for z in range(-count//2, count//2+1):
            for x in range(-count//2, count//2+1):
                a = [unit*(x-0.5), 0.0, z*unit]
                b = [unit*(x+0.5), 0.0, z*unit]
                _draw_line(self.canvas, a, b, mvp, (200, 200, 200), 1)

    def draw_cube(self):
        a = [-1.0, -1.0, -1.0]
        b = [+1.0, -1.0, -1.0]
        c = [+1.0, -1.0, +1.0]
        d = [-1.0, -1.0, +1.0]
        e = [-1.0, +1.0, -1.0]
        f = [+1.0, +1.0, -1.0]
        g = [+1.0, +1.0, +1.0]
        h = [-1.0, +1.0, +1.0]
        mvp = self.MVP
        _draw_line(self.canvas, a, b, mvp, (200, 50, 50), 1)
        _draw_line(self.canvas, a, d, mvp, (200, 50, 50), 1)
        _draw_line(self.canvas, a, e, mvp, (200, 50, 50), 1)
        _draw_line(self.canvas, b, c, mvp, (200, 50, 50), 1)
        _draw_line(self.canvas, b, f, mvp, (200, 50, 50), 1)
        _draw_line(self.canvas, c, d, mvp, (200, 50, 50), 1)
        _draw_line(self.canvas, c, g, mvp, (200, 50, 50), 1)
        _draw_line(self.canvas, d, h, mvp, (200, 50, 50), 1)
        _draw_line(self.canvas, e, f, mvp, (200, 50, 50), 1)
        _draw_line(self.canvas, e, h, mvp, (200, 50, 50), 1)
        _draw_line(self.canvas, f, g, mvp, (200, 50, 50), 1)
        _draw_line(self.canvas, f, g, mvp, (200, 50, 50), 1)
        _draw_line(self.canvas, g, h, mvp, (200, 50, 50), 1)

    def end(self):
        pass

    def render(self, lines, color=(200, 200, 100), linewidth=2):
        """
        :param lines: [
                        [[x1, y1, z1], [x2, y2, z2]]
                      ]
        """
        for [x1, y1, z1], [x2, y2, z2] in lines:
            _draw_line(self.canvas, [x1, y1, z1], [x2, y2, z2], self.MVP, color, linewidth)


def create_video_from_images(images, output_file, fps=30, size=None, format='avc1'):
    """
    Creates a video from a list of images.

    :param images: List of numpy arrays (images)
    :param output_file: Name of the output video file
    :param fps: Frames per second of the output video
    :param size: Size of the video frames (width, height). If None, the size of the first image is used.
    :return: None
    """
    if not images:
        raise ValueError("The images list is empty.")
    
    if size is None:
        # Assuming all images are the same size, use the size of the first image
        size = (images[0].shape[1], images[0].shape[0])
    
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*format)
    out = cv.VideoWriter(output_file, fourcc, fps, size)
    
    for img in images:
        if img.shape[:2] != size[::-1]:
            # Resize the image if it does not match the specified size
            img = cv.resize(img, size)
        
        # Write the frame to the video
        out.write(img)
    
    # Release everything when the job is finished
    out.release()


def render_pose(pindex, pos, output, fps=60, scale=None, 
                cam_y=1.5, cam_r=2.0, cam_t=45.0, 
                cam_damping=0.8, format="avc1"):
    """
    pindex:
    pos: [J, 3, T]
    output: path to save the video, None for displaying it
    fps:
    scale: scale the pose, None for auto scaling
    cam_y: camera y position
    cam_r: camera radius
    cam_t: camera rotation theta (xz plane)
    cam_damping: camera tracking damping

    return: list of rendered images
    """
    if not isinstance(pos, np.ndarray):
        import torch
        assert isinstance(pos, torch.Tensor)
        pos = pos.detach().cpu().numpy()
    if scale is None:
        scale = (pos[:, 1, :].max() - pos[:, 1, :].min())

    pos = pos / (scale)
    
    J, _, T = pos.shape

    target = pos[0].copy()  # root
    tgt = target[:, 0]

    images = []
    rnd = Renderer()
    loop = output is None
    pause = False

    bar = tqdm(range(T)) if output is not None else None
    while loop:
        t = 0
        while t < T:
            if bar is not None:
                next(bar)
            lines = []
            for j in range(J):
                p = pindex[j]
                if p < 0: continue
                lines.append([pos[p, :, t].tolist(), pos[j, :, t].tolist()])
            rnd.begin()

            tgt = cam_damping * tgt + (1-cam_damping) * target[:, t]
            rnd.camera_follow(tgt, cam_r, cam_t/180*3.14, cam_y)

            rnd.draw_grid(count=19, unit=0.2)
            rnd.draw_origin()

            rnd.render(lines)
            # rnd.draw_cube()

            if output is None:
                cv.imshow('cap', rnd.canvas)
                key = cv.waitKey(1000//fps)
                if key & 0xFF == ord('q'):
                    loop = False
                    break
                elif key & 0xFF == ord(' '):
                    pause = not pause
                elif key & 0xFF == ord('['):
                    t -= 1
                elif key & 0xFF == ord(']'):
                    t += 1
                elif key & 0xFF == ord('a'):
                    cam_t -= 1
                elif key & 0xFF == ord('d'):
                    cam_t += 1
                elif key & 0xFF == ord('s'):
                    cam_y -= 0.1
                elif key & 0xFF == ord('w'):
                    cam_y += 0.1
                elif key & 0xFF == ord('z'):
                    cam_r -= 0.1
                elif key & 0xFF == ord('x'):
                    cam_r += 0.1
            else:
                images.append(rnd.canvas.copy())
            rnd.end()
            if not pause:
                t += 1

    if output:
        create_video_from_images(images, output, fps, format=format)

    return images
