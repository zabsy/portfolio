"""Render the hero video: Apptronik Apollo (MuJoCo Menagerie, Apache-2.0) idling in a slow
seamless loop, drawn like a screenprint: the lighting cut into three flat greens, plus a slightly
misregistered dark outline, on a white ground.
The page lays it over the paper colour with `mix-blend-mode: multiply`, so white disappears.

    pip install mujoco numpy pillow imageio-ffmpeg
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
    MUJOCO_GL=glfw python render_robot.py frames/            # writes PNG frames
    MUJOCO_GL=glfw python render_robot.py --sheet sheet.png  # quick contact sheet instead

Then encode (see the ffmpeg lines at the bottom of this file).
Edit KEYS (glances) and SWAY (idle drift) to change how it moves, TONES / OUTLINE / LIGHT for the print.
"""
import sys, os
import numpy as np
import mujoco
from PIL import Image

MODEL = os.environ.get("APOLLO_XML", "mujoco_menagerie/apptronik_apollo/apptronik_apollo.xml")
SIZE = 1080            # output frame is SIZE x SIZE
SS = 2                 # supersampling for smooth edges
FPS = 30
LOOP = 16.0            # seconds
TONES = [np.array(c) / 255.0 for c in ((47, 74, 54), (92, 134, 102), (150, 186, 156))]  # shadow, mid, lit
CUTS = (0.45, 0.75)    # lighting levels where one tone switches to the next
OUTLINE = np.array([30, 40, 34]) / 255.0
OUTLINE_OFFSET = (4, -3)   # px (down, left) at output size: the "misregistered" second ink
OUTLINE_WIDTH = 2          # px at output size

# ---- what the robot does: mostly standing still, with a few glances around. ----
# Joint targets (radians) at times (s); between keys it eases, and it holds where two keys match.
REST = dict(r_shoulder_aa=-0.12, r_elbow_fe=-0.35, l_shoulder_aa=0.12, l_elbow_fe=-0.35)
KEYS = [
    (0.0,  REST),
    (2.0,  REST),
    (3.0,  {**REST, "neck_yaw": 0.42, "neck_pitch": 0.05, "torso_yaw": 0.05}),    # glance to its left
    (5.2,  {**REST, "neck_yaw": 0.42, "neck_pitch": 0.05, "torso_yaw": 0.05}),
    (6.4,  REST),
    (8.6,  REST),
    (9.6,  {**REST, "neck_yaw": -0.36, "neck_pitch": 0.14, "torso_yaw": -0.04,  # glance right, a little down
            "r_elbow_fe": -0.45}),
    (11.6, {**REST, "neck_yaw": -0.36, "neck_pitch": 0.14, "torso_yaw": -0.04,
            "r_elbow_fe": -0.45}),
    (12.8, {**REST, "neck_pitch": -0.04}),                                    # back to centre, chin up a touch
    (16.0, REST),
]
# slow idle sway layered on top: joint -> (amplitude rad, cycles per loop, phase). Whole cycles keep the loop seamless.
SWAY = {
    "torso_roll":    (0.012, 2, 0.0),
    "torso_pitch":   (0.010, 4, 1.1),   # "breathing"
    "r_shoulder_aa": (0.025, 2, 0.6),
    "l_shoulder_aa": (0.025, 2, 2.2),
    "r_elbow_fe":    (0.05, 3, 1.7),
    "l_elbow_fe":    (0.05, 3, 0.3),
}

m = mujoco.MjModel.from_xml_path(MODEL)
R = SIZE * SS
m.vis.global_.offwidth = R
m.vis.global_.offheight = R
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
q0 = d.qpos.copy()
adr = lambda j: m.joint(j).qposadr[0]

# ---- trajectory: linear between keys, then a circular blur so it is smooth and loops seamlessly ----
N = int(LOOP * FPS)
joints = sorted({j for _, k in KEYS for j in k} | set(SWAY))
t = np.arange(N) / FPS
traj = {}
for j in joints:
    kt = [k[0] for k in KEYS]
    kv = [k[1].get(j, 0.0) for k in KEYS]
    v = np.interp(t, kt, kv)
    sigma = 0.32 * FPS
    kx = np.arange(-int(4 * sigma), int(4 * sigma) + 1)
    kern = np.exp(-0.5 * (kx / sigma) ** 2); kern /= kern.sum()
    v = np.convolve(np.concatenate([v, v, v]), kern, mode="same")[N:2 * N]
    if j in SWAY:
        amp, cycles, phase = SWAY[j]
        v = v + amp * np.sin(2 * np.pi * cycles * t / LOOP + phase)
    traj[j] = v

cam = mujoco.MjvCamera()
cam.lookat[:] = [0.0, 0.0, 1.43]
cam.distance = 1.72
cam.azimuth = 168
cam.elevation = -2

ren = mujoco.Renderer(m, R, R)

# camera rays, for turning depth into 3D points (and then surface normals)
fovy = np.deg2rad(m.vis.global_.fovy)
f = (R / 2) / np.tan(fovy / 2)
az, el = np.deg2rad(cam.azimuth), np.deg2rad(cam.elevation)
fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
up = np.cross(right, fwd)
eye = cam.lookat - fwd * cam.distance
ys, xs = np.mgrid[0:R, 0:R].astype(np.float32)
rays = (fwd + ((xs - R / 2) / f)[..., None] * right + ((R / 2 - ys) / f)[..., None] * up).astype(np.float32)
LIGHT = -fwd + 0.7 * up + 0.6 * right   # from the upper left of the viewer
LIGHT /= np.linalg.norm(LIGHT)


def shift(a, dy, dx, fill):
    """Shift without wrapping; `fill` is what slides in from the border."""
    out = np.full_like(a, fill)
    H_, W_ = a.shape[:2]
    out[max(dy, 0):H_ + min(dy, 0), max(dx, 0):W_ + min(dx, 0)] = a[max(-dy, 0):H_ + min(-dy, 0), max(-dx, 0):W_ + min(-dx, 0)]
    return out


def box_blur(a, r):
    k = np.ones(2 * r + 1, np.float32) / (2 * r + 1)
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)


def frame(i):
    d.qpos[:] = q0
    for j in joints:
        d.qpos[adr(j)] = traj[j][i % N]
    mujoco.mj_forward(m, d)
    ren.enable_segmentation_rendering(); ren.update_scene(d, cam); seg = ren.render(); ren.disable_segmentation_rendering()
    ren.enable_depth_rendering(); ren.update_scene(d, cam); depth = ren.render(); ren.disable_depth_rendering()
    mask = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)

    pts = eye + rays * depth[..., None]
    n = np.cross(np.gradient(pts, axis=1), np.gradient(pts, axis=0))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    n[np.einsum("ijk,k->ij", n, fwd) > 0] *= -1
    shade = np.clip(np.einsum("ijk,k->ij", n, LIGHT), 0, 1)
    shade = box_blur(np.where(mask, shade, 0.6), 2)      # calm the bands down so they don't shimmer
    level = np.digitize(shade, CUTS)                      # 0 shadow, 1 mid, 2 lit

    img = np.ones((R, R, 3), np.float32)
    for k, col in enumerate(TONES):
        img[mask & (level == k)] = col
    w = OUTLINE_WIDTH * SS
    inner = mask.copy()
    for dy, dx in ((w, 0), (-w, 0), (0, w), (0, -w)):
        inner &= shift(mask, dy, dx, True)      # the robot continues past the frame edge: no outline there
    edge = shift(mask & ~inner, OUTLINE_OFFSET[0] * SS, OUTLINE_OFFSET[1] * SS, False)
    img[edge] = img[edge] * 0.1 + OUTLINE * 0.9
    return img.reshape(SIZE, SS, SIZE, SS, 3).mean(axis=(1, 3))


if __name__ == "__main__":
    if sys.argv[1] == "--sheet":
        ids = list(range(0, N, N // 12))[:12]
        sheet = Image.new("RGB", (360 * 4, 360 * 3), "white")
        for n_, i in enumerate(ids):
            tile = Image.fromarray((frame(i) * 255).astype(np.uint8)).resize((360, 360))
            sheet.paste(tile, ((n_ % 4) * 360, (n_ // 4) * 360))
        sheet.save(sys.argv[2])
    else:
        out = sys.argv[1]
        os.makedirs(out, exist_ok=True)
        for i in range(N):
            Image.fromarray((frame(i) * 255 + 0.5).astype(np.uint8)).save(f"{out}/f_{i:04d}.png")
        print("frames", N)

# Encode (ffmpeg from `python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`).
# The crop trims the empty margins of the square render; the poster is the first frame.
#   ffmpeg -framerate 30 -i frames/f_%04d.png \
#          -vf "crop=640:860:200:220,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p" \
#          -c:v libx264 -crf 27 -preset slow -tune animation -color_range tv -colorspace bt709 -color_primaries bt709 \
#          -color_trc bt709 -movflags +faststart -an ../images/robot.mp4
#   ffmpeg -i frames/f_0000.png -vf "crop=640:860:200:220" -q:v 3 ../images/robot-poster.jpg
