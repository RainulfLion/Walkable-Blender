"""Turn art/source/occidentals_crest.jpg into the banner and wall-picture images.

    python blender/make_crest_art.py        (needs Blender's Python or `pip install bpy`)

Writes art/banner_red.jpg, art/banner_blue.jpg, art/fresco4.jpg (Occidentals crest)
and art/fresco1.jpg, art/fresco2.jpg (the two lion shields).
"""
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from castle_textures import write_png  # noqa: E402

ART = os.path.join(os.path.dirname(HERE), 'art')
PARCHMENT = np.array([0.905, 0.870, 0.795])
GOLD = np.array([0.80, 0.60, 0.20])


def load(path, with_alpha=False):
    im = bpy.data.images.load(path)
    im.alpha_mode = 'STRAIGHT'
    w, h = im.size
    a = np.array(im.pixels[:]).reshape(h, w, 4)[::-1].copy()
    bpy.data.images.remove(im)
    return (a[..., :3], a[..., 3:4]) if with_alpha else a[..., :3]


def resize(img, w, h):
    """Area-averaged bilinear resize (good enough for down- and mild up-scaling)."""
    H, W, _ = img.shape
    ys = (np.arange(h) + 0.5) * H / h - 0.5
    xs = (np.arange(w) + 0.5) * W / w - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, W - 2)
    fy = np.clip(ys - y0, 0, 1)[:, None, None]
    fx = np.clip(xs - x0, 0, 1)[None, :, None]
    a, b = img[y0][:, x0], img[y0][:, x0 + 1]
    c, d = img[y0 + 1][:, x0], img[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def save_jpg(name, img):
    tmp = os.path.join(ART, name + '.tmp.png')
    write_png(tmp, (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    im = bpy.data.images.load(tmp)
    bpy.context.scene.view_settings.view_transform = 'Standard'   # save colours as painted
    s = bpy.context.scene.render.image_settings
    s.file_format, s.quality = 'JPEG', 90
    im.save_render(os.path.join(ART, name + '.jpg'))
    bpy.data.images.remove(im)
    os.remove(tmp)
    print('wrote art/%s.jpg' % name)


def cutout(crop):
    """Alpha mask for the crest: 0 where the parchment background is, 1 on the arms.
    Background = parchment-coloured pixels connected to the edge of the crop, plus
    small enclosed pockets that have the same warm tint (so the silver helmet stays)."""
    diff = np.abs(crop - PARCHMENT).max(axis=2)
    warm = (crop[..., 0] - crop[..., 2]) > 0.06
    cand = (diff < 0.13) & warm & (crop.min(axis=2) > 0.62)
    bg = np.zeros(cand.shape, bool)
    bg[0, :], bg[-1, :], bg[:, 0], bg[:, -1] = cand[0, :], cand[-1, :], cand[:, 0], cand[:, -1]
    while True:
        grow = bg.copy()
        grow[1:, :] |= bg[:-1, :]
        grow[:-1, :] |= bg[1:, :]
        grow[:, 1:] |= bg[:, :-1]
        grow[:, :-1] |= bg[:, 1:]
        grow &= cand
        if (grow == bg).all():
            break
        bg = grow
    bg |= (diff < 0.07) & warm & (crop.min(axis=2) > 0.70)      # enclosed pockets
    return _soften(bg)


def cutout_white(img):
    """Alpha mask for art on a plain white page: only white that touches the edge of
    the image is removed, so white charges inside the shield are kept."""
    cand = img.min(axis=2) > 0.90
    bg = np.zeros(cand.shape, bool)
    bg[0, :], bg[-1, :], bg[:, 0], bg[:, -1] = cand[0, :], cand[-1, :], cand[:, 0], cand[:, -1]
    while True:
        grow = bg.copy()
        grow[1:, :] |= bg[:-1, :]
        grow[:-1, :] |= bg[1:, :]
        grow[:, 1:] |= bg[:, :-1]
        grow[:, :-1] |= bg[:, 1:]
        grow &= cand
        if (grow == bg).all():
            break
        bg = grow
    return _soften(bg)


def _soften(bg):
    fg = (~bg).astype(float)
    # pull the edge in by a pixel and soften it so no pale fringe is left
    er = fg.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        er = np.minimum(er, np.roll(fg, (dy, dx), (0, 1)))
    soft = sum(np.roll(er, (dy, dx), (0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
    return soft[..., None]


def place(img, crest, alpha, width, x_mid, y_top):
    h = round(crest.shape[0] * width / crest.shape[1])
    c, a = resize(crest, width, h), resize(np.repeat(alpha, 3, axis=2), width, h)
    x0 = x_mid - width // 2
    img[y_top:y_top + h, x0:x0 + width] = img[y_top:y_top + h, x0:x0 + width] * (1 - a) + c * a
    return y_top + h


def banner(crest, alpha, field):
    """512 x 1229 hanging banner: the crest straight on the coloured cloth."""
    W, H = 512, 1229
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.ones((H, W, 3)) * field
    edge = np.minimum(xx, W - 1 - xx)
    img[(edge >= 14) & (edge < 21)] = GOLD                # gold piping down both sides
    place(img, crest, alpha, 456, W // 2, 150)
    img[yy < 70] = GOLD                                   # sleeve at the rod
    return img


def picture(crest, alpha, field, height=876):
    """1024 square wall picture: the arms on a plain painted field with a gold border."""
    img = np.ones((1024, 1024, 3)) * field
    width = round(crest.shape[1] * height / crest.shape[0])
    h = round(crest.shape[0] * width / crest.shape[1])
    place(img, crest, alpha, width, 512, (1024 - h) // 2)
    yy, xx = np.mgrid[0:1024, 0:1024]
    edge = np.minimum(np.minimum(xx, 1023 - xx), np.minimum(yy, 1023 - yy))
    img[edge < 60] = GOLD
    img[edge < 46] = np.array([0.55, 0.25, 0.16])
    return img


def main():
    src = load(os.path.join(ART, 'source', 'occidentals_crest.jpg'))
    crest = src[30:760, 385:895]                          # the arms without the empty margins
    alpha = cutout(crest)
    save_jpg('banner_red', banner(crest, alpha, np.array([0.60, 0.10, 0.10])))
    save_jpg('banner_blue', banner(crest, alpha, np.array([0.12, 0.22, 0.52])))
    green = np.array([0.10, 0.22, 0.17])
    save_jpg('fresco4', picture(crest, alpha, green))
    # the two lion shields replace the placeholder castle (fresco1) and device (fresco2)
    for name, source in (('fresco1', 'lion_crescent.webp'), ('fresco2', 'lion_star.webp')):
        shield, alpha = load(os.path.join(ART, 'source', source), with_alpha=True)
        if alpha.min() > 0.99:                            # no transparency in the file: key out white
            alpha = cutout_white(shield)
        save_jpg(name, picture(shield, alpha, green, height=800))


if __name__ == '__main__':
    main()
