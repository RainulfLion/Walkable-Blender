"""Procedural, seamlessly tileable textures for the clubhouse castle.

Pure numpy, no downloads: every map is generated from seeded noise so the
build is reproducible and the textures carry no third-party licence.
All colour maps are sRGB uint8 arrays (row 0 = top of the image).
"""
import struct
import zlib

import numpy as np


# ── helpers ────────────────────────────────────────────────────────────
def write_png(path, arr):
    """Minimal PNG writer (8-bit RGB)."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w, _ = arr.shape
    raw = np.hstack([np.zeros((h, 1), np.uint8), arr.reshape(h, w * 3)]).tobytes()

    def chunk(tag, data):
        body = tag + data
        return struct.pack('>I', len(data)) + body + struct.pack('>I', zlib.crc32(body) & 0xFFFFFFFF)

    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b'IDAT', zlib.compress(raw, 6)))
        f.write(chunk(b'IEND', b''))


def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def pnoise(S, cx, cy, rng):
    """Periodic value noise: cx by cy random cells stretched over S pixels."""
    g = rng.random((cy, cx))
    ys = np.arange(S) * cy / S
    xs = np.arange(S) * cx / S
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = smooth(ys - y0)[:, None]
    fx = smooth(xs - x0)[None, :]
    y1 = (y0 + 1) % cy
    x1 = (x0 + 1) % cx
    a = g[np.ix_(y0, x0)]
    b = g[np.ix_(y0, x1)]
    c = g[np.ix_(y1, x0)]
    d = g[np.ix_(y1, x1)]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(S, cx, cy, rng, octaves=5, gain=0.5):
    out = np.zeros((S, S))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        if max(cx, cy) * 2 ** o > S // 2:
            break
        out += amp * pnoise(S, cx * 2 ** o, cy * 2 ** o, rng)
        tot += amp
        amp *= gain
    out /= tot
    lo, hi = out.min(), out.max()
    return (out - lo) / (hi - lo + 1e-9)


def normal_from_height(h, strength):
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * strength
    dy = (np.roll(h, 1, 0) - np.roll(h, -1, 0)) * strength
    n = np.dstack([-dx, -dy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return to_u8(n * 0.5 + 0.5)


def _grid(S):
    yy, xx = np.mgrid[0:S, 0:S]
    return yy / S, xx / S


# ── tileable materials ─────────────────────────────────────────────────
def stone_wall(S, seed=1):
    """Coursed ashlar blocks. One tile = 3 m: 8 courses, 4 blocks per course."""
    rng = np.random.default_rng(seed)
    R, C = 8, 4
    yy, xx = _grid(S)
    row = np.floor(yy * R).astype(int)
    fv = yy * R - row
    off = rng.random(R) * 0.3 + (np.arange(R) % 2) * 0.5
    u = (xx * C + off[row]) % C
    col = np.floor(u).astype(int) % C
    fu = u - np.floor(u)
    tint = rng.random((R, C))[row, col]
    hue = (rng.random((R, C, 3)) * 0.03 - 0.015)[row, col]
    d = np.minimum(np.minimum(fu, 1 - fu) * (S / C), np.minimum(fv, 1 - fv) * (S / R))
    m = smooth(d / 5.0)
    edge = smooth(d / 16.0)
    n1 = fbm(S, 6, 6, rng)
    n2 = fbm(S, 48, 48, rng, octaves=3)
    base = np.array([0.60, 0.57, 0.52])
    colr = base * (0.78 + 0.34 * tint[..., None]) + hue
    colr = colr * (0.84 + 0.30 * n1[..., None]) * (0.92 + 0.16 * n2[..., None])
    colr = colr * (0.80 + 0.20 * edge[..., None])
    mortar = np.array([0.36, 0.34, 0.31]) * (0.9 + 0.2 * n2[..., None])
    out = mortar * (1 - m[..., None]) + colr * m[..., None]
    height = m * (0.55 + 0.2 * tint + 0.3 * n1 + 0.15 * n2)
    return to_u8(out), normal_from_height(height, 4.0 * S / 1024)


def stone_floor(S, seed=2):
    """Irregular flagstones from a periodic Voronoi diagram. One tile = 3 m."""
    rng = np.random.default_rng(seed)
    N = 5
    yy, xx = _grid(S)
    f1 = np.full((S, S), 9.0)
    f2 = np.full((S, S), 9.0)
    ident = np.zeros((S, S), int)
    k = 0
    for j in range(N):
        for i in range(N):
            px = (i + 0.15 + 0.7 * rng.random()) / N
            py = (j + 0.15 + 0.7 * rng.random()) / N
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    dist = (xx - px - ox) ** 2 + (yy - py - oy) ** 2
                    closer = dist < f1
                    f2 = np.where(closer, f1, np.minimum(f2, dist))
                    ident = np.where(closer, k, ident)
                    f1 = np.where(closer, dist, f1)
            k += 1
    edge = np.sqrt(f2) - np.sqrt(f1)
    m = smooth(edge / 0.014)
    tint = rng.random(k)[ident]
    hue = (rng.random((k, 3)) * 0.025 - 0.0125)[ident]
    n1 = fbm(S, 6, 6, rng)
    n2 = fbm(S, 48, 48, rng, octaves=3)
    base = np.array([0.53, 0.51, 0.47])
    colr = (base * (0.84 + 0.22 * tint[..., None]) + hue) * (0.85 + 0.28 * n1[..., None])
    colr *= (0.93 + 0.14 * n2[..., None])
    grout = np.array([0.30, 0.28, 0.25]) * (0.9 + 0.2 * n2[..., None])
    out = grout * (1 - m[..., None]) + colr * m[..., None]
    height = m * (0.6 + 0.15 * tint + 0.2 * n1 + 0.1 * n2)
    return to_u8(out), normal_from_height(height, 3.5 * S / 1024)


def wood(S, seed=3):
    """Six planks running along V. One tile = 2 m."""
    rng = np.random.default_rng(seed)
    P = 6
    yy, xx = _grid(S)
    grain = fbm(S, 64, 3, rng, octaves=4)
    fine = fbm(S, 128, 8, rng, octaves=2)
    w = S // P
    for p in range(P):
        sl = slice(p * w, (p + 1) * w if p < P - 1 else S)
        grain[:, sl] = np.roll(grain[:, sl], int(rng.integers(S)), axis=0)
    plank = np.minimum(np.floor(xx * P).astype(int), P - 1)
    tint = rng.random(P)[plank]
    val = np.clip(0.65 * grain + 0.35 * fine, 0, 1)
    dark = np.array([0.27, 0.16, 0.08])
    light = np.array([0.55, 0.37, 0.20])
    colr = dark + (light - dark) * val[..., None]
    colr = colr * (0.82 + 0.34 * tint[..., None])
    fu = xx * P - plank
    gap = smooth(np.minimum(fu, 1 - fu) * (S / P) / 3.0)
    joint_y = rng.random(P)[plank]
    dj = np.abs(((yy - joint_y + 0.5) % 1.0) - 0.5) * S
    joint = smooth(dj / 2.5)
    shade = (0.3 + 0.7 * gap) * (0.45 + 0.55 * joint)
    return to_u8(colr * shade[..., None])


def plaster(S, seed=4):
    rng = np.random.default_rng(seed)
    n1 = fbm(S, 4, 4, rng)
    n2 = fbm(S, 64, 64, rng, octaves=3)
    base = np.array([0.87, 0.82, 0.71])
    out = base * (0.90 + 0.12 * n1[..., None]) * (0.96 + 0.08 * n2[..., None])
    return to_u8(out)


def roof_tiles(S, seed=5):
    """Terracotta shingles, lower edge toward the bottom of the image."""
    rng = np.random.default_rng(seed)
    R, C = 8, 6
    yy, xx = _grid(S)
    row = np.floor(yy * R).astype(int)
    fv = yy * R - row
    u = (xx * C + (np.arange(R) % 2)[row] * 0.5) % C
    col = np.floor(u).astype(int) % C
    fu = u - np.floor(u)
    tint = rng.random((R, C))[row, col]
    hue = (rng.random((R, C, 3)) * 0.08 - 0.04)[row, col]
    n = fbm(S, 32, 32, rng, octaves=3)
    base = np.array([0.62, 0.30, 0.18])
    colr = (base * (0.75 + 0.4 * tint[..., None]) + hue) * (0.9 + 0.2 * n[..., None])
    shade = 0.72 + 0.32 * fv
    shade = shade * (0.45 + 0.55 * smooth((1 - fv) * (S / R) / 5.0))
    shade = shade * (0.5 + 0.5 * smooth(np.minimum(fu, 1 - fu) * (S / C) / 2.5))
    height = fv * 0.8 + 0.2 * tint
    return to_u8(colr * shade[..., None]), normal_from_height(height, 2.0 * S / 1024)


def grass(S, seed=6):
    rng = np.random.default_rng(seed)
    n1 = fbm(S, 5, 5, rng)
    n2 = fbm(S, 40, 40, rng, octaves=4)
    n3 = fbm(S, 9, 9, rng)
    base = np.array([0.27, 0.40, 0.16])
    dry = np.array([0.46, 0.45, 0.22])
    colr = base * (0.75 + 0.5 * n1[..., None]) * (0.85 + 0.3 * n2[..., None])
    mix = (smooth((n3 - 0.55) / 0.3) * 0.4)[..., None]
    colr = colr * (1 - mix) + dry * mix
    colr += (rng.random((S, S, 1)) - 0.5) * 0.05
    return to_u8(colr)


def dirt(S, seed=7):
    rng = np.random.default_rng(seed)
    n1 = fbm(S, 5, 5, rng)
    n2 = fbm(S, 40, 40, rng, octaves=4)
    base = np.array([0.47, 0.38, 0.27])
    colr = base * (0.78 + 0.4 * n1[..., None]) * (0.86 + 0.28 * n2[..., None])
    colr += (rng.random((S, S, 1)) - 0.5) * 0.06
    return to_u8(colr)


# ── fresco panels ──────────────────────────────────────────────────────
def _paint(img, mask, color):
    m = np.clip(mask, 0, 1)[..., None]
    return img * (1 - m) + np.array(color) * m


def _inside(sdf, S):
    """Anti-aliased mask from a signed distance in panel units (<0 = inside)."""
    return np.clip(0.5 - sdf * S / 1.5, 0, 1)


def _circle(X, Y, cx, cy, r, S):
    return _inside(np.hypot(X - cx, Y - cy) - r, S)


def _rect(X, Y, x0, x1, y0, y1, S):
    d = np.maximum(np.maximum(x0 - X, X - x1), np.maximum(y0 - Y, Y - y1))
    return _inside(d, S)


def _fresco_landscape(X, Y, S):
    t = Y[..., None]
    img = np.array([0.90, 0.85, 0.70]) * (1 - t) + np.array([0.55, 0.68, 0.78]) * t
    img = _paint(img, _circle(X, Y, 0.76, 0.80, 0.085, S), (0.86, 0.62, 0.20))
    img = _paint(img, _inside(Y - (0.42 + 0.08 * np.sin(X * 6.3 + 1.0)), S), (0.45, 0.52, 0.36))
    wall = (0.62, 0.36, 0.26)
    roof = (0.60, 0.22, 0.16)
    for x0, x1, top in ((0.30, 0.62, 0.60), (0.25, 0.34, 0.70), (0.58, 0.67, 0.70), (0.41, 0.51, 0.76)):
        img = _paint(img, _rect(X, Y, x0, x1, 0.38, top, S), wall)
        cren = _rect(X, Y, x0, x1, top, top + 0.025, S) * (np.floor(X * 60) % 2 == 0)
        img = _paint(img, cren, wall)
    for xc, top in ((0.295, 0.725), (0.625, 0.725), (0.46, 0.785)):
        tri = np.maximum(np.abs(X - xc) - (top + 0.10 - Y) * 0.55, top - Y)
        img = _paint(img, _inside(tri, S), roof)
    img = _paint(img, _rect(X, Y, 0.435, 0.485, 0.38, 0.47, S), (0.20, 0.14, 0.10))
    img = _paint(img, _inside(Y - (0.30 + 0.06 * np.sin(X * 9.4 + 2.5) + 0.02 * np.sin(X * 19)), S),
                 (0.33, 0.42, 0.24))
    road_c = 0.46 + 0.20 * (0.40 - Y)
    road = np.maximum(np.abs(X - road_c) - (0.02 + 0.10 * (0.40 - Y)), Y - 0.40)
    img = _paint(img, _inside(road, S), (0.72, 0.62, 0.44))
    for tx, ty in ((0.12, 0.30), (0.20, 0.26), (0.80, 0.27), (0.88, 0.31), (0.72, 0.22)):
        img = _paint(img, _rect(X, Y, tx - 0.008, tx + 0.008, ty - 0.05, ty, S), (0.30, 0.20, 0.12))
        img = _paint(img, _circle(X, Y, tx, ty + 0.03, 0.04, S), (0.22, 0.33, 0.18))
    return img


def _shield_sdf(X, Y, scale=1.0):
    sx = np.abs(X - 0.5) / scale
    y = 0.5 + (Y - 0.5) / scale
    half = np.where(y >= 0.5, 0.27, 0.27 * np.sqrt(np.clip(1 - ((0.5 - y) / 0.36) ** 2, 0, 1)))
    return np.maximum(np.maximum(sx - half, y - 0.82), 0.14 - y) * scale


def _fresco_shield(X, Y, S):
    img = np.ones(X.shape + (3,)) * np.array([0.50, 0.14, 0.12])
    lat = np.minimum(np.abs(((X + Y) * 10) % 1 - 0.5), np.abs(((X - Y) * 10) % 1 - 0.5))
    img = _paint(img, _inside(lat / 10 - 0.004, S), (0.40, 0.09, 0.08))
    img = _paint(img, _inside(_shield_sdf(X, Y, 1.07), S), (0.20, 0.13, 0.08))
    inside = _inside(_shield_sdf(X, Y), S)
    img = _paint(img, inside, (0.80, 0.62, 0.22))
    chev = np.abs(Y - (0.64 - 0.8 * np.abs(X - 0.5))) - 0.055
    img = _paint(img, _inside(chev, S) * inside, (0.16, 0.25, 0.50))
    for cx, cy in ((0.36, 0.72), (0.64, 0.72), (0.50, 0.34)):
        img = _paint(img, _circle(X, Y, cx, cy, 0.045, S) * inside, (0.60, 0.15, 0.12))
    return img


def _fresco_tree(X, Y, S):
    img = np.ones(X.shape + (3,)) * np.array([0.88, 0.82, 0.66])
    img = _paint(img, _circle(X, Y, 0.16, 0.84, 0.06, S), (0.82, 0.62, 0.22))
    img = _paint(img, _inside(Y - (0.16 + 0.015 * np.sin(X * 25)), S), (0.40, 0.46, 0.28))
    trunk = np.maximum(np.abs(X - 0.5) - (0.03 + 0.10 * np.clip(0.45 - Y, 0, 1) ** 2 * 4), np.maximum(0.13 - Y, Y - 0.56))
    img = _paint(img, _inside(trunk, S), (0.38, 0.25, 0.15))
    greens = ((0.30, 0.42, 0.22), (0.24, 0.36, 0.20))
    blobs = ((0.50, 0.68, 0.17), (0.35, 0.60, 0.12), (0.65, 0.60, 0.12),
             (0.41, 0.77, 0.11), (0.59, 0.77, 0.11), (0.50, 0.84, 0.09))
    for i, (cx, cy, r) in enumerate(blobs):
        img = _paint(img, _circle(X, Y, cx, cy, r, S), greens[i % 2])
    fruit = ((0.42, 0.62), (0.58, 0.64), (0.50, 0.74), (0.33, 0.58), (0.67, 0.57),
             (0.45, 0.82), (0.57, 0.80), (0.50, 0.58), (0.38, 0.70), (0.63, 0.71))
    for i, (cx, cy) in enumerate(fruit):
        img = _paint(img, _circle(X, Y, cx, cy, 0.018, S), (0.70, 0.20, 0.14) if i % 3 else (0.84, 0.66, 0.24))
    return img


def _fresco_quartered(X, Y, S):
    blue = np.array([0.16, 0.25, 0.48])
    cream = np.array([0.86, 0.80, 0.64])
    q = ((X < 0.5) ^ (Y < 0.5))[..., None]
    img = np.where(q, blue, cream)
    for cx, cy in ((0.27, 0.73), (0.73, 0.27)):
        ang = np.arctan2(Y - cy, X - cx)
        r = 0.12 * (0.45 + 0.55 * np.cos(4 * ang) ** 2)
        img = _paint(img, _inside(np.hypot(X - cx, Y - cy) - r, S), (0.82, 0.64, 0.24))
    red = (0.58, 0.16, 0.13)
    for cx, cy in ((0.73, 0.73), (0.27, 0.27)):
        img = _paint(img, _rect(X, Y, cx - 0.06, cx + 0.06, cy - 0.12, cy + 0.06, S), red)
        img = _paint(img, _rect(X, Y, cx - 0.08, cx + 0.08, cy + 0.06, cy + 0.09, S), red)
        cren = _rect(X, Y, cx - 0.08, cx + 0.08, cy + 0.09, cy + 0.12, S) * (np.floor((X - cx + 0.08) * 31.25) % 2 == 0)
        img = _paint(img, cren, red)
        img = _paint(img, _rect(X, Y, cx - 0.02, cx + 0.02, cy - 0.12, cy - 0.04, S), (0.22, 0.14, 0.10))
    cross = np.minimum(np.abs(X - 0.5), np.abs(Y - 0.5)) - 0.008
    img = _paint(img, _inside(cross, S), (0.78, 0.62, 0.25))
    return img


def fresco_panels(P, seed=8):
    """Four square painted panels (the default wall art), each P pixels."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:P, 0:P]
    X = (xx + 0.5) / P
    Y = 1.0 - (yy + 0.5) / P
    panels = []
    for painter in (_fresco_landscape, _fresco_shield, _fresco_tree, _fresco_quartered):
        img = painter(X, Y, P)
        border = np.minimum(np.minimum(X, 1 - X), np.minimum(Y, 1 - Y))
        img = _paint(img, _inside(border - 0.058, P), (0.78, 0.62, 0.25))
        img = _paint(img, _inside(border - 0.045, P), (0.55, 0.25, 0.16))
        # age it: pull toward grey, mottle, and flake back to bare plaster
        lum = img.mean(axis=2, keepdims=True)
        img = img * 0.82 + lum * 0.18
        img = img * (0.84 + 0.2 * fbm(P, 6, 6, rng)[..., None])
        flake = smooth((fbm(P, 20, 20, rng, octaves=4) - 0.72) / 0.1)[..., None] * 0.7
        img = img * (1 - flake) + np.array([0.84, 0.79, 0.68]) * flake
        img += (rng.random((P, P, 1)) - 0.5) * 0.04
        panels.append(to_u8(img))
    return panels


def banner(field, W=512, H=1024, seed=9):
    """Default banner cloth (W x H, hangs with the top of the image at the rod).
    The bottom corners are cut away by the banner's pointed shape."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    X = (xx + 0.5) / W
    Y = 1.0 - (yy + 0.5) / H
    gold = (0.80, 0.60, 0.20)
    img = np.ones((H, W, 3)) * np.array(field)
    edge = np.minimum(X, 1 - X) * W
    img = _paint(img, np.clip((0.07 * W - edge) / 1.5, 0, 1), gold)
    img = _paint(img, np.clip((0.035 * W - edge) / 1.5, 0, 1), field)
    img = _paint(img, _rect(X, Y, 0, 1, 0.93, 1.0, H), gold)
    diamond = np.abs(X - 0.5) / 0.30 + np.abs(Y - 0.58) / 0.16 - 1.0
    img = _paint(img, np.clip(0.5 - diamond * 40, 0, 1), gold)
    weave = fbm(max(W, H), 6, 6, rng)[:H, :W]
    img = img * (0.86 + 0.2 * weave[..., None])
    return to_u8(img)


# name -> generator for the default art; real files live in the repo's art/ folder
ART = {
    'fresco1': lambda size: fresco_panels(size)[0],
    'fresco2': lambda size: fresco_panels(size)[1],
    'fresco3': lambda size: fresco_panels(size)[2],
    'fresco4': lambda size: fresco_panels(size)[3],
    'fresco5': lambda size: fresco_panels(size)[1],
    'fresco6': lambda size: fresco_panels(size)[3],
    'banner_red': lambda size: banner((0.52, 0.08, 0.08)),
    'banner_blue': lambda size: banner((0.10, 0.18, 0.44)),
}
ART_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')


def find_art(art_dir, name):
    """Path of art/<name>.<ext> if the user (or a previous build) put one there."""
    import os
    for ext in ART_EXTENSIONS:
        p = os.path.join(art_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


def generate_all(out_dir, size=1024):
    """Write every texture as PNG and return {name: path}."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    def save(name, arr):
        p = os.path.join(out_dir, name + '.png')
        write_png(p, arr)
        paths[name] = p

    c, n = stone_wall(size)
    save('stone_wall', c)
    save('stone_wall_n', n)
    c, n = stone_floor(size)
    save('stone_floor', c)
    save('stone_floor_n', n)
    c, n = roof_tiles(size)
    save('roof', c)
    save('roof_n', n)
    save('wood', wood(size))
    save('plaster', plaster(size))
    save('grass', grass(size))
    save('dirt', dirt(size))
    return paths


if __name__ == '__main__':
    import sys
    print(generate_all(sys.argv[1] if len(sys.argv) > 1 else 'textures'))
