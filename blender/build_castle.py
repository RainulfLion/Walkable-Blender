"""Clubhouse castle generator for Blender (tested on 4.5 and 5.x).

Builds a low-poly medieval castle laid out for a first-person web
walkthrough and exports it as a single GLB:

    blender --background --python blender/build_castle.py -- --out ClubhouseCastle.glb
    # or, with the pip "bpy" module:  python blender/build_castle.py --out ClubhouseCastle.glb

You can also open this file in Blender's Scripting tab and press Run; the
castle is added to the current scene in a "ClubhouseCastle" collection.

Conventions used by the web viewer (index.html):
  * objects named COL_*  -> invisible collision geometry (walls, floors, ramps)
  * empty named "Spawn"  -> player start position / facing
  * point lights         -> exported as KHR_lights_punctual torches
  * materials Art_<name> -> the viewer swaps in art/<name>.png|jpg|webp at load time,
                            so frescoes and banners can be changed without Blender

Layout (Blender units = metres, +Y is north, the gate faces south):
  gatehouse + drawbridge (south) -> courtyard with sunken fighting pit
  north range : throne room | great hall | kitchen
  east range  : armory | forge          west range : 4 guest rooms over the tavern cellar
  NW corner   : tall tower with spiral stair and top windows
  battlement walkways on all four walls, reached by two stairs beside the gate
"""
import math
import os
import random
import sys
import tempfile
from contextlib import contextmanager

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd())
import castle_textures  # noqa: E402

# ══════════════════════════════════════════════════════════════════════
# Geometry kit
# ══════════════════════════════════════════════════════════════════════
# metres covered by one repeat of each tileable texture
TILE = {'stone_wall': 3.0, 'stone_floor': 3.0, 'wood': 2.0, 'plaster': 3.0,
        'roof': 2.4, 'grass': 6.0, 'dirt': 4.0}


class Builder:
    """Accumulates polygons (with UVs and per-face materials) for one mesh."""

    def __init__(self):
        self.verts, self.faces, self.uvs, self.mat_idx, self.mat_names = [], [], [], [], []

    def face(self, pts, mat, uv=None, uvscale=1.0):
        P = np.asarray(pts, dtype=float)
        n = np.zeros(3)
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            n += np.cross(a, b)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            return
        n /= ln
        if uv is None:
            if abs(n[2]) > 0.95:
                t, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
            else:
                t = np.array([-n[1], n[0], 0.0])
                t /= np.linalg.norm(t)
                b = np.cross(n, t)
            s = TILE.get(mat, 1.0) * uvscale
            uv = [(float(p @ t) / s, float(p @ b) / s) for p in P]
        base = len(self.verts)
        self.verts.extend(map(tuple, P))
        self.faces.append(tuple(range(base, base + len(P))))
        self.uvs.extend(uv)
        if mat not in self.mat_names:
            self.mat_names.append(mat)
        self.mat_idx.append(self.mat_names.index(mat))

    def tri_count(self):
        return sum(len(f) - 2 for f in self.faces)

    def to_mesh(self, name, materials):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.verts, [], self.faces)
        uvl = me.uv_layers.new(name='UVMap')
        uvl.data.foreach_set('uv', [c for uv in self.uvs for c in uv])
        for m in self.mat_names:
            me.materials.append(materials[m])
        me.polygons.foreach_set('material_index', self.mat_idx)
        me.update()
        return me


class Ctx:
    vis = None          # Builder receiving visible faces
    col = None          # Builder receiving collision faces
    M = np.eye(4)       # current local -> world transform
    uvscale = 1.0
    section = None


SECTIONS = {}           # name -> (vis Builder, col Builder)
MODULES = {}            # name -> (vis Builder, col Builder)   reusable pieces
INSTANCES = []          # (module, 4x4 matrix, section)
LIGHTS = []             # (name, (x, y, z), color, power, section)


def _mat(tx=0, ty=0, tz=0, rot=0, sx=1, sy=1, sz=1):
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    return np.array([[c * sx, -s * sy, 0, tx],
                     [s * sx, c * sy, 0, ty],
                     [0, 0, sz, tz],
                     [0, 0, 0, 1.0]])


@contextmanager
def xf(tx=0, ty=0, tz=0, rot=0, sx=1, sy=1, sz=1):
    old = Ctx.M
    Ctx.M = old @ _mat(tx, ty, tz, rot, sx, sy, sz)
    try:
        yield
    finally:
        Ctx.M = old


@contextmanager
def section(name):
    if name not in SECTIONS:
        SECTIONS[name] = (Builder(), Builder())
    saved = (Ctx.vis, Ctx.col, Ctx.section)
    Ctx.vis, Ctx.col = SECTIONS[name]
    Ctx.section = name
    try:
        yield
    finally:
        Ctx.vis, Ctx.col, Ctx.section = saved


@contextmanager
def module(name, uvscale=1.0):
    """Define a reusable piece once; place() then makes linked instances."""
    MODULES[name] = (Builder(), Builder())
    saved = (Ctx.vis, Ctx.col, Ctx.M, Ctx.uvscale, Ctx.section)
    Ctx.vis, Ctx.col = MODULES[name]
    Ctx.M, Ctx.uvscale, Ctx.section = np.eye(4), uvscale, None
    try:
        yield
    finally:
        Ctx.vis, Ctx.col, Ctx.M, Ctx.uvscale, Ctx.section = saved


def place(name, x, y, z=0.0, rot=0.0, scale=(1, 1, 1), col=True):
    """Instance a module (shared mesh data) and bake its collision in place."""
    if isinstance(scale, (int, float)):
        scale = (scale, scale, scale)
    M = Ctx.M @ _mat(x, y, z, rot, *scale)
    INSTANCES.append((name, M, Ctx.section))
    if col:
        cb = MODULES[name][1]
        for f in cb.faces:
            P = np.array([cb.verts[i] for i in f])
            W = P @ M[:3, :3].T + M[:3, 3]
            Ctx.col.face(W, 'collision')


def emit(pts, mat, col=True, vis=True, uv=None):
    P = np.asarray(pts, dtype=float)
    W = P @ Ctx.M[:3, :3].T + Ctx.M[:3, 3]
    if vis and mat:
        Ctx.vis.face(W, mat, uv, Ctx.uvscale)
    if col:
        Ctx.col.face(W, 'collision')


def box(x0, x1, y0, y1, z0, z1, mat, col=True, vis=True, **f):
    """Axis-aligned box. Per-face overrides: xm xp ym yp zm zp = material or False."""
    def m(key):
        return f.get(key, mat)
    faces = (
        ('xm', [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)]),
        ('xp', [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
        ('ym', [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
        ('yp', [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)]),
        ('zp', [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
        ('zm', [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)]),
    )
    for key, pts in faces:
        mk = m(key)
        if mk is False:
            continue
        emit(pts, mk, col=col, vis=vis)


def extrude(pts, vec, front, back=None, side=None, col=True, vis=True):
    """Prism from a planar polygon swept along vec.
    front = material of the cap at pts, back = far cap, side = swept faces."""
    P = np.asarray(pts, dtype=float)
    keep = [i for i in range(len(P)) if np.linalg.norm(P[i] - P[i - 1]) > 1e-9]
    P = P[keep]
    v = np.asarray(vec, dtype=float)
    back = front if back is None else back
    side = front if side is None else side
    n = np.zeros(3)
    for i in range(len(P)):
        n += np.cross(P[i], P[(i + 1) % len(P)])
    if n @ v > 0:
        P = P[::-1]
    emit(P, front, col=col, vis=vis)
    emit((P + v)[::-1], back, col=col, vis=vis)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        emit([b, a, a + v, b + v], side, col=col, vis=vis)


def polar(cx, cy, r, deg, z=0.0):
    a = math.radians(deg)
    return (cx + r * math.cos(a), cy + r * math.sin(a), z)


def octagon(cx, cy, R, z, phase=22.5, n=8):
    return [polar(cx, cy, R, phase + i * 360.0 / n, z) for i in range(n)]


def drum(cx, cy, R, z0, z1, mat, top=None, bottom=None, col=True, vis=True):
    """Solid octagonal column (R = corner radius)."""
    lo, hi = octagon(cx, cy, R, z0), octagon(cx, cy, R, z1)
    for i in range(8):
        j = (i + 1) % 8
        emit([lo[i], lo[j], hi[j], hi[i]], mat, col=col, vis=vis)
    if top:
        emit(hi, top, col=col, vis=vis)
    if bottom:
        emit(lo[::-1], bottom, col=col, vis=vis)


def cone(cx, cy, R, z0, z1, mat, Rin=None, under='wood'):
    """Octagonal spire with an optional soffit ring under the eaves."""
    e = octagon(cx, cy, R, z0)
    for i in range(8):
        emit([e[i], e[(i + 1) % 8], (cx, cy, z1)], mat, col=False)
    if Rin:
        q = octagon(cx, cy, Rin, z0)
        for i in range(8):
            j = (i + 1) % 8
            emit([q[i], q[j], e[j], e[i]], under, col=False)


def pyramid(cx, cy, z0, z1, r, mat):
    b = [(cx - r, cy - r, z0), (cx + r, cy - r, z0), (cx + r, cy + r, z0), (cx - r, cy + r, z0)]
    for i in range(4):
        emit([b[i], b[(i + 1) % 4], (cx, cy, z1)], mat, col=False)
    emit(b[::-1], mat, col=False)


def ring(cx, cy, Ro, Ri, z0, z1, mo, mi, openings=None, top=None, bottom=None, col=True):
    """Hollow octagonal wall with mitred corners.
    openings: {face: [(width, z_bottom, z_top, offset), ...]}, face k is centred on angle 45k.
    All openings on one face share the first entry's width/offset."""
    openings = openings or {}
    for k in range(8):
        a0, a1 = 45 * k - 22.5, 45 * k + 22.5
        o0, o1 = np.array(polar(cx, cy, Ro, a0)), np.array(polar(cx, cy, Ro, a1))
        i0, i1 = np.array(polar(cx, cy, Ri, a0)), np.array(polar(cx, cy, Ri, a1))

        def O(t, z):
            p = o0 + (o1 - o0) * t
            return (p[0], p[1], z)

        def I(t, z):
            p = i0 + (i1 - i0) * t
            return (p[0], p[1], z)

        def solid(ta, tb, za, zb):
            emit([O(ta, za), O(tb, za), O(tb, zb), O(ta, zb)], mo, col=col)
            emit([I(tb, za), I(ta, za), I(ta, zb), I(tb, zb)], mi, col=col)

        ops = sorted(openings.get(k, []), key=lambda o: o[1])
        if not ops:
            solid(0, 1, z0, z1)
        else:
            L = float(np.linalg.norm(o1 - o0))
            w = ops[0][0]
            off = ops[0][3] if len(ops[0]) > 3 else 0.0
            ta = ((L - w) / 2 + off) / L
            tb = ta + w / L
            solid(0, ta, z0, z1)
            solid(tb, 1, z0, z1)
            zs = z0
            for op in ops:
                zb, zt = op[1], op[2]
                if zb > zs + 1e-6:
                    solid(ta, tb, zs, zb)
                    emit([O(ta, zb), O(tb, zb), I(tb, zb), I(ta, zb)], mo, col=col)      # sill
                emit([O(ta, zb), I(ta, zb), I(ta, zt), O(ta, zt)], mo, col=col)          # jamb
                emit([I(tb, zb), O(tb, zb), O(tb, zt), I(tb, zt)], mo, col=col)          # jamb
                emit([O(ta, zt), I(ta, zt), I(tb, zt), O(tb, zt)], mo, col=col)          # soffit
                zs = zt
            if zs < z1 - 1e-6:
                solid(ta, tb, zs, z1)
        if top:
            emit([O(0, z1), O(1, z1), I(1, z1), I(0, z1)], top, col=col)
        if bottom:
            emit([O(1, z0), O(0, z0), I(0, z0), I(1, z0)], bottom, col=False)


def wall(p0, p1, z0, z1, t, mL, mR, openings=(), jamb=None, cap=None, col=True):
    """Straight wall from p0 to p1 (XY), thickness t centred on the line.
    mL / mR are the materials of the faces to the left / right of the direction.
    openings: (s0, s1, z_bottom, z_top, kind) with s measured along the wall and
    kind in '', 'arch' (door with arched head) or 'window' (arch + mullions)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    jamb = jamb or mR
    cap = cap or jamb
    with xf(p0[0], p0[1], 0, rot=math.degrees(math.atan2(dy, dx))):
        def piece(sa, sb, za, zb):
            if sb - sa < 1e-6 or zb - za < 1e-6:
                return
            box(sa, sb, -t / 2, t / 2, za, zb, jamb, col=col, yp=mL, ym=mR, zp=cap,
                zm=False if za <= z0 + 1e-6 else jamb)
        s = 0.0
        for op in sorted(openings):
            s0, s1, zb, zt = op[:4]
            kind = op[4] if len(op) > 4 else ''
            piece(s, s0, z0, z1)
            piece(s0, s1, z0, zb)
            piece(s0, s1, zt, z1)
            w = s1 - s0
            if kind in ('arch', 'window'):
                place('Arch', (s0 + s1) / 2, 0, zt - 0.4 * w, scale=(w, t + 0.06, w), col=False)
            if kind == 'window':
                place('Window', (s0 + s1) / 2, 0, zb, scale=(w, 1, zt - 0.4 * w - zb), col=False)
            s = s1
        piece(s, L, z0, z1)


def slab_with_holes(x0, x1, y0, y1, z0, z1, holes, mat, top):
    """Rectangular slab minus rectangular holes (x0, x1, y0, y1), as a few boxes."""
    xs = sorted({x0, x1, *[min(max(h[0], x0), x1) for h in holes], *[min(max(h[1], x0), x1) for h in holes]})
    ys = sorted({y0, y1, *[min(max(h[2], y0), y1) for h in holes], *[min(max(h[3], y0), y1) for h in holes]})
    for j in range(len(ys) - 1):
        run = None
        for i in range(len(xs) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(h[0] < cx < h[1] and h[2] < cy < h[3] for h in holes):
                if run:
                    box(run[0], run[1], ys[j], ys[j + 1], z0, z1, mat, zp=top, zm=False)
                run = None
            else:
                run = (run[0] if run else xs[i], xs[i + 1])
        if run:
            box(run[0], run[1], ys[j], ys[j + 1], z0, z1, mat, zp=top, zm=False)


def stair(x, y, z0, ang, n, rise, run, width, base=None, mat='stone_wall', tread='stone_floor',
          rail=None, rail_mat='stone_wall'):
    """Straight flight climbing along local +X from (x, y, z0); width spans local +Y.
    Visual = individual steps; collision = one smooth ramp so the walk feels fluid.
    rail = 'L' or 'R' adds a sloped parapet on that side."""
    L = n * run
    z1 = z0 + n * rise
    base = z0 if base is None else base
    with xf(x, y, 0, rot=ang):
        for i in range(n):
            box(i * run, (i + 1) * run, 0, width, base, z0 + (i + 1) * rise, mat, zp=tread, zm=False, col=False)
        extrude([(0, 0, base), (L, 0, base), (L, 0, z1), (0, 0, z0)], (0, width, 0), None, vis=False)
        if rail:
            yr = width if rail == 'L' else -0.25
            extrude([(0, yr, z0), (L, yr, z1), (L, yr, z1 + 0.95), (0, yr, z0 + 0.95)], (0, 0.25, 0), rail_mat)
            extrude([(0, yr, base), (L, yr, base), (L, yr, z1), (0, yr, z0)], (0, 0.25, 0), rail_mat, col=False)


def battlements(p0, p1, z):
    """Crenellated parapet along p0->p1, sitting on the left of the direction."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    n = max(1, round(L / 4.0))
    sx = L / (4.0 * n)
    with xf(p0[0], p0[1], z, rot=math.degrees(math.atan2(dy, dx))):
        for i in range(n):
            place('Battlement', i * 4.0 * sx, 0, 0, scale=(sx, 1, 1), col=False)
        box(0, L, 0, 0.4, 0, 1.8, None, vis=False)


def light(name, x, y, z, power=30.0, color=(1.0, 0.70, 0.40)):
    LIGHTS.append((name, (x, y, z), color, power, Ctx.section))


# ══════════════════════════════════════════════════════════════════════
# Reusable modules (each becomes ONE mesh, shared by all of its instances)
# ══════════════════════════════════════════════════════════════════════
def define_modules():
    # ── architecture ──
    with module('Arch', uvscale=1 / 3.0):
        # segmental arch head: unit width, 0.4 tall, unit depth (scaled per opening)
        R, cz = 0.5667, -0.2667
        prof = [(x, -0.5, cz + math.sqrt(R * R - x * x)) for x in np.linspace(-0.5, 0.5, 11)]
        prof += [(0.5, -0.5, 0.4), (-0.5, -0.5, 0.4)]
        extrude(prof, (0, 1, 0), 'stone_wall', col=False)

    with module('Window'):
        for x0, x1 in ((-0.5, -0.44), (-0.03, 0.03), (0.44, 0.5)):
            box(x0, x1, -0.03, 0.03, 0, 1, 'iron', col=False)
        for z0, z1 in ((0, 0.04), (0.48, 0.52), (0.96, 1.0)):
            box(-0.44, 0.44, -0.03, 0.03, z0, z1, 'iron', col=False)

    with module('Battlement'):
        box(0, 4, 0, 0.4, 0, 1.0, 'stone_wall', col=False, zm=False)
        box(0.5, 1.5, 0, 0.4, 1.0, 1.8, 'stone_wall', col=False, zm=False)
        box(2.5, 3.5, 0, 0.4, 1.0, 1.8, 'stone_wall', col=False, zm=False)

    with module('DoorLeaf'):
        box(0, 1, -0.04, 0.04, 0, 1, 'wood')
        for z in (0.2, 0.75):
            box(0, 0.9, -0.055, 0.055, z, z + 0.05, 'iron', col=False)

    with module('Truss'):
        # king-post roof truss spanning 11.2 m along local +X
        box(0, 11.2, -0.12, 0.12, 0, 0.3, 'wood', col=False)
        box(5.45, 5.75, -0.1, 0.1, 0.3, 4.3, 'wood', col=False)
        extrude([(0, -0.1, 0.3), (5.6, -0.1, 4.2), (5.6, -0.1, 4.45), (0, -0.1, 0.55)], (0, 0.2, 0), 'wood', col=False)
        extrude([(11.2, -0.1, 0.3), (11.2, -0.1, 0.55), (5.6, -0.1, 4.45), (5.6, -0.1, 4.2)], (0, 0.2, 0), 'wood', col=False)

    R = 4.4
    apo = R * math.cos(math.radians(22.5))
    Ri = (apo - 0.8) / math.cos(math.radians(22.5))
    with module('CornerTower'):
        # solid drum to the wall-walk, hollow guard room above with doors on local +X and +Y
        drum(0, 0, R, -2.5, 7.0, 'stone_wall')
        emit(octagon(0, 0, R, 7.02), 'stone_floor')
        ring(0, 0, R, Ri, 7.0, 10.6, 'stone_wall', 'stone_wall',
             openings={0: [(1.6, 7.0, 9.5)], 2: [(1.6, 7.0, 9.5)]})
        emit(octagon(0, 0, Ri, 10.5)[::-1], 'wood', col=False)
        cone(0, 0, 5.1, 10.3, 15.4, 'roof', Rin=R - 0.1)
        for k in (1, 3, 4, 5, 6, 7):
            with xf(rot=45 * k):
                box(apo - 0.05, apo + 0.03, -0.13, 0.13, 8.0, 9.5, 'dark', col=False)

    with module('Turret'):
        drum(0, 0, 2.6, -2.5, 10.0, 'stone_wall')
        drum(0, 0, 2.85, 10.0, 11.0, 'stone_wall', bottom='stone_wall')
        cone(0, 0, 3.2, 10.9, 15.0, 'roof', Rin=2.8)
        a = 2.6 * math.cos(math.radians(22.5))
        for k in range(8):
            with xf(rot=45 * k):
                box(a - 0.05, a + 0.03, -0.11, 0.11, 7.6, 9.0, 'dark', col=False)

    # ── furniture ──
    with module('TableLong'):
        box(-1.75, 1.75, -0.55, 0.55, 0.74, 0.82, 'wood', col=False)
        for x in (-1.3, 1.3):
            box(x - 0.06, x + 0.06, -0.45, 0.45, 0, 0.74, 'wood', col=False, zm=False)
        box(-1.3, 1.3, -0.04, 0.04, 0.3, 0.4, 'wood', col=False)
        box(-1.75, 1.75, -0.55, 0.55, 0, 0.82, None, vis=False)

    with module('Bench'):
        box(-1.75, 1.75, -0.18, 0.18, 0.40, 0.46, 'wood', col=False)
        for x in (-1.4, 1.4):
            box(x - 0.05, x + 0.05, -0.15, 0.15, 0, 0.40, 'wood', col=False, zm=False)
        box(-1.75, 1.75, -0.18, 0.18, 0, 0.46, None, vis=False)

    with module('RoundTable'):
        drum(0, 0, 0.75, 0.72, 0.80, 'wood', top='wood', bottom='wood', col=False)
        drum(0, 0, 0.12, 0.06, 0.72, 'wood', col=False)
        drum(0, 0, 0.40, 0, 0.06, 'wood', top='wood', col=False)
        box(-0.6, 0.6, -0.6, 0.6, 0, 0.8, None, vis=False)

    with module('Stool'):
        drum(0, 0, 0.22, 0.40, 0.45, 'wood', top='wood', bottom='wood', col=False)
        for a in (30, 150, 270):
            px, py, _ = polar(0, 0, 0.14, a)
            box(px - 0.03, px + 0.03, py - 0.03, py + 0.03, 0, 0.40, 'wood', col=False, zm=False)

    with module('SmallTable'):
        box(-0.4, 0.4, -0.4, 0.4, 0.70, 0.76, 'wood', col=False)
        for x in (-0.33, 0.33):
            for y in (-0.33, 0.33):
                box(x - 0.04, x + 0.04, y - 0.04, y + 0.04, 0, 0.70, 'wood', col=False, zm=False)
        pyramid(0, 0, 0.90, 0.99, 0.025, 'flame')
        box(-0.025, 0.025, -0.025, 0.025, 0.76, 0.90, 'linen', col=False)
        box(-0.4, 0.4, -0.4, 0.4, 0, 0.76, None, vis=False)

    with module('Bed'):
        # head toward local -X
        box(-1.0, 1.0, -0.5, 0.5, 0.15, 0.35, 'wood', col=False)
        box(-1.0, -0.92, -0.5, 0.5, 0, 0.95, 'wood', col=False, zm=False)
        box(0.92, 1.0, -0.5, 0.5, 0, 0.55, 'wood', col=False, zm=False)
        box(-0.92, 0.92, -0.45, 0.45, 0.35, 0.50, 'linen', col=False, zm=False)
        box(-0.25, 0.92, -0.47, 0.47, 0.50, 0.54, 'cloth_blue', col=False, zm=False)
        box(-0.86, -0.50, -0.30, 0.30, 0.50, 0.60, 'linen', col=False, zm=False)
        box(-1.0, 1.0, -0.5, 0.5, 0, 0.6, None, vis=False)

    with module('Chest'):
        box(-0.5, 0.5, -0.3, 0.3, 0, 0.5, 'wood', zm=False)
        for x in (-0.35, 0.35):
            box(x - 0.04, x + 0.04, -0.31, 0.31, 0, 0.51, 'iron', col=False, zm=False)
        box(-0.06, 0.06, -0.33, -0.3, 0.3, 0.42, 'iron', col=False)

    with module('Barrel'):
        drum(0, 0, 0.38, 0, 1.0, 'wood', top='wood', col=False)
        for z in (0.18, 0.76):
            drum(0, 0, 0.40, z, z + 0.07, 'iron', top='iron', bottom='iron', col=False)
        box(-0.36, 0.36, -0.36, 0.36, 0, 1.0, None, vis=False)

    with module('Crate'):
        box(-0.4, 0.4, -0.4, 0.4, 0, 0.8, 'wood', zm=False)

    with module('HayBale'):
        box(-0.6, 0.6, -0.35, 0.35, 0, 0.6, 'straw', zm=False)

    with module('Throne'):
        # faces local -Y
        box(-0.45, 0.45, -0.40, 0.40, 0, 0.55, 'wood', col=False, zm=False)
        box(-0.40, 0.40, -0.38, 0.28, 0.55, 0.62, 'cloth_red', col=False, zm=False)
        box(-0.52, 0.52, 0.30, 0.46, 0, 2.2, 'wood', col=False, zm=False)
        box(-0.38, 0.38, 0.27, 0.30, 0.72, 2.0, 'cloth_red', col=False)
        for x0, x1 in ((-0.58, -0.44), (0.44, 0.58)):
            box(x0, x1, -0.40, 0.34, 0, 0.92, 'wood', col=False, zm=False)
        for x in (-0.44, 0.44):
            pyramid(x, 0.38, 2.2, 2.5, 0.08, 'gold')
        pyramid(0, 0.38, 2.2, 2.65, 0.11, 'gold')
        box(-0.58, 0.58, -0.40, 0.46, 0, 1.2, None, vis=False)

    with module('WeaponRack'):
        # faces local -Y
        box(-0.9, 0.9, -0.2, 0.2, 0, 0.12, 'wood', col=False, zm=False)
        for x in (-0.9, 0.82):
            box(x, x + 0.08, 0.05, 0.15, 0.12, 1.5, 'wood', col=False)
        box(-0.9, 0.9, 0.05, 0.15, 1.3, 1.4, 'wood', col=False)
        for i in range(5):
            x = -0.64 + i * 0.32
            box(x - 0.02, x + 0.02, 0.0, 0.04, 0.12, 2.2, 'wood', col=False)
            pyramid(x, 0.02, 2.2, 2.5, 0.045, 'iron')
        box(-0.9, 0.9, -0.2, 0.2, 0, 1.5, None, vis=False)

    for name, cloth in (('ShieldRed', 'cloth_red'), ('ShieldBlue', 'cloth_blue')):
        with module(name):
            prof = [(-0.4, -0.06, 1.0), (-0.4, -0.06, 0.45), (-0.25, -0.06, 0.15), (0, -0.06, 0),
                    (0.25, -0.06, 0.15), (0.4, -0.06, 0.45), (0.4, -0.06, 1.0)]
            extrude(prof, (0, 0.06, 0), cloth, back='wood', side='iron', col=False)
            box(-0.08, 0.08, -0.09, -0.06, 0.52, 0.68, 'gold', col=False)

    with module('ArmorStand'):
        box(-0.25, 0.25, -0.25, 0.25, 0, 0.06, 'wood', col=False, zm=False)
        box(-0.04, 0.04, -0.04, 0.04, 0.06, 1.55, 'wood', col=False)
        box(-0.25, 0.25, -0.14, 0.14, 0.95, 1.40, 'iron', col=False)
        box(-0.38, 0.38, -0.12, 0.12, 1.40, 1.52, 'iron', col=False)
        drum(0, 0, 0.15, 1.56, 1.86, 'iron', top='iron', bottom='iron', col=False)
        box(-0.3, 0.3, -0.25, 0.25, 0, 1.8, None, vis=False)

    with module('Anvil'):
        drum(0, 0, 0.32, 0, 0.55, 'wood', top='wood', col=False)
        box(-0.16, 0.16, -0.09, 0.09, 0.55, 0.68, 'iron', col=False)
        box(-0.30, 0.30, -0.11, 0.11, 0.68, 0.82, 'iron', col=False)
        box(0.30, 0.52, -0.06, 0.06, 0.74, 0.82, 'iron', col=False)
        box(-0.32, 0.32, -0.32, 0.32, 0, 0.82, None, vis=False)

    with module('Dummy'):
        box(-0.06, 0.06, -0.06, 0.06, 0, 1.75, 'wood', col=False, zm=False)
        box(-0.65, 0.65, -0.04, 0.04, 1.22, 1.30, 'wood', col=False)
        box(-0.22, 0.22, -0.15, 0.15, 0.85, 1.45, 'straw', col=False)
        box(-0.13, 0.13, -0.13, 0.13, 1.50, 1.76, 'straw', col=False)
        box(-0.22, 0.22, -0.15, 0.15, 0, 1.75, None, vis=False)

    # ── lighting / decoration ──
    with module('Torch'):
        # mounts on a wall at local Y=0, sticks out toward local -Y
        box(-0.06, 0.06, -0.03, 0, -0.15, 0.15, 'iron', col=False)
        box(-0.025, 0.025, -0.30, -0.03, -0.025, 0.025, 'iron', col=False)
        box(-0.04, 0.04, -0.34, -0.26, -0.12, 0.34, 'iron', col=False)
        pyramid(0, -0.30, 0.34, 0.74, 0.09, 'flame')

    with module('Fire'):
        for x, y in ((-0.3, 0.1), (0.3, -0.1)):
            box(x - 0.35, x + 0.35, y - 0.07, y + 0.07, 0, 0.14, 'wood', col=False)
        pyramid(0, 0, 0.1, 0.9, 0.22, 'flame')
        pyramid(-0.3, 0.05, 0.1, 0.6, 0.15, 'flame')
        pyramid(0.3, -0.05, 0.1, 0.65, 0.15, 'flame')

    with module('Brazier'):
        for a in (90, 210, 330):
            px, py, _ = polar(0, 0, 0.22, a)
            box(px - 0.03, px + 0.03, py - 0.03, py + 0.03, 0, 0.75, 'iron', col=False, zm=False)
        drum(0, 0, 0.36, 0.72, 0.90, 'iron', bottom='iron', top='ember', col=False)
        pyramid(0, 0, 0.9, 1.5, 0.2, 'flame')
        box(-0.3, 0.3, -0.3, 0.3, 0, 0.9, None, vis=False)

    with module('Chandelier'):
        ring(0, 0, 0.95, 0.82, 0, 0.08, 'iron', 'iron', top='iron', bottom='iron', col=False)
        box(-0.9, 0.9, -0.025, 0.025, 0.02, 0.07, 'iron', col=False)
        box(-0.025, 0.025, -0.9, 0.9, 0.02, 0.07, 'iron', col=False)
        box(-0.025, 0.025, -0.025, 0.025, 0.07, 2.5, 'iron', col=False)
        for a in range(0, 360, 45):
            px, py, _ = polar(0, 0, 0.88, a)
            box(px - 0.03, px + 0.03, py - 0.03, py + 0.03, 0.08, 0.24, 'linen', col=False, zm=False)
            pyramid(px, py, 0.24, 0.38, 0.03, 'flame')

    for name, cloth, art in (('BannerRed', 'cloth_red', 'banner_red'), ('BannerBlue', 'cloth_blue', 'banner_blue')):
        with module(name):
            # hangs down from z=0, flat against a wall at local Y=0, facing -Y
            box(-0.6, 0.6, -0.07, -0.01, -0.02, 0.05, 'wood', col=False)
            prof = [(-0.5, -0.045, 0), (-0.5, -0.045, -2.0), (0, -0.045, -2.4), (0.5, -0.045, -2.0), (0.5, -0.045, 0)]
            extrude(prof, (0, 0.02, 0), cloth, col=False)
            # painted side: its own image (art/banner_red.*, art/banner_blue.*), top of image at the rod
            front = [(-0.5, -2.0), (0, -2.4), (0.5, -2.0), (0.5, 0), (-0.5, 0)]
            emit([(x, -0.05, z) for x, z in front], art, col=False,
                 uv=[(x + 0.5, (z + 2.4) / 2.4) for x, z in front])

    for i in range(4):
        with module('Fresco%d' % (i + 1)):
            # 2.4 m framed panel, back against a wall at local Y=0, painted side faces -Y
            box(-1.2, -1.1, -0.08, 0, 0, 2.4, 'wood', col=False)
            box(1.1, 1.2, -0.08, 0, 0, 2.4, 'wood', col=False)
            box(-1.1, 1.1, -0.08, 0, 0, 0.1, 'wood', col=False)
            box(-1.1, 1.1, -0.08, 0, 2.3, 2.4, 'wood', col=False)
            # the picture has its own image: art/fresco1.* ... art/fresco4.*
            emit([(-1.1, -0.02, 0.1), (1.1, -0.02, 0.1), (1.1, -0.02, 2.3), (-1.1, -0.02, 2.3)], 'fresco%d' % (i + 1),
                 col=False, uv=[(0, 0), (1, 0), (1, 1), (0, 1)])

    with module('Tree'):
        drum(0, 0, 0.28, 0, 2.2, 'wood', col=False)
        cone(0, 0, 2.0, 1.6, 4.8, 'leaves', Rin=0.2, under='leaves')
        cone(0, 0, 1.4, 3.4, 6.2, 'leaves', Rin=0.2, under='leaves')
        box(-0.25, 0.25, -0.25, 0.25, 0, 2.2, None, vis=False)


# ══════════════════════════════════════════════════════════════════════
# Castle sections
# ══════════════════════════════════════════════════════════════════════
WZ = 7.0                      # wall-walk height
TR = 4.4                      # corner tower corner-radius
APO = TR * math.cos(math.radians(22.5))


def build_terrain():
    with section('Terrain'):
        # meadow around the moat
        for x0, x1, y0, y1 in ((-70, 70, -73, -33), (-70, 70, 41, 70), (-70, -35, -33, 41), (35, 70, -33, 41)):
            box(x0, x1, y0, y1, -3, 0, 'dirt', zp='grass', zm=False)
        # moat bed + water
        for x0, x1, y0, y1 in ((-35, 35, -33, -26), (-35, 35, 34, 41), (-35, -28, -26, 34), (28, 35, -26, 34)):
            box(x0, x1, y0, y1, -3, -2.5, 'dirt', zm=False)
            emit([(x0, y0, -0.9), (x1, y0, -0.9), (x1, y1, -0.9), (x0, y1, -0.9)], 'water', col=False)
        # low kerb so nobody strolls into the moat
        for x0, x1, y0, y1 in ((-35.4, -2.1, -33.4, -33), (2.1, 35.4, -33.4, -33), (-35.4, 35.4, 41, 41.4),
                               (-35.4, -35, -33, 41), (35, 35.4, -33, 41)):
            box(x0, x1, y0, y1, 0, 0.9, 'stone_wall', zm=False, zp='stone_floor')
        # approach road
        box(-2.2, 2.2, -73, -33.4, 0, 0.03, 'dirt', col=False, zm=False)
        # invisible edge of the world
        for x0, x1, y0, y1 in ((-70, 70, -73.5, -72.5), (-70, 70, 69.5, 70.5), (-70.5, -69.5, -73, 70), (69.5, 70.5, -73, 70)):
            box(x0, x1, y0, y1, 0, 4, None, vis=False)
        rnd = random.Random(7)
        spots = [(-14, -44), (-22, -52), (-9, -62), (12, -46), (20, -58), (8, -66), (-40, -40), (44, -44),
                 (-50, -10), (-46, 14), (-52, 36), (48, -8), (52, 20), (46, 44), (-30, 54), (-8, 58), (16, 52),
                 (34, 60), (-58, -50), (58, -56)]
        for x, y in spots:
            place('Tree', x, y, 0, rot=rnd.uniform(0, 360), scale=rnd.uniform(0.9, 1.5))


def build_curtain():
    with section('Curtain'):
        # walls (side = ashlar, top = flagstone wall-walk)
        for x0, x1, y0, y1 in ((-23.2, 23.2, 32, 34), (-23.2, -7, -26, -24), (7, 23.2, -26, -24),
                               (-28, -26, -21.2, 29.2), (26, 28, -21.2, 29.2)):
            box(x0, x1, y0, y1, -2.5, WZ, 'stone_wall', zp='stone_floor', zm=False)
        e = 27 - APO                                      # where walls meet tower faces
        battlements((e, 34), (-e, 34), WZ)                # north
        battlements((-e, -26), (-9.4, -26), WZ)           # south-west
        battlements((9.4, -26), (e, -26), WZ)             # south-east
        battlements((-28, 33 - APO), (-28, -25 + APO), WZ)  # west
        battlements((28, -25 + APO), (28, 33 - APO), WZ)  # east
        # inner kerbs (gaps at the stair landings)
        for x0, x1, y0, y1 in ((-e, e, 32, 32.25), (-e, -8.3, -24.25, -24), (8.3, e, -24.25, -24),
                               (-26.25, -26, -25 + APO, 33 - APO), (26, 26.25, -25 + APO, 33 - APO)):
            box(x0, x1, y0, y1, WZ, WZ + 0.9, 'stone_wall', zm=False)
        # three guard towers (the fourth corner is the tall tower)
        place('CornerTower', -27, -25, rot=0)
        place('CornerTower', 27, -25, rot=90)
        place('CornerTower', 27, 33, rot=180)
        place('Turret', -30.1, 4)
        place('Turret', 30.1, 4)
        # stairs up to the wall-walk, either side of the gatehouse
        stair(-20, -24, 0, 0, 39, WZ / 39, 0.3, 1.75, rail='L')
        box(-8.3, -7, -24, -22, 0, WZ, 'stone_wall', zp='stone_floor', zm=False)
        box(-8.3, -7, -22.25, -22, WZ, WZ + 0.95, 'stone_wall', zm=False)
        stair(20, -22.25, 0, 180, 39, WZ / 39, 0.3, 1.75, rail='R')
        box(7, 8.3, -24, -22, 0, WZ, 'stone_wall', zp='stone_floor', zm=False)
        box(7, 8.3, -22.25, -22, WZ, WZ + 0.95, 'stone_wall', zm=False)
        # torches and banners on the inner faces
        for x in (-16, 16):
            place('Torch', x, 32, 3.0, rot=0)
        for y in (-18, 22):
            place('Torch', 26, y, 3.0, rot=-90)
            place('Torch', -26, y, 3.0, rot=90)


def build_gatehouse():
    with section('Gatehouse'):
        for x0, x1 in ((-7, -2), (2, 7)):
            box(x0, x1, -28, -22, -2.5, WZ, 'stone_wall', zp='stone_floor', zm=False)
        box(-2, 2, -28, -22, 4.6, WZ, 'stone_wall', zp='stone_floor')
        box(-2, 2, -28, -24, -2.5, 0, 'stone_wall', zp='stone_floor', zm=False)   # passage floor
        place('Arch', 0, -27.72, 3.0, scale=(4, 0.6, 4), col=False)
        place('Arch', 0, -22.28, 3.0, scale=(4, 0.6, 4), col=False)
        place('Turret', -7, -28.3)
        place('Turret', 7, -28.3)
        battlements((-4.6, -28), (4.6, -28), WZ)
        box(-7, 7, -22.25, -22, WZ, WZ + 0.9, 'stone_wall', zm=False)
        # raised portcullis
        for i in range(9):
            x = -1.8 + i * 0.45
            box(x - 0.04, x + 0.04, -27.25, -27.17, 3.2, 4.6, 'iron', col=False)
        for z in (3.5, 4.1):
            box(-2, 2, -27.25, -27.17, z, z + 0.08, 'iron', col=False)
        # gates swung open against the inner face
        place('DoorLeaf', -2, -21.9, 0, rot=180, scale=(2, 1, 3.6))
        place('DoorLeaf', 2, -21.9, 0, rot=0, scale=(2, 1, 3.6))
        # drawbridge
        box(-1.9, 1.9, -33.6, -27.9, -0.22, 0.03, 'wood')
        for x0, x1 in ((-1.9, -1.7), (1.7, 1.9)):
            box(x0, x1, -33.6, -28, 0.03, 0.3, 'wood', col=False, zm=False)
        for x0, x1 in ((-2.05, -1.9), (1.9, 2.05)):
            box(x0, x1, -33.6, -28, -0.2, 1.3, None, vis=False)
        for x in (-1.8, 1.8):
            extrude([(x - 0.03, -33.3, 0.3), (x + 0.03, -33.3, 0.3), (x + 0.03, -33.3, 0.36), (x - 0.03, -33.3, 0.36)],
                    (0, 5.25, 4.6), 'iron', col=False)
        place('Torch', -2, -25, 2.6, rot=90)
        place('Torch', 2, -25, 2.6, rot=-90)
        place('BannerRed', -3.3, -28, 6.6)
        place('BannerBlue', 3.3, -28, 6.6)


def build_courtyard():
    with section('Courtyard'):
        holes = [(-7, 7, -11, 3),                    # fighting pit
                 (-25.2, -18, -9.2, 15.2),           # tavern cellar
                 (-25, -16.8, -11.6, -10),           # cellar stairwell
                 (-24.8, -23.3, -10, -9.2)]          # cellar doorway
        slab_with_holes(-26, 26, -24, 32, -5, 0, holes, 'stone_wall', 'stone_floor')

        # ── sunken fighting pit: 2 m terrace at -0.8, sand arena at -1.6 ──
        for x0, x1, y0, y1 in ((-7, 7, -11, -9), (-7, 7, 1, 3), (-7, -5, -9, 1), (5, 7, -9, 1)):
            box(x0, x1, y0, y1, -5, -0.8, 'stone_wall', zp='stone_floor', zm=False)
        box(-5, 5, -9, 1, -5, -1.6, 'stone_wall', zp='dirt', zm=False)
        stair(-1, -9.5, -0.8, -90, 5, 0.16, 0.3, 2.0)           # south: yard -> terrace
        stair(-1, -7.5, -1.6, -90, 5, 0.16, 0.3, 2.0)           # south: terrace -> arena
        stair(1, 1.5, -0.8, 90, 5, 0.16, 0.3, 2.0)              # north: yard -> terrace
        stair(1, -0.5, -1.6, 90, 5, 0.16, 0.3, 2.0)             # north: terrace -> arena
        for x, y in ((-6, -10), (6, -10), (-6, 2), (6, 2)):
            place('Brazier', x, y, -0.8)
        place('WeaponRack', -6.4, -4, -0.8, rot=90)
        place('WeaponRack', 6.4, -4, -0.8, rot=-90)
        for x, r in ((-8.6, 90), (8.6, 90)):
            place('Bench', x, -6.5, 0, rot=r)
            place('Bench', x, -1.5, 0, rot=r)
        place('Dummy', 11.5, -7.5, 0, rot=20)
        place('Dummy', 13.0, -3.5, 0, rot=-30)

        # ── well ──
        ring(10, 9, 1.25, 0.95, 0, 0.9, 'stone_wall', 'stone_wall', top='stone_floor')
        emit(octagon(10, 9, 0.95, 0.45), 'water', col=False)
        box(9.55, 10.45, 8.55, 9.45, 0, 0.9, None, vis=False)
        for x in (8.85, 11.05):
            box(x, x + 0.1, 8.95, 9.05, 0.9, 2.4, 'wood', col=False)
        box(8.8, 11.2, 8.93, 9.07, 2.3, 2.42, 'wood', col=False)
        box(9.9, 10.1, 8.9, 9.1, 1.2, 1.45, 'wood', col=False)
        box(9.99, 10.01, 8.99, 9.01, 1.45, 2.3, 'iron', col=False)

        # ── yard clutter ──
        for x, y in ((15.8, -8.6), (15.8, -7.7), (15.0, -8.4)):
            place('HayBale', x, y, 0, rot=90 if x > 15.5 else 10)
        place('HayBale', 15.8, -8.2, 0.6, rot=90)
        for x, y in ((16.3, 11.8), (16.3, 12.9), (12.6, 18.5), (-16.2, 17.6)):
            place('Barrel', x, y, 0)
        for x, y, r in ((13.6, 18.4, 12), (16.2, 14.2, 0), (-15.9, 16.4, 30)):
            place('Crate', x, y, 0, rot=r)
        place('Crate', 13.6, 18.4, 0.8, rot=40, scale=0.7)
        for x in (-17, -14.5):
            place('Bench', x, 18.6, 0)


def _hearth(xc, w, section_light=None):
    """Stone fireplace against the north wall of the north range (inner face y = 31.2)."""
    y = 31.2
    for x0, x1 in ((xc - w / 2, xc - w / 2 + 0.5), (xc + w / 2 - 0.5, xc + w / 2)):
        box(x0, x1, y - 0.9, y, 0.05, 2.0, 'stone_wall', zm=False)
    box(xc - w / 2, xc + w / 2, y - 0.9, y, 2.0, 2.5, 'stone_wall')
    box(xc - w / 2 + 0.2, xc + w / 2 - 0.2, y - 0.7, y, 2.5, 9.0, 'stone_wall', col=False)
    box(xc - 0.6, xc + 0.6, y - 1.5, y - 0.3, 9.0, 11.8, 'stone_wall', col=False)      # chimney
    box(xc - w / 2 + 0.5, xc + w / 2 - 0.5, y - 0.05, y, 0.05, 2.0, 'dark', col=False)
    box(xc - w / 2 + 0.5, xc + w / 2 - 0.5, y - 0.9, y, 0.05, 0.12, 'stone_floor', col=False, zm=False)
    box(xc - w / 2, xc + w / 2, y - 0.9, y, 0, 2.0, None, vis=False)
    place('Fire', xc, y - 0.45, 0.12, scale=min(1.3, w / 3.0))


def build_north_range():
    with section('NorthRange'):
        S, P = 'stone_wall', 'plaster'
        # south (courtyard) wall with windows and doors;  s = x + 22.6
        ops = []
        for xc in (-19.3, -14.3):
            ops.append((xc - 0.6 + 22.6, xc + 0.6 + 22.6, 1.4, 4.6, 'window'))
        for xc in (-8.2, -4.8, 4.8, 8.2):
            ops.append((xc - 0.6 + 22.6, xc + 0.6 + 22.6, 1.4, 5.0, 'window'))
        ops.append((-1.5 + 22.6, 1.5 + 22.6, 0, 4.0, 'arch'))            # great hall doors
        ops.append((14.2 + 22.6, 15.8 + 22.6, 0, 2.6, ''))               # kitchen door
        ops.append((18.6 + 22.6, 19.8 + 22.6, 1.4, 3.6, 'window'))
        wall((-22.6, 19.6), (22.6, 19.6), 0, 8, 0.8, P, S, ops, jamb=S)
        wall((-22.6, 31.6), (22.6, 31.6), 0, 8, 0.8, S, P, jamb=S)
        wall((-22.2, 20), (-22.2, 31.2), 0, 8, 0.8, S, P, [(8.7, 10.2, 0, 2.6, '')], jamb=S)     # to tower alley
        wall((22.2, 20), (22.2, 31.2), 0, 8, 0.8, P, S, [(4.0, 5.5, 0, 2.6, '')], jamb=S)        # kitchen back door
        wall((-11.4, 20), (-11.4, 31.2), 0, 8, 0.8, P, P, [(4.4, 6.8, 0, 3.4, 'arch')])          # hall | throne
        wall((11.4, 20), (11.4, 31.2), 0, 8, 0.8, P, P, [(2.0, 3.6, 0, 2.6, '')])                # hall | kitchen
        # gables
        for x0, front, back in ((-22.6, S, P), (-11.8, P, P), (11.0, P, P), (21.8, P, S)):
            extrude([(x0, 19.2, 8), (x0, 32, 8), (x0, 25.6, 12.5)], (0.8, 0, 0), front, back, S, col=False)
        # roof: two slabs (tiles above, boards below) + ridge
        k = 4.5 / 6.4
        zs = 8 + (18.5 - 19.2) * k
        nrm = np.array([0, -4.5, 6.4]) / math.hypot(4.5, 6.4) * 0.25
        extrude([(-23, 18.5, zs), (23, 18.5, zs), (23, 25.6, 12.5), (-23, 25.6, 12.5)], nrm, 'wood', 'roof', 'wood', col=False)
        nrm = np.array([0, 4.5, 6.4]) / math.hypot(4.5, 6.4) * 0.25
        extrude([(-23, 25.6, 12.5), (23, 25.6, 12.5), (23, 32, 8), (-23, 32, 8)], nrm, 'wood', 'roof', 'wood', col=False)
        box(-23, 23, 25.4, 25.8, 12.55, 12.9, 'roof', col=False)
        box(-21.8, 21.8, 20, 31.2, -0.1, 0.05, 'stone_floor', zm=False)
        for x in (-18.6, -15.0, -7.4, -3.7, 0.0, 3.7, 7.4, 15.0, 18.6):
            place('Truss', x, 20, 8.0, rot=90, col=False)
        place('Torch', -2.4, 19.2, 2.9)
        place('Torch', 2.4, 19.2, 2.9)
        place('BannerRed', -3.6, 19.2, 7.4)
        place('BannerBlue', 3.6, 19.2, 7.4)

        # ── great hall ──
        _hearth(0, 3.6)
        for i, x in enumerate((-8.4, -5.2, 5.2, 8.4)):
            place('Fresco%d' % (i + 1), x, 31.2, 1.9)
        place('Fresco2', -11.0, 22.3, 2.0, rot=90, scale=0.85)
        place('Fresco3', -11.0, 29.0, 2.0, rot=90, scale=0.85)
        place('Fresco1', 11.0, 27.4, 2.0, rot=-90, scale=0.95)
        for y in (23.2, 28.0):
            for x in (-7.25, -3.75, 3.75, 7.25):
                place('TableLong', x, y, 0.05)
                place('Bench', x, y - 0.9, 0.05)
                place('Bench', x, y + 0.9, 0.05)
        for x in (-3.7, 3.7):
            place('Chandelier', x, 25.6, 5.6, col=False)
            light('HallChandelier', x, 25.6, 5.2, 34)
        for x in (-6.5, -3.0, 3.0, 6.5):
            place('Torch', x, 20, 2.8, rot=180)
        place('DoorLeaf', -1.5, 20.0, 0.05, rot=90, scale=(1.5, 1, 2.8))
        place('DoorLeaf', 1.5, 20.0, 0.05, rot=90, scale=(1.5, 1, 2.8))

        # ── throne room ──
        box(-21.8, -18.8, 22.8, 28.4, 0.05, 0.25, 'stone_wall', zp='stone_floor', zm=False)
        box(-21.8, -19.6, 23.6, 27.6, 0.25, 0.45, 'stone_wall', zp='stone_floor', zm=False)
        place('Throne', -20.7, 25.6, 0.45, rot=90, scale=1.15)
        box(-18.8, -11.9, 25.0, 26.2, 0.05, 0.07, 'cloth_red', col=False, zm=False)
        place('BannerRed', -21.8, 23.6, 6.0, rot=90)
        place('BannerBlue', -21.8, 27.6, 6.0, rot=90)
        place('Fresco4', -16.6, 31.2, 1.9)
        place('Bench', -15.5, 20.4, 0.05)
        place('Bench', -14.5, 30.8, 0.05, scale=(0.8, 1, 1))
        for x in (-19.6, -13.6):
            place('Torch', x, 31.2, 2.8)
        light('ThroneLight', -16.6, 25.6, 4.6, 30)

        # ── kitchen ──
        _hearth(16.8, 4.4)
        place('TableLong', 16.8, 25.3, 0.05)
        box(16.4, 21.7, 20.05, 20.85, 0.05, 0.95, 'wood', zm=False)          # counter under the window
        for x, y in ((21.2, 30.6), (21.2, 29.7), (20.3, 30.7)):
            place('Barrel', x, y, 0.05)
        place('Crate', 12.4, 30.6, 0.05)
        place('Crate', 12.4, 29.7, 0.05, rot=8)
        place('Crate', 12.4, 30.2, 0.85, rot=20, scale=0.75)
        place('Torch', 13.0, 20, 2.6, rot=180)
        light('KitchenHearth', 16.8, 29.4, 2.2, 26)


def _timber(xf_face, sign, ys, y_lo, y_hi):
    """Half-timber strips on a plaster courtyard face at x = xf_face (sign = outward direction)."""
    x0, x1 = sorted((xf_face, xf_face + sign * 0.06))
    box(x0, x1, y_lo, y_hi, 3.72, 3.98, 'wood', col=False)
    box(x0, x1, y_lo, y_hi, 0, 0.22, 'wood', col=False, zm=False)
    for y in ys:
        box(x0, x1, y - 0.1, y + 0.1, 0.22, 3.72, 'wood', col=False)


def build_east_range():
    with section('EastRange'):
        P = 'plaster'
        k = 2.6 / 8.8                                # lean-to slope
        # courtyard wall;  s = y + 9.2
        ops = [(-6.6 + 9.2, -0.6 + 9.2, 0, 3.2, ''),               # open forge front
               (5.7 + 9.2, 6.7 + 9.2, 1.3, 2.9, 'window'),
               (8.8 + 9.2, 10.4 + 9.2, 0, 2.6, ''),                # armory door
               (12.5 + 9.2, 13.5 + 9.2, 1.3, 2.9, 'window')]
        wall((17.6, -9.2), (17.6, 15.2), 0, 4.0, 0.8, P, P, ops)
        wall((25.6, -9.2), (25.6, 15.2), 0, 6.3, 0.8, P, P)
        for y in (-10.0, 15.2):
            extrude([(17.2, y, 0), (26, y, 0), (26, y, 6.4), (17.2, y, 3.8)], (0, 0.8, 0), P)
        wall((18, 3.6), (25.2, 3.6), 0, 3.8, 0.8, P, P, [(2.8, 4.4, 0, 2.6, '')])
        extrude([(18, 3.2, 3.8), (25.2, 3.2, 3.8), (25.2, 3.2, 6.16), (18, 3.2, 4.04)], (0, 0.8, 0), P, col=False)
        zl = 3.8 + (16.6 - 17.2) * k
        nrm = np.array([-2.6, 0, 8.8]) / math.hypot(2.6, 8.8) * 0.25
        extrude([(16.6, -10.4, zl), (26, -10.4, 6.4), (26, 16.4, 6.4), (16.6, 16.4, zl)], nrm, 'wood', 'roof', 'wood', col=False)
        box(18, 25.2, 4.0, 15.2, -0.1, 0.05, 'stone_floor', zm=False)
        box(18, 25.2, -9.2, 3.2, -0.1, 0.05, 'dirt', zm=False)
        _timber(17.2, -1, (-10.1, -6.7, -0.5, 3.6, 8.7, 10.5, 16.1), -10.2, 16.2)
        box(17.1, 18.05, -6.9, -0.3, 3.08, 3.5, 'wood', col=False)             # forge lintel beam (hangs below the plaster soffit)

        # ── forge ──
        box(23.4, 25.2, -5.2, -2.0, 0.05, 1.0, 'stone_wall', zm=False)
        box(23.6, 25.0, -5.0, -2.2, 1.0, 1.05, 'ember', col=False, zm=False)
        box(23.7, 25.2, -4.8, -2.4, 2.4, 6.3, 'stone_wall', col=False)         # hood
        box(24.0, 25.2, -4.4, -2.8, 6.3, 8.8, 'stone_wall', col=False)         # chimney
        place('Anvil', 21.3, -3.6, 0.05, rot=90)
        box(18.5, 20.1, -8.9, -8.1, 0.05, 0.7, 'wood', zp='water', zm=False)   # quench trough
        place('TableLong', 22.7, -8.5, 0.05)
        place('WeaponRack', 24.9, 1.2, 0.05, rot=-90)
        place('Barrel', 18.7, 2.5, 0.05)
        place('Barrel', 18.7, 1.5, 0.05)
        place('Torch', 21.6, -9.2, 2.4, rot=180)
        light('ForgeGlow', 22.6, -3.6, 2.0, 30, (1.0, 0.55, 0.25))

        # ── armory ──
        for y in (6.3, 9.6, 12.9):
            place('WeaponRack', 24.9, y, 0.05, rot=-90)
        for i, x in enumerate((19.6, 21.6, 23.6)):
            place('ShieldRed' if i % 2 == 0 else 'ShieldBlue', x, 15.2, 1.7)
        for x in (18.8, 19.8, 23.3, 24.3):
            place('ArmorStand', x, 4.6, 0.05)
        place('Chest', 18.5, 14.5, 0.05, rot=90)
        place('Chest', 18.5, 12.9, 0.05, rot=90)
        place('TableLong', 21.2, 9.6, 0.05)
        place('Torch', 18, 11.6, 2.4, rot=90)
        light('ArmoryLight', 21.4, 9.6, 3.0, 26)


def build_west_range():
    with section('WestRange'):
        P = 'plaster'
        k = 2.6 / 8.8
        rooms = (-9.2, -3.0, 3.2, 9.4)               # south edge of each 5.8 m guest room
        ops = []
        for y0 in rooms:
            ops.append((y0 + 0.6 + 9.2, y0 + 1.9 + 9.2, 0, 2.5, ''))
            ops.append((y0 + 3.5 + 9.2, y0 + 4.5 + 9.2, 1.2, 2.8, 'window'))
        wall((-17.6, -9.2), (-17.6, 15.2), 0, 4.0, 0.8, P, P, ops)
        wall((-25.6, -9.2), (-25.6, 15.2), 0, 6.3, 0.8, P, P)
        for y in (-10.0, 15.2):
            extrude([(-26, y, 0), (-17.2, y, 0), (-17.2, y, 3.8), (-26, y, 6.4)], (0, 0.8, 0), P)
        for y in (-3.4, 2.8, 9.0):
            extrude([(-25.2, y, 0), (-18, y, 0), (-18, y, 4.04), (-25.2, y, 6.16)], (0, 0.4, 0), P)
        zl = 3.8 + (16.6 - 17.2) * k
        nrm = np.array([2.6, 0, 8.8]) / math.hypot(2.6, 8.8) * 0.25
        extrude([(-26, -10.4, 6.4), (-16.6, -10.4, zl), (-16.6, 16.4, zl), (-26, 16.4, 6.4)], nrm, 'wood', 'roof', 'wood', col=False)
        box(-25.2, -18, -9.2, 15.2, -0.4, 0.05, 'wood')            # guest floor = cellar ceiling
        ys = [-10.1, 16.1]
        for y0 in rooms:
            ys += [y0 + 0.5, y0 + 2.0]
        _timber(-17.2, 1, ys, -10.2, 16.2)

        # ── guest rooms ──
        for i, y0 in enumerate(rooms):
            place('Bed', -24.15, y0 + 3.0, 0.05)
            place('Bed', -24.15, y0 + 4.9, 0.05)
            place('Chest', -22.6, y0 + 4.9, 0.05, rot=90)
            place('SmallTable', -18.6, y0 + 4.0, 0.05)
            place('Stool', -19.5, y0 + 4.0, 0.05)
            box(-22.0, -19.6, y0 + 1.6, y0 + 3.2, 0.05, 0.065,
                'cloth_red' if i % 2 == 0 else 'cloth_blue', col=False, zm=False)
            place('Torch', -25.2, y0 + 1.2, 2.3, rot=90)

        # ── tavern cellar (floor at -3.6) ──
        box(-25.2, -18, -9.2, 15.2, -5, -3.6, 'stone_wall', zp='stone_floor', zm=False)
        box(-25, -22.9, -11.6, -10, -5, -3.6, 'stone_wall', zp='stone_floor', zm=False)       # stair landing
        box(-24.8, -23.3, -10, -9.2, -5, -3.6, 'stone_wall', zp='stone_floor', zm=False)      # door sill
        box(-24.8, -23.3, -10, -9.2, -1.2, 0, 'stone_wall', zp='stone_floor')                 # door lintel
        stair(-22.9, -11.6, -3.6, 0, 20, 0.18, 0.305, 1.6, base=-5)
        box(-25.25, -16.8, -11.85, -11.6, 0, 0.9, 'stone_wall', zm=False)
        box(-25.25, -25, -11.6, -10, 0, 0.9, 'stone_wall', zm=False)
        for y in np.linspace(-7.2, 13.2, 9):
            box(-25.2, -18, y - 0.12, y + 0.12, -0.7, -0.4, 'wood', col=False)
        box(-24.4, -19.6, 12.2, 12.9, -3.6, -2.5, 'wood', zm=False)                          # bar
        for x in (-24.4, -23.4, -22.4, -21.4, -20.4):
            place('Barrel', x, 14.5, -3.6)
        for x in (-23.9, -21.9):
            place('Barrel', x, 14.5, -2.6, scale=0.9)
        for y in (-5.5, 0.5, 6.5):
            place('RoundTable', -20.6, y, -3.6)
            for a in (0, 90, 180, 270):
                px, py, _ = polar(-20.6, y, 1.05, a + 45)
                place('Stool', px, py, -3.6)
        for y in (-3.6, -2.5, -1.4, 2.6, 3.7, 8.6, 9.7):
            place('Barrel', -24.6, y, -3.6)
        place('Crate', -18.6, -8.6, -3.6)
        place('Crate', -18.6, -7.7, -3.6, rot=12)
        place('Crate', -18.6, -8.2, -2.8, rot=30, scale=0.75)
        for y in (-6, 4.6, 11):
            place('Torch', -25.2, y, -1.6, rot=90)
        for y in (-3, 3.5, 9.5):
            place('Torch', -18, y, -1.6, rot=-90)
        light('CellarLight', -21.6, -3.0, -1.2, 26)
        light('CellarLight', -21.6, 8.6, -1.2, 26)


def build_tall_tower():
    """NW corner: 20 m tower, spiral stair (4.75 turns), landing with windows over the courtyard."""
    cx, cy = -27.0, 33.0
    Ri = (APO - 0.8) / math.cos(math.radians(22.5))
    rise = 3.5 / 16                    # 16 treads per turn, 3.5 m per turn
    N = 76                             # treads -> landing at 16.625 m
    th0 = 258.75                       # stair starts here and climbs clockwise
    top = N * rise
    with section('TallTower'):
        ring(cx, cy, TR, Ri, -2.5, 20.1, 'stone_wall', 'stone_wall', openings={
            7: [(1.5, 0.1, 2.6, 0.0), (1.5, top + 1.0, top + 2.5, 0.0)],     # SE: ground door + top window
            6: [(1.5, 7.0, 9.5, 0.15), (1.5, top + 1.0, top + 2.5, 0.15)],   # S: wall-walk door + top window
        })
        emit(octagon(cx, cy, Ri, 0.1), 'stone_floor')
        emit(octagon(cx, cy, Ri, 20.0)[::-1], 'wood', col=False)
        cone(cx, cy, 5.1, 19.9, 26.8, 'roof', Rin=TR - 0.1)
        drum(cx, cy, 0.8, 0, top + 1.0, 'stone_wall', top='stone_floor')     # newel
        r0, r1 = 0.75, 3.62
        for j in range(N):
            hi, lo = th0 - 22.5 * j, th0 - 22.5 * (j + 1)
            z = (j + 1) * rise
            tread = [polar(cx, cy, r0, lo, z - 0.3), polar(cx, cy, r1, lo, z - 0.3),
                     polar(cx, cy, r1, hi, z - 0.3), polar(cx, cy, r0, hi, z - 0.3)]
            extrude(tread, (0, 0, 0.3), 'stone_wall', 'stone_floor', 'stone_wall', col=False)
            # collision: a smooth helical ramp through the tread centres
            za, zb = (j + 0.5) * rise, (j + 1.5) * rise
            A, B = polar(cx, cy, r0, lo, zb), polar(cx, cy, r1, lo, zb)
            C, D = polar(cx, cy, r1, hi, za), polar(cx, cy, r0, hi, za)
            emit([A, B, C], None, vis=False)
            emit([A, C, D], None, vis=False)
        end = th0 - 22.5 * N
        for m in range(5):                                                   # top landing
            hi, lo = end - 22.5 * m, end - 22.5 * (m + 1)
            pts = [polar(cx, cy, r0, lo, top - 0.3), polar(cx, cy, r1, lo, top - 0.3),
                   polar(cx, cy, r1, hi, top - 0.3), polar(cx, cy, r0, hi, top - 0.3)]
            extrude(pts, (0, 0, 0.3), 'wood', 'stone_floor', 'stone_wall')
        with xf(cx, cy, 0, rot=end - 112.5):                                 # rail at the landing's open end
            box(0.75, 3.5, -0.08, 0.08, top, top + 1.0, 'wood')
        with xf(cx, cy, 0, rot=22.5):                                        # closes the low space under the stair
            box(0.75, 3.5, -0.1, 0.1, 0.1, 2.2, 'stone_wall')
        for k, zs in ((0, (11.0,)), (1, (5.0, 13.0)), (2, (9.0, 16.5)), (3, (4.0, 12.0, 17.5)),
                      (4, (8.0, 15.0)), (5, (5.5, 13.5))):
            for z in zs:
                with xf(cx, cy, 0, rot=45 * k):
                    box(APO - 0.05, APO + 0.03, -0.12, 0.12, z, z + 1.4, 'dark', col=False)
        tx, ty, _ = polar(cx, cy, Ri * math.cos(math.radians(22.5)) - 0.02, 180)
        place('Torch', tx, ty, 1.9, rot=90)
        tx, ty, _ = polar(cx, cy, Ri * math.cos(math.radians(22.5)) - 0.02, 135)
        place('Torch', tx, ty, top + 1.9, rot=45)
        place('BannerRed', *polar(cx, cy, APO + 0.02, 315, 15.0), rot=45)


def build_all():
    define_modules()
    build_terrain()
    build_curtain()
    build_gatehouse()
    build_courtyard()
    build_north_range()
    build_east_range()
    build_west_range()
    build_tall_tower()


# ══════════════════════════════════════════════════════════════════════
# Blender scene assembly + export
# ══════════════════════════════════════════════════════════════════════
def make_materials(tex):
    def img(name, non_color=False):
        im = bpy.data.images.load(tex[name])
        im.name = 'castle_' + name
        if non_color:
            im.colorspace_settings.name = 'Non-Color'
        im.pack()
        return im

    def mat(name, color=(0.8, 0.8, 0.8), texture=None, normal=None, rough=0.85, metal=0.0, emit=None, strength=0.0):
        m = bpy.data.materials.new(name if name.startswith('Art_') else 'Castle_' + name)
        if not m.node_tree or bpy.app.version < (5, 0, 0):
            m.use_nodes = True
        m.use_backface_culling = True
        nt = m.node_tree
        bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
        bsdf.inputs['Base Color'].default_value = (*color, 1.0)
        bsdf.inputs['Roughness'].default_value = rough
        bsdf.inputs['Metallic'].default_value = metal
        if texture:
            t = nt.nodes.new('ShaderNodeTexImage')
            t.image = img(texture)
            t.location = (-400, 300)
            nt.links.new(t.outputs['Color'], bsdf.inputs['Base Color'])
        if normal:
            t = nt.nodes.new('ShaderNodeTexImage')
            t.image = img(normal, True)
            t.location = (-700, -100)
            nm = nt.nodes.new('ShaderNodeNormalMap')
            nm.location = (-350, -100)
            nt.links.new(t.outputs['Color'], nm.inputs['Color'])
            nt.links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
        if emit:
            key = 'Emission Color' if 'Emission Color' in bsdf.inputs else 'Emission'
            bsdf.inputs[key].default_value = (*emit, 1.0)
            bsdf.inputs['Emission Strength'].default_value = strength
        m.diffuse_color = (*color, 1.0)
        return m

    mats = {name: mat('Art_' + name, texture=name, rough=0.9) for name in castle_textures.ART}
    mats.update({
        'stone_wall': mat('StoneWall', texture='stone_wall', normal='stone_wall_n', rough=0.9),
        'stone_floor': mat('StoneFloor', texture='stone_floor', normal='stone_floor_n', rough=0.85),
        'wood': mat('Wood', texture='wood', rough=0.75),
        'plaster': mat('Plaster', texture='plaster', rough=0.95),
        'roof': mat('RoofTiles', texture='roof', normal='roof_n', rough=0.8),
        'grass': mat('Grass', texture='grass', rough=1.0),
        'dirt': mat('Dirt', texture='dirt', rough=1.0),
        'iron': mat('Iron', (0.10, 0.10, 0.11), rough=0.55, metal=0.6),
        'gold': mat('Gold', (0.83, 0.62, 0.22), rough=0.4, metal=0.8),
        'cloth_red': mat('ClothRed', (0.50, 0.07, 0.07), rough=0.95),
        'cloth_blue': mat('ClothBlue', (0.09, 0.17, 0.42), rough=0.95),
        'linen': mat('Linen', (0.82, 0.77, 0.66), rough=0.95),
        'straw': mat('Straw', (0.72, 0.58, 0.26), rough=1.0),
        'leaves': mat('Leaves', (0.13, 0.30, 0.11), rough=1.0),
        'water': mat('Water', (0.07, 0.19, 0.22), rough=0.08),
        'dark': mat('Dark', (0.015, 0.015, 0.02), rough=1.0),
        'flame': mat('Flame', (1.0, 0.40, 0.05), emit=(1.0, 0.30, 0.03), strength=1.8),
        'ember': mat('Ember', (0.45, 0.10, 0.03), emit=(1.0, 0.16, 0.02), strength=1.2),
        'collision': mat('Collision', (1.0, 0.0, 1.0)),
    })
    return mats


def assemble(materials):
    scene = bpy.context.scene
    root = bpy.data.collections.new('ClubhouseCastle')
    scene.collection.children.link(root)
    col_coll = bpy.data.collections.new('Collision')
    root.children.link(col_coll)
    colls = {}
    stats = {'vis': 0, 'col': 0, 'inst': 0}

    def coll_for(name):
        if name not in colls:
            colls[name] = bpy.data.collections.new(name)
            root.children.link(colls[name])
        return colls[name]

    for name, (vb, cb) in SECTIONS.items():
        ob = bpy.data.objects.new(name, vb.to_mesh(name, materials))
        coll_for(name).objects.link(ob)
        stats['vis'] += vb.tri_count()
        if cb.faces:
            co = bpy.data.objects.new('COL_' + name, cb.to_mesh('COL_' + name, materials))
            co.display_type = 'WIRE'
            col_coll.objects.link(co)
            stats['col'] += cb.tri_count()

    meshes = {name: vb.to_mesh(name, materials) for name, (vb, _) in MODULES.items()}
    for name, M, sect in INSTANCES:
        ob = bpy.data.objects.new(name, meshes[name])
        ob.matrix_world = Matrix(M.tolist())
        coll_for(sect).objects.link(ob)
        stats['vis'] += MODULES[name][0].tri_count()
        stats['inst'] += 1

    for name, loc, color, power, sect in LIGHTS:
        ld = bpy.data.lights.new(name, 'POINT')
        ld.energy = power
        ld.color = color
        ld.shadow_soft_size = 0.15
        ob = bpy.data.objects.new(name, ld)
        ob.location = loc
        coll_for(sect).objects.link(ob)

    spawn = bpy.data.objects.new('Spawn', None)
    spawn.empty_display_type = 'ARROWS'
    spawn.location = (0.0, -52.0, 0.0)
    root.objects.link(spawn)
    return stats


ART_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'art') \
    if '__file__' in globals() else os.path.abspath('art')


def collect_art(tex_dir, size):
    """Wall art comes from the repo's art/ folder. Missing pieces get a generated default
    (saved as JPEG so the folder stays small); files already there are never overwritten."""
    os.makedirs(ART_DIR, exist_ok=True)
    found = {}
    for name, make in castle_textures.ART.items():
        path = castle_textures.find_art(ART_DIR, name)
        if not path:
            tmp = os.path.join(tex_dir, name + '.png')
            castle_textures.write_png(tmp, make(size))
            im = bpy.data.images.load(tmp)
            path = os.path.join(ART_DIR, name + '.jpg')
            bpy.context.scene.render.image_settings.file_format = 'JPEG'
            bpy.context.scene.render.image_settings.quality = 88
            im.save_render(path)
            bpy.data.images.remove(im)
            print('  wrote default', path)
        found[name] = path
    return found


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='ClubhouseCastle.glb')
    ap.add_argument('--blend', default=None, help='also save a .blend here')
    ap.add_argument('--tex', default=None, help='directory for generated texture PNGs')
    ap.add_argument('--size', type=int, default=1024, help='texture size (1024 or 2048)')
    ap.add_argument('--quality', type=int, default=80, help='WebP quality for embedded textures')
    args, _ = ap.parse_known_args(argv)

    if bpy.app.background:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    tex_dir = args.tex or os.path.join(tempfile.gettempdir(), 'clubhouse_castle_textures')
    print('Generating textures in', tex_dir)
    tex = castle_textures.generate_all(tex_dir, args.size)
    tex.update(collect_art(tex_dir, args.size))
    build_all()
    stats = assemble(make_materials(tex))
    print('Visible triangles: %(vis)d   collision triangles: %(col)d   instances: %(inst)d' % stats)
    print('Unique meshes: %d sections + %d modules, %d lights' % (len(SECTIONS), len(MODULES), len(LIGHTS)))

    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.blend))
    out = os.path.abspath(args.out)
    kw = dict(filepath=out, export_format='GLB', export_yup=True, export_apply=False, export_cameras=False,
              export_lights=True, export_animations=False, export_image_format='WEBP',
              export_image_quality=args.quality, export_import_convert_lighting_mode='RAW')
    try:
        bpy.ops.export_scene.gltf(**kw)
    except TypeError:
        kw.pop('export_import_convert_lighting_mode')
        bpy.ops.export_scene.gltf(**kw)
    print('Wrote %s (%.2f MB)' % (out, os.path.getsize(out) / 1e6))


if __name__ == '__main__':
    main()
