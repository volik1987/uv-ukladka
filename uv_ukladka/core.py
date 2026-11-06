"""Работа с мешами Blender: поиск UV-островов, запись, все операции аддона.

Функции этого модуля вызываются и операторами, и скриптовым API (api.py).
"""
import math
import os
import time

import bmesh
import bpy
import numpy as np

from . import packer

UV_EPS = 1e-5


class Island:
    __slots__ = ("md", "faces", "loops", "pts", "tris", "co", "area3d", "selected",
                 "pinned", "index")

    def __init__(self, md):
        self.md = md

    @property
    def area_uv(self):
        return packer.tri_area(self.pts, self.tris)

    def bounds(self):
        return self.pts.min(0), self.pts.max(0)

    def center(self):
        mn, mx = self.bounds()
        return (mn + mx) / 2

    def transform(self, M, t):
        self.pts = self.pts @ np.asarray(M).T + t

    def rotate(self, a, pivot=None):
        c = self.center() if pivot is None else pivot
        self.pts = (self.pts - c) @ packer.rot(a).T + c

    def scale(self, s, pivot=None):
        c = self.center() if pivot is None else pivot
        self.pts = (self.pts - c) * s + c

    def texel(self, tex_size):
        """Плотность текселей, пикселей на метр."""
        if self.area3d <= 0:
            return 0.0
        return math.sqrt(max(self.area_uv, 0) / self.area3d) * tex_size


class MeshData:
    def __init__(self, obj, bm, uv, edit):
        self.obj, self.bm, self.uv, self.edit = obj, bm, uv, edit
        self.islands = []

    def write(self, islands=None):
        uv = self.uv
        for isl in (self.islands if islands is None else islands):
            for l, p in zip(isl.loops, isl.pts.tolist()):
                l[uv].uv = p

    def commit(self):
        me = self.obj.data
        if self.edit:
            bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
        else:
            self.bm.to_mesh(me)
            me.update()
            self.bm.free()

    def free(self):
        if not self.edit:
            self.bm.free()


def _uv_sync(context):
    return context.scene.tool_settings.use_uv_select_sync


def _objects(context):
    if context.mode == "EDIT_MESH":
        objs = list(context.objects_in_mode_unique_data)
    else:
        seen, objs = set(), []
        for o in context.selected_objects:
            if o.type == "MESH" and o.data not in seen:
                seen.add(o.data)
                objs.append(o)
    return [o for o in objs if o.type == "MESH"]


def gather(context, need_3d=True):
    """Все видимые острова всех редактируемых объектов (список MeshData)."""
    edit = context.mode == "EDIT_MESH"
    sync = _uv_sync(context)
    out = []
    for obj in _objects(context):
        me = obj.data
        if edit:
            bm = bmesh.from_edit_mesh(me)
        else:
            bm = bmesh.new()
            bm.from_mesh(me)
        uv = bm.loops.layers.uv.active
        if uv is None:
            if not edit:
                bm.free()
            continue
        md = MeshData(obj, bm, uv, edit)
        _find_islands(md, edit, sync, need_3d)
        out.append(md)
    return out


def _find_islands(md, edit, sync, need_3d):
    bm, uv = md.bm, md.uv
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    if edit:
        vis = [(not f.hide) and (sync or f.select) for f in bm.faces]
    else:
        vis = [True] * len(bm.faces)
    parent = list(range(len(bm.faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    eps2 = UV_EPS * UV_EPS
    for e in bm.edges:
        ll = e.link_loops
        if len(ll) < 2:
            continue
        for ia in range(len(ll)):
            la = ll[ia]
            fa = la.face.index
            if not vis[fa]:
                continue
            for ib in range(ia + 1, len(ll)):
                lb = ll[ib]
                fb = lb.face.index
                if not vis[fb]:
                    continue
                if lb.vert is la.vert:
                    a2, b2 = lb, lb.link_loop_next
                else:
                    a2, b2 = lb.link_loop_next, lb
                p1, p2 = la[uv].uv, a2[uv].uv
                q1, q2 = la.link_loop_next[uv].uv, b2[uv].uv
                if ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 < eps2
                        and (q1.x - q2.x) ** 2 + (q1.y - q2.y) ** 2 < eps2):
                    ra, rb = find(fa), find(fb)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)

    groups = {}
    for f in bm.faces:
        if vis[f.index]:
            groups.setdefault(find(f.index), []).append(f)

    ltris = {}
    for t in bm.calc_loop_triangles():
        ltris.setdefault(t[0].face.index, []).append(t)

    mw = np.array(md.obj.matrix_world)
    for faces in groups.values():
        isl = Island(md)
        isl.faces = faces
        loops = []
        lidx = {}
        for f in faces:
            for l in f.loops:
                lidx[l] = len(loops)
                loops.append(l)
        tris = []
        for f in faces:
            for t in ltris.get(f.index, ()):
                tris.append((lidx[t[0]], lidx[t[1]], lidx[t[2]]))
        isl.loops = loops
        isl.tris = np.array(tris, np.int64).reshape(-1, 3)
        isl.pts = np.array([l[uv].uv[:] for l in loops], np.float64).reshape(-1, 2)
        if need_3d:
            co = np.array([l.vert.co[:] for l in loops], np.float64).reshape(-1, 3)
            co = co @ mw[:3, :3].T + mw[:3, 3]
            isl.area3d = packer.tri_area3d(co, isl.tris)
        else:
            isl.area3d = 0.0
        if not edit:
            isl.selected = True
        elif sync:
            isl.selected = any(f.select for f in faces)
        else:
            isl.selected = any(l.uv_select_vert for l in loops)
        isl.pinned = any(l[uv].pin_uv for l in loops)
        md.islands.append(isl)


def all_islands(meshes):
    return [i for md in meshes for i in md.islands]


def finish(meshes, changed=True):
    for md in meshes:
        if changed:
            md.write()
            md.commit()
        else:
            md.free()


def target_islands(meshes, selected_only):
    isl = all_islands(meshes)
    if selected_only:
        isl = [i for i in isl if i.selected]
    return isl


def select_islands(context, meshes, islands, deselect_others=True):
    sync = _uv_sync(context)
    wanted = {id(i) for i in islands}
    for md in meshes:
        if sync:
            # сначала снять, потом выделить: иначе снятие соседнего острова
            # снимает общие вершины уже выделенного
            if deselect_others:
                for isl in md.islands:
                    if id(isl) not in wanted:
                        for f in isl.faces:
                            f.select_set(False)
            for isl in md.islands:
                if id(isl) in wanted:
                    for f in isl.faces:
                        f.select_set(True)
        for isl in md.islands if not sync else ():
            on = id(isl) in wanted
            if on or deselect_others:
                for l in isl.loops:
                    l.uv_select_vert = on
                    l.uv_select_edge = on
                for f in isl.faces:
                    if hasattr(f, "uv_select"):
                        f.uv_select = on
        if md.edit:
            bmesh.update_edit_mesh(md.obj.data, loop_triangles=False, destructive=False)


# ----------------------------------------------------------------- упаковка

def udim_offset(tile):
    t = max(1001, int(tile)) - 1001
    return np.array([t % 10, t // 10], float)


def find_stacks(islands, tol=0.002, allow_reflect=True):
    """Группы одинаковых островов: {мастер: [копии]}. Совпадение — по порядку
    углов (копии из массивов/дубликатов) с точностью до поворота и сдвига."""
    groups = {}
    for isl in islands:
        if len(isl.pts) < 3:
            continue
        key = (len(isl.loops), len(isl.tris), len(isl.faces))
        groups.setdefault(key, []).append(isl)
    stacks = {}
    for members in groups.values():
        masters = []
        for isl in members:
            for m in masters:
                ra = math.sqrt(max(m.area_uv, 1e-18))
                rb = math.sqrt(max(isl.area_uv, 1e-18))
                if abs(ra - rb) > 0.01 * ra:
                    continue
                if packer.kabsch(m.pts, isl.pts, allow_reflect) < tol:
                    stacks[m].append(isl)
                    break
            else:
                masters.append(isl)
                stacks[isl] = []
    return stacks


def pack(context, *, selected_only=False, margin_px=8, texture=2048, rotate=True,
         rot_step=90, pre_orient=True, normalize=False, stack=False, lock_pinned=True,
         avoid_fixed=True, fixed_scale=False, tile=1001, resolution=384, orders=1,
         threads=0, meshes=None):
    """Упаковать острова. Возвращает dict со статистикой."""
    t0 = time.perf_counter()
    own = meshes is None
    if own:
        meshes = gather(context)
    allisl = all_islands(meshes)
    targets = [i for i in allisl if i.selected or not selected_only]
    movable = [i for i in targets if not (lock_pinned and i.pinned)]
    if not movable:
        if own:
            finish(meshes, False)
        return {"packed": 0, "error": "Нет островов для упаковки"}

    off = udim_offset(tile)
    fixed = []
    if selected_only and avoid_fixed:
        fixed += [i for i in allisl if i not in targets]
    if lock_pinned:
        fixed += [i for i in targets if i.pinned]
    obstacles = []
    for i in fixed:
        mn, mx = i.bounds()
        if (mx > off).all() and (mn < off + 1).all():
            obstacles.append((i.pts - off, i.tris))

    copies = {}
    if stack:
        st = find_stacks(movable)
        movable = list(st.keys())
        copies = st

    if normalize:
        dens = [math.sqrt(i.area3d / i.area_uv) if i.area_uv > 1e-12 and i.area3d > 0 else 0
                for i in movable]
        good = [d for d in dens if d > 0]
        ref = float(np.median(good)) if good else 1.0
        scales = [(d / ref if d > 0 else 1.0) for d in dens]
    else:
        scales = [1.0] * len(movable)

    items = []
    for isl, s in zip(movable, scales):
        a0 = packer.min_rect_angle(isl.pts) if pre_orient and rotate else 0.0
        items.append({"pts": isl.pts, "tris": isl.tris, "scale": s, "angle": a0})

    if rotate and rot_step > 0:
        step = math.radians(rot_step)
        angles = [k * step for k in range(max(1, int(round(360 / rot_step))))]
    else:
        angles = [0.0]
    threads = threads or min(8, os.cpu_count() or 4)

    res = packer.pack(items, resolution=resolution, margin=margin_px / max(texture, 1),
                      angles=angles, threads=threads, fixed_scale=fixed_scale,
                      obstacles=obstacles, orders=orders)
    packed = 0
    failed = 0
    for isl, tr in zip(movable, res["transforms"]):
        if tr is None:
            failed += 1
            continue
        M, t = tr
        isl.transform(M, t + off)
        packed += 1
        for c in copies.get(isl, ()):
            c.pts = isl.pts.copy()
            packed += 1

    stats = {"packed": packed, "failed": failed, "islands": len(movable),
             "stacked": sum(len(v) for v in copies.values()),
             "time": time.perf_counter() - t0}
    if res.get("N") is None:
        stats["error"] = "Не удалось упаковать"
    if own:
        finish(meshes, packed > 0)
        stats["coverage"] = None
    return stats


# ----------------------------------------------------------------- выравнивание

def orient(context, selected_only=True, vertical=True):
    meshes = gather(context, need_3d=False)
    isl = target_islands(meshes, selected_only)
    for i in isl:
        i.rotate(packer.min_rect_angle(i.pts, vertical))
    finish(meshes, bool(isl))
    return len(isl)


def straighten(context):
    """Повернуть остров так, чтобы выделенное UV-ребро стало строго
    горизонтальным/вертикальным (ближайшее)."""
    meshes = gather(context, need_3d=False)
    sync = _uv_sync(context)
    n = 0
    for md in meshes:
        uv = md.uv
        for isl in md.islands:
            best = None
            for l in isl.loops:
                sel = l.edge.select if sync else l.uv_select_edge
                if not sel:
                    continue
                d = l.link_loop_next[uv].uv - l[uv].uv
                if best is None or d.length > best.length:
                    best = d.copy()
            if best is None or best.length < 1e-9:
                continue
            a = math.atan2(best.y, best.x)
            target = round(a / (math.pi / 2)) * (math.pi / 2)
            isl.rotate(target - a)
            n += 1
    finish(meshes, n > 0)
    return n


def align(context, mode):
    meshes = gather(context, need_3d=False)
    isl = target_islands(meshes, True)
    if len(isl) < 1:
        finish(meshes, False)
        return 0
    b = [i.bounds() for i in isl]
    mn = np.min([x[0] for x in b], 0)
    mx = np.max([x[1] for x in b], 0)
    c = (mn + mx) / 2
    for i, (lo, hi) in zip(isl, b):
        d = np.zeros(2)
        if mode == "LEFT":
            d[0] = mn[0] - lo[0]
        elif mode == "RIGHT":
            d[0] = mx[0] - hi[0]
        elif mode == "BOTTOM":
            d[1] = mn[1] - lo[1]
        elif mode == "TOP":
            d[1] = mx[1] - hi[1]
        elif mode == "CENTER_H":
            d[0] = c[0] - (lo[0] + hi[0]) / 2
        elif mode == "CENTER_V":
            d[1] = c[1] - (lo[1] + hi[1]) / 2
        i.pts = i.pts + d
    finish(meshes, True)
    return len(isl)


def distribute(context, axis="X", gap=None):
    meshes = gather(context, need_3d=False)
    isl = target_islands(meshes, True)
    if len(isl) < 2:
        finish(meshes, False)
        return 0
    k = 0 if axis == "X" else 1
    isl.sort(key=lambda i: i.center()[k])
    b = [i.bounds() for i in isl]
    sizes = [hi[k] - lo[k] for lo, hi in b]
    start, end = b[0][0][k], max(hi[k] for lo, hi in b)
    if gap is None:
        gap = (end - start - sum(sizes)) / (len(isl) - 1)
    pos = start
    for i, (lo, hi), s in zip(isl, b, sizes):
        d = np.zeros(2)
        d[k] = pos - lo[k]
        i.pts = i.pts + d
        pos += s + gap
    finish(meshes, True)
    return len(isl)


def stack_similar(context, selected_only=True, tol=0.002):
    meshes = gather(context, need_3d=False)
    isl = target_islands(meshes, selected_only)
    st = find_stacks(isl, tol)
    n = 0
    for m, cs in st.items():
        for c in cs:
            c.pts = m.pts.copy()
            n += 1
    finish(meshes, n > 0)
    return n


def select_similar(context, tol=0.002):
    meshes = gather(context, need_3d=False)
    allisl = all_islands(meshes)
    sel = [i for i in allisl if i.selected]
    st = find_stacks(allisl, tol)
    picked = []
    for m, cs in st.items():
        grp = [m] + cs
        if any(i.selected for i in grp):
            picked += grp
    if sel:
        select_islands(context, meshes, picked)
    for md in meshes:
        md.free()
    return len(picked)


# ----------------------------------------------------------------- тексели

def texel_get(context, texture):
    meshes = gather(context)
    isl = target_islands(meshes, True) or all_islands(meshes)
    a3 = sum(i.area3d for i in isl)
    auv = sum(i.area_uv for i in isl)
    finish(meshes, False)
    if a3 <= 0:
        return 0.0
    return math.sqrt(auv / a3) * texture


def texel_set(context, texture, density, selected_only=True):
    """Масштабировать каждый остров (вокруг своего центра) до нужной плотности."""
    meshes = gather(context)
    isl = target_islands(meshes, selected_only)
    n = 0
    for i in isl:
        cur = i.texel(texture)
        if cur > 0:
            i.scale(density / cur)
            n += 1
    finish(meshes, n > 0)
    return n


def texel_equalize(context, texture, selected_only=True):
    meshes = gather(context)
    isl = target_islands(meshes, selected_only)
    a3 = sum(i.area3d for i in isl)
    auv = sum(i.area_uv for i in isl)
    finish(meshes, False)
    if a3 <= 0:
        return 0
    return texel_set(context, texture, math.sqrt(auv / a3) * texture, selected_only)


# ----------------------------------------------------------------- проверка

def check(context, resolution=1024, tile=1001, select_overlaps=True):
    """Заполнение плитки, перекрытия, острова за пределами плитки."""
    meshes = gather(context, need_3d=False)
    isl = all_islands(meshes)
    off = udim_offset(tile)
    N = int(resolution)
    cells = []
    outside = 0
    for i in isl:
        mn, mx = i.bounds()
        if (mn < off - 1e-6).any() or (mx > off + 1 + 1e-6).any():
            outside += 1
        cells.append(packer.coverage_cells(i.pts - off, i.tris, N))
    if cells:
        allc = np.concatenate(cells)
        uniq, cnt = np.unique(allc, return_counts=True)
        coverage = len(uniq) / (N * N)
        multi = uniq[cnt > 1]
    else:
        coverage, multi = 0.0, np.zeros(0, np.int64)
    over = [i for i, c in zip(isl, cells) if len(c) and np.isin(c, multi, assume_unique=True).any()]
    if select_overlaps and over:
        select_islands(context, meshes, over)
    for md in meshes:
        md.free()
    return {"islands": len(isl), "coverage": coverage, "overlaps": len(over),
            "outside": outside}
