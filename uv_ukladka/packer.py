"""Ядро упаковки UV-островов (только numpy, без bpy — тестируется отдельно).

Идея: каждый остров растеризуется в консервативную битовую маску (с отступом),
маски раскладываются по сетке N×N «снизу-слева»; свободные позиции для сложной
формы ищутся через FFT-корреляцию маски с занятостью сетки, для мелких и почти
прямоугольных — через интегральное изображение. Минимальный N (то есть
максимальный масштаб островов) ищется бисекцией, кандидаты N проверяются
параллельно в нескольких потоках (numpy отпускает GIL в FFT и операциях над
массивами).
"""
import math
import os
import pickle
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MAX_N = 8192


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------- геометрия

def tri_area(P, tris):
    if len(tris) == 0:
        return 0.0
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    ab, ac = b - a, c - a
    return float(np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]).sum() * 0.5)


def tri_area3d(Q, tris):
    if len(tris) == 0:
        return 0.0
    a, b, c = Q[tris[:, 0]], Q[tris[:, 1]], Q[tris[:, 2]]
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() * 0.5)


def convex_hull(P):
    pts = np.unique(np.round(P, 9), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))].tolist()

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h

    lower, upper = half(pts), half(reversed(pts))
    return np.array(lower[:-1] + upper[:-1])


def min_rect_angle(P, vertical=True):
    """Угол поворота, после которого рамка острова минимальна (и длинная сторона
    вертикальна при vertical=True, иначе горизонтальна)."""
    H = convex_hull(P)
    if len(H) < 3:
        return 0.0
    E = np.roll(H, -1, axis=0) - H
    ang = np.unique(np.round(np.mod(np.arctan2(E[:, 1], E[:, 0]), math.pi / 2), 9))
    c, s = np.cos(-ang), np.sin(-ang)
    X = H[:, 0][None] * c[:, None] - H[:, 1][None] * s[:, None]
    Y = H[:, 0][None] * s[:, None] + H[:, 1][None] * c[:, None]
    w = X.max(1) - X.min(1)
    h = Y.max(1) - Y.min(1)
    i = int(np.argmin(w * h + 1e-12 * np.arange(len(ang))))
    a = -float(ang[i])
    if (h[i] < w[i]) == vertical:
        a += math.pi / 2
    return a


def bbox(P):
    return P.min(0), P.max(0)


# ---------------------------------------------------------------- растеризация

def _center_cells(Q, tris):
    """(x, y) клеток, центры которых внутри треугольников. Q — в единицах клеток."""
    if len(tris) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    Qc = Q - 0.5  # центр клетки (i+0.5) → i
    A, B, C = Qc[tris[:, 0]], Qc[tris[:, 1]], Qc[tris[:, 2]]
    lo = np.ceil(np.minimum(np.minimum(A, B), C)).astype(np.int64)
    hi = np.floor(np.maximum(np.maximum(A, B), C)).astype(np.int64)
    size = np.maximum(hi - lo + 1, 0).max(1)
    ok = size > 0
    xs, ys = [], []
    bucket = np.where(ok, np.ceil(np.log2(np.maximum(size, 1))).astype(np.int64), -1)
    for b in np.unique(bucket[ok]):
        S = 1 << int(b)
        idx = np.nonzero(bucket == b)[0]
        gy, gx = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
        gx, gy = gx.ravel(), gy.ravel()
        step = max(1, 2_000_000 // (S * S))
        for s0 in range(0, len(idx), step):
            ii = idx[s0:s0 + step]
            x = lo[ii, 0][:, None] + gx[None]
            y = lo[ii, 1][:, None] + gy[None]
            a, b_, c = A[ii], B[ii], C[ii]
            v0 = c - a
            v1 = b_ - a
            px = x - a[:, 0:1]
            py = y - a[:, 1:2]
            d00 = (v0 * v0).sum(1)[:, None]
            d01 = (v0 * v1).sum(1)[:, None]
            d11 = (v1 * v1).sum(1)[:, None]
            d20 = px * v0[:, 0:1] + py * v0[:, 1:2]
            d21 = px * v1[:, 0:1] + py * v1[:, 1:2]
            den = d00 * d11 - d01 * d01
            den = np.where(np.abs(den) < 1e-18, np.inf, den)
            v = (d11 * d20 - d01 * d21) / den
            w = (d00 * d21 - d01 * d20) / den
            inside = (v >= -1e-9) & (w >= -1e-9) & (v + w <= 1 + 1e-9)
            inside &= (x <= hi[ii, 0][:, None]) & (y <= hi[ii, 1][:, None])
            xs.append(x[inside])
            ys.append(y[inside])
    if not xs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(xs).astype(np.int64), np.concatenate(ys).astype(np.int64)


def _edge_cells(Q, tris, step=0.25):
    """Клетки, через которые проходят рёбра (выборка с шагом 1/4 клетки)."""
    if len(tris) == 0:
        f = np.floor(Q).astype(np.int64)
        return f[:, 0], f[:, 1]
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    A, B = Q[e[:, 0]], Q[e[:, 1]]
    n = np.clip(np.ceil(np.linalg.norm(B - A, axis=1) / step), 1, 1 << 16).astype(np.int64)
    parts = [Q]
    for nv in np.unique(n):
        ii = np.nonzero(n == nv)[0]
        t = (np.arange(1, nv) / nv)[None, :, None]
        if nv > 1:
            parts.append((A[ii][:, None, :] + (B[ii] - A[ii])[:, None, :] * t).reshape(-1, 2))
    f = np.floor(np.concatenate(parts)).astype(np.int64)
    return f[:, 0], f[:, 1]


def cells(Q, tris):
    """Все клетки, которых касается остров: центры внутри + рёбра + вершины.
    Пропустить можно лишь уголок клетки, срезанный ребром короче 1/4 клетки."""
    x1, y1 = _center_cells(Q, tris)
    x2, y2 = _edge_cells(Q, tris)
    return np.concatenate([x1, x2]), np.concatenate([y1, y2])


def dilate(m, r, square=False):
    """Расширение маски на r клеток (восьмиугольник ≈ круг; square — квадрат)."""
    for k in range(r):
        p = np.pad(m, 1)
        out = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        if square or k % 2 == 0:
            out |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
        m = out
    return m


def rasterize_local(loc, tris, g):
    """loc — пиксельные координаты острова, min(loc) == g. Возвращает (форма,
    проверочная маска = форма, расширенная на отступ g)."""
    x, y = cells(loc, tris)
    W = int(x.max()) + g + 1
    H = int(y.max()) + g + 1
    core = np.zeros((H, W), bool)
    core[y, x] = True
    return core, dilate(core, g)


def rasterize_fixed(P, tris, N):
    """Неподвижный остров в сетке N×N (координаты UV × N)."""
    x, y = cells(P * N, tris)
    ok = (x >= 0) & (x < N) & (y >= 0) & (y < N)
    m = np.zeros((N, N), bool)
    m[y[ok], x[ok]] = True
    return m


def coverage_cells(P, tris, N):
    """Клетки сетки N×N, центры которых лежат внутри треугольников (точная заливка
    по центрам — для проверки перекрытий и процента заполнения)."""
    x, y = _center_cells(P * N, tris)
    ok = (x >= 0) & (x < N) & (y >= 0) & (y < N)
    return np.unique(y[ok] * N + x[ok])


# ---------------------------------------------------------------- раскладка

def inner_rect(m):
    """Большой прямоугольник целиком внутри маски (жадный рост от «центра»):
    (y, x, h, w). Нужен как нижняя граница при поиске места для формы."""
    ys, xs = np.nonzero(m)
    c = np.array([ys.mean(), xs.mean()])
    i = int(((ys - c[0]) ** 2 + (xs - c[1]) ** 2).argmin())
    y0 = y1 = int(ys[i])
    x0 = x1 = int(xs[i])
    H, Wd = m.shape
    step = max(1, min(H, Wd) // 16)
    while step >= 1:
        grown = True
        while grown:
            grown = False
            if y0 - step >= 0 and m[y0 - step:y0, x0:x1 + 1].all():
                y0 -= step; grown = True
            if y1 + step < H and m[y1 + 1:y1 + step + 1, x0:x1 + 1].all():
                y1 += step; grown = True
            if x0 - step >= 0 and m[y0:y1 + 1, x0 - step:x0].all():
                x0 -= step; grown = True
            if x1 + step < Wd and m[y0:y1 + 1, x1 + 1:x1 + step + 1].all():
                x1 += step; grown = True
        step //= 2
    return y0, x0, y1 - y0 + 1, x1 - x0 + 1


class Variant:
    __slots__ = ("angle", "shift", "mask", "core", "area", "carea", "box", "rect")

    def __init__(self, angle, shift, core, mask, box=None, rect=None):
        self.angle = angle
        self.shift = shift          # px-координаты: loc = R·p·k + shift
        self.mask = mask            # с отступом — проверяется на пересечение
        self.core = core            # сама форма — кладётся в сетку
        self.area = int(mask.sum())
        self.carea = int(core.sum())
        # почти прямоугольную или крошечную маску проверяем по рамке (быстро)
        if box is None:
            box = mask.size <= 100 or self.area >= 0.9 * mask.size
            rect = None if box else inner_rect(mask)
        self.box, self.rect = box, rect

    def plain(self):
        return (self.angle, self.shift, self.core, self.mask, self.box, self.rect)


def from_plain(data):
    return [[Variant(*t) for t in vs] for vs in data]


def build_chunk(items, k_base, g, angles):
    return [build_variants(it["pts"], it["tris"], k_base * it.get("scale", 1.0), g,
                           [it.get("angle", 0.0) + a for a in angles]) for it in items]


def build_all(items, k_base, g, angles, workers=1, procs=False):
    """Маски всех островов; при procs — параллельно в процессах-помощниках."""
    exe = _python_exe() if procs and workers > 1 else None
    if not exe:
        return build_chunk(items, k_base, g, angles)
    n = len(items)
    parts = [items[i * n // workers:(i + 1) * n // workers] for i in range(workers)]
    paths, jobs = [], []
    try:
        for part in parts:
            fd, path = tempfile.mkstemp(prefix="ukl_b_", suffix=".pkl")
            paths.append(path)
            with os.fdopen(fd, "wb") as f:
                pickle.dump(([{"pts": it["pts"], "tris": it["tris"], "scale": it.get("scale", 1.0),
                               "angle": it.get("angle", 0.0)} for it in part], k_base, g, list(angles)),
                            f, protocol=pickle.HIGHEST_PROTOCOL)
            jobs.append(subprocess.Popen([exe, "-I", _WORKER, "build", path],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
        out = []
        for part, p in zip(parts, jobs):
            data, _ = p.communicate()
            try:
                if p.returncode != 0:
                    raise RuntimeError
                out += from_plain(pickle.loads(data))
            except Exception:
                out += build_chunk(part, k_base, g, angles)
        return out
    finally:
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass


def build_variants(pts, tris, k, g, angles):
    out = []
    seen = set()
    for a in angles:
        q = (pts @ rot(a).T) * k
        qmin = q.min(0)
        shift = -qmin + g
        core, m = rasterize_local(q + shift, tris, g)
        key = (m.shape, core.tobytes())
        if key in seen:  # симметричные повороты не повторяем
            continue
        seen.add(key)
        out.append(Variant(a, shift, core, m))
    return out


def fast_len(n):
    """Ближайший сверху размер вида 2^a·3^b·5^c — на «простых» размерах FFT в разы медленнее."""
    best = 1 << max(0, int(n - 1).bit_length())
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p = p35
            while p < n:
                p *= 2
            best = min(best, p)
            p35 *= 3
        p5 *= 5
    return best


INF = np.iinfo(np.int64).max
DEPTH = 3           # глубже 3 высот острова под найденным местом «дырки» не ищем (0 — без ограничения)
CELLS_PER_ISLAND = 90


class _Grid:
    """Сетка занятости с быстрым поиском «снизу-слева».

    LB[h, w] — нижняя граница ряда, ниже которого рамка h×w заведомо не влезает
    (занятость только растёт, поэтому граница остаётся верной до конца раскладки)."""

    def __init__(self, N, Hc, maxh, maxw, obst):
        self.N = N                  # ширина
        self.Hc = Hc                # высота
        self.W = 2 * N + 2
        self.occ = np.zeros((Hc + maxh, N + maxw), np.float32)
        self.occ[Hc:, :] = 1
        self.occ[:, N:] = 1
        if obst is not None:
            self.occ[:Hc, :N] = obst
        self.rowfull = self.occ[:Hc, :N].all(1)
        self.LB = np.zeros((maxh + 1, maxw + 1), np.int64)
        self._ylo = 0
        self._cache = None

    def y_lo(self):
        rf = self.rowfull
        y = self._ylo
        while y < self.Hc and rf[y]:
            y += 1
        self._ylo = y
        return y

    def _band(self, y, rows):
        """Интегральное изображение полосы рядов [y, y+rows) (кэш до следующей укладки)."""
        c = self._cache
        if c is not None and c[0] <= y and y + rows <= c[1]:
            return c[0], c[2]
        rows = max(rows, 64)
        rows = min(rows, self.occ.shape[0] - y)
        N = self.N
        S = np.zeros((rows + 1, N + 1))
        np.cumsum(self.occ[y:y + rows, :N], 0, out=S[1:, 1:])
        np.cumsum(S[1:, 1:], 1, out=S[1:, 1:])
        self._cache = (y, y + rows, S)
        return y, S

    def find_box(self, h, w):
        """Нижняя-левая позиция, где рамка h×w целиком свободна: (y, x) или None."""
        N, Hc = self.N, self.Hc
        y = max(self.y_lo(), int(self.LB[h, w]))
        band = max(16, 2 * h)
        nx = N - w + 1
        while y <= Hc - h:
            y1 = min(Hc - h, y + band - 1)
            ny = y1 - y + 1
            y0, S = self._band(y, ny + h - 1)
            o = y - y0
            box = (S[o + h:o + h + ny, w:w + nx] - S[o:o + ny, w:w + nx]
                   - S[o + h:o + h + ny, 0:nx] + S[o:o + ny, 0:nx])
            free = box < 0.5
            ra = free.any(1)
            if ra.any():
                r = int(ra.argmax())
                yy = y + r
                np.maximum(self.LB[h:, w:], yy, out=self.LB[h:, w:])
                return yy, int(free[r].argmax())
            y = y1 + 1
            band *= 2
        np.maximum(self.LB[h:, w:], Hc, out=self.LB[h:, w:])
        return None

    def find_exact(self, v, limit):
        """Позиция точной формы v.mask с ценой < limit: (цена, x, y) или None."""
        N, W = self.N, self.W
        h, w = v.mask.shape
        ry, rx, rh, rw = v.rect
        pr = self.find_box(rh, rw)  # вписанный прямоугольник: нижняя граница
        if pr is None:
            return None
        y_a = max(self.y_lo(), pr[0] - ry, 0)
        Hc = self.Hc
        y_b = Hc - h if limit == INF else min(Hc - h, limit // W - h)
        if DEPTH and limit != INF:  # глубже DEPTH высот острова под рамкой не ищем
            y_a = max(y_a, y_b - DEPTH * h)
        if y_b < y_a:
            return None
        B = y_b - y_a + h
        shape = (fast_len(B), fast_len(N))
        F = np.fft.rfft2(self.occ[y_a:y_a + B, :N], s=shape)
        M = np.fft.rfft2(v.mask.astype(np.float32), s=shape)
        nx = N - w + 1
        corr = np.fft.irfft2(F * np.conj(M), s=shape)[:y_b - y_a + 1, :nx]
        free = corr < 0.5
        ra = free.any(1)
        if not ra.any():
            return None
        r = int(ra.argmax())
        x = int(free[r].argmax())
        y = y_a + r
        c = (y + h) * W + x + w
        return (c, x, y) if c < limit else None

    def place(self, m, x, y):
        self._cache = None
        sub = self.occ[y:y + m.shape[0], x:x + m.shape[1]]
        np.maximum(sub, m, out=sub)
        hh = min(m.shape[0], self.Hc - y)
        self.rowfull[y:y + hh] = self.occ[y:y + hh, :self.N].all(1)


def pack_once(variants, N, obst=None, partial=False, Hc=None):
    """Раскладка в сетку N×N (или полосу ширины N и высоты Hc). variants — список (на остров) списков Variant в
    порядке укладки. Возвращает список (vi, x, y) или None при неудаче
    (partial=True: None на месте не влезшего острова).

    Сначала ищется лучшая позиция, где рамка острова целиком свободна (дёшево);
    точная форма проверяется FFT-корреляцией только в полосе ниже неё."""
    maxh = max((v.mask.shape[0] for vs in variants for v in vs), default=1)
    maxw = max((v.mask.shape[1] for vs in variants for v in vs), default=1)
    Hc = N if Hc is None else Hc
    g = _Grid(N, Hc, maxh, maxw, obst)
    W = g.W
    res = []
    for vs in variants:
        best = None
        fits = [v.mask.shape[0] <= Hc and v.mask.shape[1] <= N for v in vs]
        asked = set()
        for vi, v in enumerate(vs):
            if not fits[vi] or v.mask.shape in asked:
                continue
            asked.add(v.mask.shape)  # повороты с той же рамкой дают тот же ответ
            h, w = v.mask.shape
            p = g.find_box(h, w)
            if p is not None:
                c = (p[0] + h) * W + p[1] + w
                if best is None or c < best[0]:
                    best = (c, vi, p[1], p[0])
        for vi, v in enumerate(vs):
            if not fits[vi] or v.box:
                continue
            e = g.find_exact(v, best[0] if best else INF)
            if e is not None:
                best = (e[0], vi, e[1], e[2])
        if best is None:
            if partial:
                res.append(None)
                continue
            return None
        _, vi, x, y = best
        g.place(vs[vi].core, x, y)
        res.append((vi, x, y))
    return res


def search_min_n(variants, lower, start, threads=4, obst_fn=None, tol=0.01,
                 max_rounds=14, progress=None):
    """Минимальное N, при котором все острова влезли. Кандидаты проверяются
    параллельно."""
    threads = max(1, int(threads))

    def trial(N):
        return pack_once(variants, N, obst_fn(N) if obst_fn else None)

    lo = max(1, lower - 1)
    hi, best = None, None
    grow = 1.06 if threads == 1 else 1.12
    with ThreadPoolExecutor(threads) as pool:
        for rnd in range(max_rounds):
            if hi is None:
                base = max(lo + 1, start) if rnd == 0 else lo + 1
                span = base * (grow - 1)  # кандидаты равномерно от base до base·grow
                if threads > 1:
                    cands = sorted({min(MAX_N, int(math.ceil(base + span * k / (threads - 1))))
                                    for k in range(threads)})
                else:
                    cands = [base if rnd == 0 else min(MAX_N, int(math.ceil(lo * grow)))]
                grow = min(1 + (grow - 1) * 2, 2.0)
            else:
                if hi - lo <= max(1, tol * hi):
                    break
                cands = sorted({int(round(x)) for x in np.linspace(lo, hi, threads + 2)[1:-1]} - {lo, hi})
                cands = [c for c in cands if lo < c < hi] or [(lo + hi) // 2]
            results = list(pool.map(trial, cands))
            for N, r in zip(cands, results):
                if r is not None and (hi is None or N < hi):
                    hi, best = N, r
            fails = [N for N, r in zip(cands, results) if r is None and (hi is None or N < hi)]
            if fails:
                lo = max(lo, max(fails))
            if progress:
                progress(rnd, lo, hi)
            if hi is None and max(cands) >= MAX_N:
                break
    return hi, best


def used_height(variants, res):
    return max((y + variants[i][vi].mask.shape[0] for i, (vi, x, y) in enumerate(res)), default=0)


def _strip_trial(variants, W):
    """Укладка в полосу ширины W: (сторона квадрата, раскладка)."""
    res = pack_once(variants, W, Hc=int(W * 2.5) + 8)
    if res is None:
        return math.inf, None
    return max(W, used_height(variants, res)), res


_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_worker.py")


def _python_exe():
    exe = sys.executable or ""
    return exe if os.path.basename(exe).lower().startswith("python") and os.path.exists(exe) else None


class Runner:
    """Параллельные пробы (ширина полосы, порядок укладки): отдельные процессы
    Python (у каждого свой GIL — настоящая загрузка всех ядер) или потоки для
    мелких задач."""

    def __init__(self, variants, orders, workers, procs):
        self.variants = variants
        self.orders = orders
        self.workers = max(1, int(workers))
        self.exe = _python_exe() if procs and self.workers > 1 else None
        self.path = None

    def __enter__(self):
        if self.exe:
            fd, self.path = tempfile.mkstemp(prefix="ukl_", suffix=".pkl")
            with os.fdopen(fd, "wb") as f:
                pickle.dump(([[v.plain() for v in vs] for vs in self.variants], self.orders), f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def __exit__(self, *a):
        if self.path:
            try:
                os.remove(self.path)
            except OSError:
                pass

    def _local(self, job):
        W, oi = job
        return _strip_trial([self.variants[i] for i in self.orders[oi]], W)

    def _score(self, job, res):
        W, oi = job
        if res is None:
            return math.inf, None
        return max(W, used_height([self.variants[i] for i in self.orders[oi]], res)), res

    def run(self, jobs):
        """jobs: список (W, номер порядка) → список (сторона, раскладка)."""
        jobs = list(jobs)
        if not self.exe:
            with ThreadPoolExecutor(self.workers) as pool:
                return list(pool.map(self._local, jobs))
        out = [None] * len(jobs)
        for s0 in range(0, len(jobs), self.workers):
            procs = []
            for i in range(s0, min(len(jobs), s0 + self.workers)):
                W, oi = jobs[i]
                p = subprocess.Popen([self.exe, "-I", _WORKER, self.path, str(W), str(oi)],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                procs.append((i, p))
            for i, p in procs:
                data, _ = p.communicate()
                try:
                    if p.returncode != 0:
                        raise RuntimeError
                    out[i] = self._score(jobs[i], pickle.loads(data))
                except Exception:  # процесс не смог — считаем здесь
                    out[i] = self._local(jobs[i])
        return out


def search_strip(variants, orders, est, workers=4, procs=True, progress=None):
    """Подбор ширины полосы W и порядка укладки, при которых квадрат
    max(W, высота) минимален. Каждая проба полезна (в отличие от бисекции,
    где неудачи пропадают). Возвращает (сторона, номер порядка, раскладка)."""
    T = max(2, int(workers))
    no = len(orders)
    best = (math.inf, None, None, None)  # сторона, W, порядок, раскладка
    tried = set()
    with Runner(variants, orders, workers, procs) as run:
        for rnd in range(2):
            if rnd == 0:  # каждый порядок — с T ширинами
                jobs = [(max(1, int(round(est * f))), oi)
                        for oi in range(no) for f in np.linspace(0.95, 1.10, T + oi % 2)]
            elif best[1] is None:
                jobs = [(max(1, int(round(est * f))), 0) for f in np.linspace(1.12, 1.5, T)]
            else:  # уточнение вокруг лучшей ширины для лучшего порядка
                jobs = [(max(1, int(round(best[1] * f))), best[2]) for f in np.linspace(0.97, 1.03, T)]
            jobs = [j for j in dict.fromkeys(jobs) if j not in tried]
            tried |= set(jobs)
            for (W, oi), (side, res) in zip(jobs, run.run(jobs)):
                if res is not None and (side, W) < (best[0], best[1] or 0):
                    best = (side, W, oi, res)
            if progress:
                progress(rnd, best[0])
    return best[0], best[2], best[3]


def make_orders(variants, count, seed=1):
    """Разные порядки укладки: по площади, длинной стороне, периметру, высоте,
    затем по площади со случайным разбросом."""
    n = len(variants)
    area = np.array([max(v.area for v in vs) for vs in variants], float)
    side = np.array([max(max(v.mask.shape) for v in vs) for vs in variants], float)
    perim = np.array([max(sum(v.mask.shape) for v in vs) for vs in variants], float)
    high = np.array([max(min(v.mask.shape) for v in vs) for vs in variants], float)
    keys = [area, side * 1e6 + area, perim * 1e6 + area, high * 1e6 + area]
    rng = np.random.default_rng(seed)
    while len(keys) < count:
        keys.append(area * rng.uniform(0.75, 1.25, n))
    return [np.argsort(-k, kind="stable").tolist() for k in keys[:max(1, count)]]


def pack(items, resolution=384, margin=0.004, angles=(0.0,), threads=4,
         fixed_scale=False, obstacles=(), progress=None, procs=True, orders=2):
    """items: список dict(pts=(n,2) UV, tris=(t,3), scale=отн. масштаб, angle=базовый угол).
    obstacles: список (pts, tris) неподвижных островов в координатах плитки 0..1.
    Возвращает dict: transforms — на каждый остров (M 2×2, t) так, что
    uv_new = M @ uv + t, либо None (не влез при fixed_scale); N, efficiency."""
    n = len(items)
    R = int(resolution)
    if not fixed_scale:  # мелким островам — достаточно клеток, иначе отступы съедят место
        R = max(R, min(2048, int(math.sqrt(n * CELLS_PER_ISLAND / 0.7))))
    if n == 0:
        return {"transforms": [], "N": R}
    areas = [tri_area(it["pts"], it["tris"]) * it.get("scale", 1.0) ** 2 for it in items]
    A = max(sum(areas), 1e-12)

    obst_cache = {}

    def obst_mask(N, r=0):
        if N not in obst_cache:
            m = np.zeros((N, N), bool)
            for P, T in obstacles:
                m |= rasterize_fixed(P, T, N)
            obst_cache[N] = m
        return obst_cache[N]

    free_frac = 1.0
    if obstacles:
        free_frac = max(0.05, 1.0 - obst_mask(R, 0).mean())

    if fixed_scale:
        k_base = float(R)
    else:
        k_base = R * math.sqrt(0.72 * free_frac / A)

    use_procs = procs and n >= 300 and threads > 1

    def build(r):
        return build_all(items, k_base, r, angles, threads, use_procs)

    def lower_n(vars_):
        return int(math.ceil(math.sqrt(sum(min(v.carea for v in vs) for vs in vars_) / free_frac)))

    def est_n(vars_):  # ожидаемая сторона: отступ делится между соседями
        return math.sqrt(sum(min(v.carea + v.area for v in vs) / 2 for vs in vars_) / free_frac / 0.92)

    r = int(math.ceil(margin * R))  # полный зазор в клетках
    variants = build(r)
    if not fixed_scale and margin > 0:
        need = int(math.ceil(margin * est_n(variants)))
        if need > r:  # отступ задан в долях текстуры — подгоняем под ожидаемый N
            r = need
            variants = build(r)
    order = None

    def run(vars_, r_):
        nonlocal order
        ords = make_orders(vars_, orders)
        order = ords[0]
        ordered = [vars_[i] for i in order]
        if fixed_scale:
            res = pack_once(ordered, R, obst_mask(R, r_) if obstacles else None, partial=True)
            return R, res
        if not obstacles:
            side, oi, res = search_strip(vars_, ords, est_n(vars_), threads, use_procs, progress)
            if res is None:
                return None, None
            order = ords[oi]
            return int(side), res
        lower = lower_n(vars_)
        fn = lambda N: obst_mask(N, r_)
        return search_min_n(ordered, lower, max(lower, int(est_n(vars_))), threads, fn,
                            progress=progress)

    N, res = run(variants, r)
    if not fixed_scale and N is not None and margin > 0:
        need = int(math.ceil(margin * N))
        if need > r:  # отступ задан в долях текстуры — уточняем под итоговый N
            r = need
            variants = build(r)
            N, res = run(variants, r)
    if N is None or res is None:
        return {"transforms": [None] * n, "N": None}

    transforms = [None] * n
    for slot, i in enumerate(order):
        pl = res[slot]
        if pl is None:
            continue
        vi, x, y = pl
        v = variants[i][vi]
        k = k_base * items[i].get("scale", 1.0)
        M = rot(v.angle) * (k / N)
        t = (v.shift + np.array([x, y], float)) / N
        transforms[i] = (M, t)
    return {"transforms": transforms, "N": N, "k": k_base}


def kabsch(A, B, allow_reflect=True):
    """Насколько форма B совпадает с A (поворот+сдвиг[+зеркало]). Возвращает
    среднеквадратичную ошибку в долях размера A."""
    a = A - A.mean(0)
    b = B - B.mean(0)
    H = b.T @ a
    U, S, Vt = np.linalg.svd(H)
    best = math.inf
    for refl in ((False, True) if allow_reflect else (False,)):
        V = Vt.T.copy()
        Rm = V @ U.T
        if (np.linalg.det(Rm) < 0) != refl:
            V[:, 1] *= -1
            Rm = V @ U.T
        err = np.sqrt(((b @ Rm.T - a) ** 2).sum(1).mean())
        best = min(best, err)
    size = max(float(np.ptp(a, axis=0).max()), 1e-12)
    return best / size
