"""Тест аддона в фоновом Blender (аддон должен быть установлен в профиль).
Запуск: tests/run_tests.sh"""
import math
import sys
import time
import traceback

import bmesh
import bpy
import numpy as np

import addon_utils

FAILS = []


def ok(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def mod():
    for m in addon_utils.modules():
        if m.__name__.endswith("uv_ukladka"):
            return sys.modules[m.__name__]
    raise RuntimeError("аддон не найден")


def clear():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)


def make_scene(copies=0):
    clear()
    ops = bpy.ops.mesh
    ops.primitive_monkey_add(location=(0, 0, 0))
    ops.primitive_cylinder_add(location=(3, 0, 0), vertices=24)
    ops.primitive_torus_add(location=(6, 0, 0))
    ops.primitive_uv_sphere_add(location=(9, 0, 0))
    ops.primitive_cube_add(location=(12, 0, 0), size=1.5)
    ops.primitive_cone_add(location=(15, 0, 0))
    for o in bpy.context.scene.objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = bpy.context.scene.objects[0]
    bpy.ops.object.join()
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    for k in range(copies):  # копии с той же развёрткой (как массив)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.duplicate()
        bpy.context.object.location.y += 3 * (k + 1)
    if copies:
        for o in bpy.context.scene.objects:
            o.select_set(True)
        bpy.ops.object.join()
        obj = bpy.context.object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    return obj


def strict_check(m, res=2048):
    """Перекрытия по центрам клеток в сетке res и выход за [0,1]."""
    core = m.core
    meshes = core.gather(bpy.context, need_3d=False)
    isl = core.all_islands(meshes)
    cells = [m.packer.coverage_cells(i.pts, i.tris, res) for i in isl]
    allc = np.concatenate(cells)
    u, c = np.unique(allc, return_counts=True)
    lo = min(i.pts.min() for i in isl)
    hi = max(i.pts.max() for i in isl)
    for md in meshes:
        md.free()
    return int((c > 1).sum()), len(u) / res ** 2, lo, hi, len(isl)


def main():
    m = mod()
    S = bpy.context.scene.uv_ukladka

    # --- упаковка
    make_scene()
    S.margin_px = 4
    S.texture = "2048"
    t = time.perf_counter()
    r = bpy.ops.uv_ukladka.pack()
    dt = time.perf_counter() - t
    ov, cov, lo, hi, n = strict_check(m)
    print(f"  упаковка: {n} островов, {dt:.1f} c, заполнение {cov * 100:.1f}%  ({S.last})")
    ok(r == {"FINISHED"}, "оператор упаковки отработал")
    ok(ov == 0, f"нет перекрытий (клеток с перекрытием: {ov})")
    ok(lo >= -1e-6 and hi <= 1 + 1e-6, f"всё внутри плитки ({lo:.4f}..{hi:.4f})")
    ok(cov > 0.5, f"заполнение больше 50% ({cov * 100:.1f}%)")
    ours = cov

    # --- встроенная упаковка Blender для сравнения
    make_scene()
    t = time.perf_counter()
    try:
        bpy.ops.uv.pack_islands(shape_method="CONCAVE", rotate=True, margin=4 / 2048,
                                margin_method="FRACTION")
    except TypeError:
        bpy.ops.uv.pack_islands(rotate=True, margin=4 / 2048)
    dtb = time.perf_counter() - t
    ov_b, cov_b, *_ = strict_check(m)
    print(f"  встроенная (Concave): {dtb:.1f} c, заполнение {cov_b * 100:.1f}%, перекрытий {ov_b}")
    ok(ours >= cov_b * 0.97, f"плотность не хуже встроенной ({ours * 100:.1f}% vs {cov_b * 100:.1f}%)")

    # --- режимы: высокая точность + 45°
    make_scene()
    S.quality = "HIGH"
    S.rot_step = "45"
    t = time.perf_counter()
    bpy.ops.uv_ukladka.pack()
    ov, cov, lo, hi, n = strict_check(m)
    print(f"  высокая точность, 45°: {time.perf_counter() - t:.1f} c, заполнение {cov * 100:.1f}%")
    ok(ov == 0 and lo >= -1e-6 and hi <= 1 + 1e-6, "45°/высокая: без перекрытий и в плитке")
    S.quality = "NORMAL"
    S.rot_step = "90"

    # --- UDIM и неподвижный масштаб
    S.tile = 1012
    bpy.ops.uv_ukladka.pack()
    meshes = m.core.gather(bpy.context, need_3d=False)
    pts = np.concatenate([i.pts for i in m.core.all_islands(meshes)])
    ok(pts[:, 0].min() >= 1 - 1e-6 and pts[:, 0].max() <= 2 + 1e-6 and pts[:, 1].min() >= 1 - 1e-6,
       "упаковка в плитку UDIM 1012")
    S.tile = 1001
    areas_before = sorted(round(i.area_uv, 7) for i in m.core.all_islands(meshes))
    for md in meshes:
        md.free()
    S.fixed_scale = True
    bpy.ops.uv_ukladka.pack()
    meshes = m.core.gather(bpy.context, need_3d=False)
    areas_after = sorted(round(i.area_uv, 7) for i in m.core.all_islands(meshes))
    for md in meshes:
        md.free()
    ok(np.allclose(areas_before, areas_after, rtol=1e-4), "«Не менять масштаб» сохраняет площади")
    S.fixed_scale = False

    # --- упаковка выделенных вокруг невыделенных
    bpy.ops.uv_ukladka.pack()
    meshes = m.core.gather(bpy.context, need_3d=False)
    isl = m.core.all_islands(meshes)
    keep = {id(i): i.pts.copy() for i in isl[: len(isl) // 2]}
    m.core.select_islands(bpy.context, meshes, isl[len(isl) // 2:])
    S.selected_only = True
    bpy.ops.uv_ukladka.pack()
    S.selected_only = False
    ov, cov, lo, hi, n = strict_check(m)
    meshes2 = m.core.gather(bpy.context, need_3d=False)
    isl2 = m.core.all_islands(meshes2)
    still = all(np.allclose(a.pts, keep[id(b)]) for a, b in zip(isl2, isl[: len(isl) // 2]))
    for md in meshes2:
        md.free()
    ok(still, "невыделенные острова остались на месте")
    ok(ov == 0, f"выделенные уложены вокруг без перекрытий ({ov})")
    bpy.ops.mesh.select_all(action="SELECT")

    # --- одинаковые острова
    make_scene(copies=2)
    meshes = m.core.gather(bpy.context, need_3d=False)
    n_isl = len(m.core.all_islands(meshes))
    st = m.core.find_stacks(m.core.all_islands(meshes))
    for md in meshes:
        md.free()
    sizes = [1 + len(c) for c in st.values()]
    ok(all(k % 3 == 0 for k in sizes), f"одинаковые найдены по три и кратно ({len(st)} форм на {n_isl} островов)")
    S.stack = True
    bpy.ops.uv_ukladka.pack()
    S.stack = False
    meshes = m.core.gather(bpy.context, need_3d=False)
    st2 = m.core.find_stacks(m.core.all_islands(meshes), tol=1e-6)
    same = all(np.allclose(c.pts, k.pts) for k, cs in st2.items() for c in cs)
    for md in meshes:
        md.free()
    ok(same and len(st2) == len(st), "копии сложены точно на мастер")
    ov, cov, *_ = strict_check(m)
    print(f"  со сложением: заполнение (уникальной площади) {cov * 100:.1f}%")

    # --- ориентация, выравнивание, распределение
    make_scene()

    def rect_areas():
        meshes = m.core.gather(bpy.context, need_3d=False)
        out = []
        for i in m.core.all_islands(meshes):
            P = i.pts @ m.packer.rot(m.packer.min_rect_angle(i.pts)).T
            out.append((np.ptp(i.pts, 0).prod(), np.ptp(P, 0).prod()))
        for md in meshes:
            md.free()
        return out

    bpy.ops.uv_ukladka.orient()
    isl = rect_areas()
    good = sum(a <= b * 1.001 + 1e-12 for a, b in isl)  # рамка уже минимальна
    ok(good == len(isl), f"ориентация к минимальной рамке ({good}/{len(isl)})")
    bpy.ops.uv_ukladka.align(mode="LEFT")
    meshes = m.core.gather(bpy.context, need_3d=False)
    xs = [i.pts[:, 0].min() for i in m.core.all_islands(meshes)]
    for md in meshes:
        md.free()
    ok(max(xs) - min(xs) < 1e-6, "выравнивание влево")
    bpy.ops.uv_ukladka.distribute(axis="Y")

    # --- плотность текселей
    bpy.ops.uv_ukladka.texel_set()
    S.density = 300
    bpy.ops.uv_ukladka.texel_set()
    meshes = m.core.gather(bpy.context)
    d = [i.texel(2048) for i in m.core.all_islands(meshes) if i.area3d > 1e-6]
    for md in meshes:
        md.free()
    ok(np.allclose(d, 300, rtol=1e-3), f"плотность 300 px/м у всех ({min(d):.1f}..{max(d):.1f})")
    bpy.ops.uv_ukladka.texel_get()
    ok(abs(S.density - 300) < 0.5, f"измерение плотности ({S.density:.1f})")

    # --- проверка: перекрытия должны найтись после «Уравнять» без упаковки
    r = m.core.check(bpy.context, 1024, 1001, True)
    ok(r["overlaps"] > 0, f"проверка видит перекрытия ({r['overlaps']})")
    bpy.ops.uv_ukladka.pack()
    r = m.core.check(bpy.context, 1024, 1001, False)
    ok(r["overlaps"] == 0 and r["outside"] == 0, f"после упаковки чисто ({r})")

    # --- сценарий
    bpy.ops.uv_ukladka.script_template()
    res = bpy.ops.uv_ukladka.run_script()
    ok(res == {"FINISHED"}, "сценарий-пример выполнился")
    r = m.api.check()
    ok(r["overlaps"] == 0, "после сценария перекрытий нет")

    # --- объектный режим
    bpy.ops.object.mode_set(mode="OBJECT")
    res = bpy.ops.uv_ukladka.pack()
    ok(res == {"FINISHED"}, "упаковка в объектном режиме")

    # --- скорость: много островов
    clear()
    for k in range(40):
        bpy.ops.mesh.primitive_monkey_add(location=(k * 3, 0, 0), rotation=(k, k * 0.7, 0))
    for o in bpy.context.scene.objects:
        o.select_set(True)
    bpy.ops.object.join()
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(50), island_margin=0.0)
    S.threads = 0
    t = time.perf_counter()
    bpy.ops.uv_ukladka.pack()
    ov, cov, lo, hi, n = strict_check(m)
    print(f"  {n} островов: {time.perf_counter() - t:.1f} c, заполнение {cov * 100:.1f}%  ({S.last})")
    ok(ov == 0 and lo >= -1e-6 and hi <= 1 + 1e-6, "много островов: без перекрытий, в плитке")
    ok(time.perf_counter() - t < 240, "много островов: быстрее 4 минут")
    S.threads = 0


try:
    main()
except Exception:
    traceback.print_exc()
    FAILS.append("исключение")
print("TEST_RESULT", "PASS" if not FAILS else f"FAIL {FAILS}")
