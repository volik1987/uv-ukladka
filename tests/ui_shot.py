"""Снимок UV-редактора с панелью аддона после упаковки:
BLENDER_USER_RESOURCES=<профиль с аддоном> blender пустой.blend --python tests/ui_shot.py -- out.png
(с .blend-файлом не показывается заставка)"""
import math
import sys

import bpy

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "/tmp/ukl_ui.png"


def step1():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
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
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.0)
    bpy.context.scene.tool_settings.use_uv_select_sync = True

    win = bpy.context.window_manager.windows[0]
    big = max(win.screen.areas, key=lambda a: a.width * a.height)
    big.type = "IMAGE_EDITOR"
    big.spaces.active.mode = "UV"
    big.spaces.active.show_region_ui = True
    s = bpy.context.scene.uv_ukladka
    s.margin_px = 4
    s.quality = "HIGH"
    with bpy.context.temp_override(window=win, area=big):
        bpy.ops.uv_ukladka.pack()
    return None


def step2():
    win = bpy.context.window_manager.windows[0]
    big = max(win.screen.areas, key=lambda a: a.width * a.height)
    ui = next(r for r in big.regions if r.type == "UI")
    try:
        ui.active_panel_category = "Укладка"
    except Exception as e:
        print("CATEGORY", e)
    reg = next(r for r in big.regions if r.type == "WINDOW")
    bpy.ops.mesh.select_all(action="DESELECT")
    with bpy.context.temp_override(window=win, area=big, region=reg):
        bpy.ops.image.view_all(fit_view=True)
        bpy.ops.image.view_zoom_ratio(ratio=2.0)
    for r in big.regions:
        r.tag_redraw()
    return None


def step2b():
    win = bpy.context.window_manager.windows[0]
    big = max(win.screen.areas, key=lambda a: a.width * a.height)
    ui = next(r for r in big.regions if r.type == "UI")
    print("CATEGORY now", ui.active_panel_category)
    ui.active_panel_category = "Укладка"
    reg = next(r for r in big.regions if r.type == "WINDOW")
    for r in big.regions:
        r.tag_redraw()


def step3():
    win = bpy.context.window_manager.windows[0]
    big = max(win.screen.areas, key=lambda a: a.width * a.height)
    with bpy.context.temp_override(window=win, area=big):
        bpy.ops.screen.screenshot_area(filepath=OUT)


def step4():
    bpy.ops.wm.quit_blender()


bpy.app.timers.register(step1, first_interval=1.0)
bpy.app.timers.register(step2, first_interval=8.0)
bpy.app.timers.register(step2b, first_interval=9.0)
bpy.app.timers.register(step3, first_interval=10.5)
bpy.app.timers.register(step4, first_interval=11.5)
