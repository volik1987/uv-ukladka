"""Скриптовый интерфейс «UV Укладки».

Пример (Текстовый редактор Blender → «Запустить сценарий» в панели аддона,
или из любого своего скрипта):

    from bl_ext.user_default.uv_ukladka import api as ukl   # в сценарии уже есть как `ukl`
    ukl.pack(margin_px=4, texture=4096, rot_step=45)
    for isl in ukl.islands():
        if isl.area3d < 0.01:       # мелочь — вдвое меньше
            isl.scale(0.5)
    ukl.commit()
    print(ukl.check())

Все функции работают с текущим контекстом (объекты в режиме правки или
выделенные меши в объектном режиме).
"""
import bpy

from . import core, packer

_session = None


def _ctx(context):
    return context or bpy.context


def islands(context=None, selected_only=False):
    """Острова для ручной обработки. После изменений вызовите commit().
    У острова: pts (numpy N×2), tris, area3d, area_uv, selected, pinned,
    bounds(), center(), rotate(rad), scale(k), transform(M, t), texel(tex)."""
    global _session
    if _session is not None:
        core.finish(_session, False)
    _session = core.gather(_ctx(context))
    isl = core.all_islands(_session)
    return [i for i in isl if i.selected] if selected_only else isl


def commit():
    """Записать изменения островов, полученных через islands()."""
    global _session
    if _session is not None:
        core.finish(_session, True)
        _session = None


def pack(context=None, **opts):
    """Упаковка. Параметры как у core.pack: selected_only, margin_px, texture,
    rotate, rot_step, pre_orient, normalize, stack, lock_pinned, avoid_fixed,
    fixed_scale, tile, resolution (сетка), orders (сколько порядков
    укладки перебрать), threads."""
    return core.pack(_ctx(context), **opts)


def orient(context=None, selected_only=True, vertical=True):
    return core.orient(_ctx(context), selected_only, vertical)


def align(mode="LEFT", context=None):
    return core.align(_ctx(context), mode)


def distribute(axis="X", context=None):
    return core.distribute(_ctx(context), axis)


def stack(context=None, selected_only=True, tol=0.002):
    return core.stack_similar(_ctx(context), selected_only, tol)


def texel(texture=2048, context=None):
    return core.texel_get(_ctx(context), texture)


def set_texel(density, texture=2048, context=None, selected_only=True):
    return core.texel_set(_ctx(context), texture, density, selected_only)


def check(context=None, resolution=1024, tile=1001, select_overlaps=False):
    return core.check(_ctx(context), resolution, tile, select_overlaps)


min_rect_angle = packer.min_rect_angle
