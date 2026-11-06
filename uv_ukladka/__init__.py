"""UV Укладка — плотная упаковка UV-островов, выравнивание, сложение
одинаковых островов, плотность текселей, проверка развёртки и сценарии на Python."""
import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)

from . import api, core

PREFIX = "uv_ukladka"

# точность: (сетка, сколько порядков укладки перебирать)
QUALITY = {"FAST": (256, 1), "NORMAL": (384, 1), "HIGH": (384, 4), "MAX": (1024, 4)}


def _edit_poll(context):
    return context.mode in {"EDIT_MESH", "OBJECT"} and any(
        o.type == "MESH" for o in (context.objects_in_mode_unique_data
                                   if context.mode == "EDIT_MESH" else context.selected_objects))


class UKL_Settings(bpy.types.PropertyGroup):
    selected_only: BoolProperty(name="Только выделенные", default=False,
                                description="Паковать только выделенные острова")
    avoid_fixed: BoolProperty(name="Обходить остальные", default=True,
                              description="Невыделенные острова остаются на месте, "
                                          "выделенные укладываются в свободное место вокруг них")
    lock_pinned: BoolProperty(name="Не двигать закреплённые", default=True,
                              description="Острова с закреплёнными (pin) вершинами остаются на месте")
    margin_px: IntProperty(name="Отступ, px", default=8, min=0, max=256,
                           description="Расстояние между островами в пикселях текстуры")
    texture: EnumProperty(name="Текстура", default="2048", items=[
        (s, s, f"Текстура {s}×{s}") for s in ("512", "1024", "2048", "4096", "8192")])
    rotate: BoolProperty(name="Поворачивать", default=True)
    rot_step: EnumProperty(name="Шаг поворота", default="90", items=[
        ("90", "90°", "Только кратно 90° — безопасно для прямых текстур"),
        ("45", "45°", ""), ("30", "30°", ""), ("15", "15°", "Медленнее, плотнее")])
    pre_orient: BoolProperty(name="Предориентация", default=True,
                             description="Сначала повернуть каждый остров к минимальной рамке")
    normalize: BoolProperty(name="Выровнять плотность", default=False,
                            description="Перед упаковкой сделать одинаковую плотность текселей "
                                        "(размер острова по его площади в 3D)")
    stack: BoolProperty(name="Сложить одинаковые", default=False,
                        description="Одинаковые острова (копии, массивы) кладутся друг на друга "
                                    "и занимают место один раз")
    fixed_scale: BoolProperty(name="Не менять масштаб", default=False,
                              description="Только переставить острова, не масштабируя")
    tile: IntProperty(name="Плитка UDIM", default=1001, min=1001, max=1100)
    quality: EnumProperty(name="Точность", default="NORMAL", items=[
        ("FAST", "Быстро", "Сетка 256, один порядок укладки"),
        ("NORMAL", "Обычная", "Сетка 384"),
        ("HIGH", "Высокая", "Сетка 384, перебор 4 порядков укладки"),
        ("MAX", "Максимум", "Сетка 1024, перебор 4 порядков — медленно, плотнее всего")])
    threads: IntProperty(name="Потоки", default=0, min=0, max=64,
                         description="Сколько вариантов масштаба проверять параллельно (0 — авто)")
    vertical: BoolProperty(name="Длинной стороной вверх", default=True)
    density: FloatProperty(name="Плотность, px/м", default=512.0, min=0.01)
    stack_tol: FloatProperty(name="Допуск", default=0.002, min=0.0, max=0.1, precision=4,
                             description="Допустимое отличие формы одинаковых островов (доля размера)")
    script: PointerProperty(name="Сценарий", type=bpy.types.Text)
    last: StringProperty(default="")

    def pack_opts(self):
        return dict(selected_only=self.selected_only, margin_px=self.margin_px,
                    texture=int(self.texture), rotate=self.rotate,
                    rot_step=int(self.rot_step), pre_orient=self.pre_orient,
                    normalize=self.normalize, stack=self.stack,
                    lock_pinned=self.lock_pinned, avoid_fixed=self.avoid_fixed,
                    fixed_scale=self.fixed_scale, tile=self.tile,
                    resolution=QUALITY[self.quality][0], orders=QUALITY[self.quality][1],
                    threads=self.threads)


def S(context):
    return context.scene.uv_ukladka


class _Base:
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _edit_poll(context)

    def done(self, context, msg):
        S(context).last = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class UKL_OT_pack(_Base, bpy.types.Operator):
    bl_idname = f"{PREFIX}.pack"
    bl_label = "Упаковать"
    bl_description = "Плотно уложить UV-острова в плитку"

    def execute(self, context):
        wm = context.window_manager
        wm.progress_begin(0, 1)
        try:
            st = core.pack(context, **S(context).pack_opts())
        finally:
            wm.progress_end()
        if st.get("error"):
            self.report({"WARNING"}, st["error"])
            return {"CANCELLED"}
        msg = f"Уложено {st['packed']} островов за {st['time']:.1f} с"
        if st["stacked"]:
            msg += f", сложено {st['stacked']}"
        if st["failed"]:
            msg += f", не влезло {st['failed']} (уменьшите масштаб)"
        cov = core.check(context, 512, S(context).tile, False)["coverage"]
        msg += f" · заполнение {cov * 100:.1f}%"
        return self.done(context, msg)


class UKL_OT_orient(_Base, bpy.types.Operator):
    bl_idname = f"{PREFIX}.orient"
    bl_label = "Ориентировать"
    bl_description = "Повернуть выделенные острова к минимальной рамке"

    def execute(self, context):
        n = core.orient(context, True, S(context).vertical)
        return self.done(context, f"Повёрнуто островов: {n}")


class UKL_OT_straighten(_Base, bpy.types.Operator):
    bl_idname = f"{PREFIX}.straighten"
    bl_label = "Выпрямить по ребру"
    bl_description = "Повернуть остров так, чтобы выделенное UV-ребро стало горизонтальным или вертикальным"

    def execute(self, context):
        n = core.straighten(context)
        if not n:
            self.report({"WARNING"}, "Выделите ребро в UV-редакторе")
            return {"CANCELLED"}
        return self.done(context, f"Выпрямлено островов: {n}")


class UKL_OT_align(_Base, bpy.types.Operator):
    bl_idname = f"{PREFIX}.align"
    bl_label = "Выровнять"
    bl_description = "Выровнять выделенные острова по общей рамке"
    mode: EnumProperty(items=[("LEFT", "Влево", ""), ("RIGHT", "Вправо", ""),
                              ("TOP", "Вверх", ""), ("BOTTOM", "Вниз", ""),
                              ("CENTER_H", "По центру ↔", ""), ("CENTER_V", "По центру ↕", "")])

    def execute(self, context):
        return self.done(context, f"Выровнено островов: {core.align(context, self.mode)}")


class UKL_OT_distribute(_Base, bpy.types.Operator):
    bl_idname = f"{PREFIX}.distribute"
    bl_label = "Распределить"
    bl_description = "Расставить выделенные острова с равными промежутками"
    axis: EnumProperty(items=[("X", "По горизонтали", ""), ("Y", "По вертикали", "")])

    def execute(self, context):
        return self.done(context, f"Распределено островов: {core.distribute(context, self.axis)}")


class UKL_OT_stack(_Base, bpy.types.Operator):
    bl_idname = f"{PREFIX}.stack"
    bl_label = "Сложить одинаковые"
    bl_description = "Положить одинаковые выделенные острова точно друг на друга"

    def execute(self, context):
        return self.done(context, f"Сложено островов: {core.stack_similar(context, True, S(context).stack_tol)}")


class UKL_OT_select_similar(_Base, bpy.types.Operator):
    bl_idname = f"{PREFIX}.select_similar"
    bl_label = "Выделить одинаковые"
    bl_description = "Добавить к выделению острова той же формы"

    def execute(self, context):
        return self.done(context, f"Выделено островов: {core.select_similar(context, S(context).stack_tol)}")


class UKL_OT_texel_get(_Base, bpy.types.Operator):
    bl_idname = f"{PREFIX}.texel_get"
    bl_label = "Измерить"
    bl_description = "Средняя плотность текселей выделенных островов"

    def execute(self, context):
        s = S(context)
        d = core.texel_get(context, int(s.texture))
        s.density = max(d, 0.01)
        return self.done(context, f"Плотность: {d:.1f} px/м при текстуре {s.texture}")


class UKL_OT_texel_set(_Base, bpy.types.Operator):
    bl_idname = f"{PREFIX}.texel_set"
    bl_label = "Задать"
    bl_description = "Масштабировать выделенные острова до заданной плотности"

    def execute(self, context):
        s = S(context)
        n = core.texel_set(context, int(s.texture), s.density, True)
        return self.done(context, f"Плотность {s.density:.1f} px/м: островов {n}")


class UKL_OT_texel_equalize(_Base, bpy.types.Operator):
    bl_idname = f"{PREFIX}.texel_equalize"
    bl_label = "Уравнять"
    bl_description = "Сделать у выделенных островов одинаковую (среднюю) плотность текселей"

    def execute(self, context):
        n = core.texel_equalize(context, int(S(context).texture), True)
        return self.done(context, f"Уравнено островов: {n}")


class UKL_OT_check(_Base, bpy.types.Operator):
    bl_idname = f"{PREFIX}.check"
    bl_label = "Проверить развёртку"
    bl_description = "Заполнение плитки, перекрытия (выделяет их), острова за пределами плитки"

    def execute(self, context):
        r = core.check(context, 1024, S(context).tile, True)
        msg = (f"Островов {r['islands']} · заполнение {r['coverage'] * 100:.1f}% · "
               f"перекрытий {r['overlaps']} · за плиткой {r['outside']}")
        return self.done(context, msg)


class UKL_OT_run_script(bpy.types.Operator):
    bl_idname = f"{PREFIX}.run_script"
    bl_label = "Запустить сценарий"
    bl_description = "Выполнить выбранный текст как сценарий; в нём доступен модуль ukl (API аддона)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return S(context).script is not None

    def execute(self, context):
        txt = S(context).script
        g = {"__name__": "__ukl_script__", "bpy": bpy, "ukl": api, "C": context}
        try:
            exec(compile(txt.as_string(), txt.name, "exec"), g)
        except Exception as e:  # покажем ошибку пользователю, не роняя Blender
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, f"{type(e).__name__}: {e}")
            return {"CANCELLED"}
        finally:
            api.commit()
        S(context).last = f"Сценарий «{txt.name}» выполнен"
        return {"FINISHED"}


class UKL_OT_script_template(bpy.types.Operator):
    bl_idname = f"{PREFIX}.script_template"
    bl_label = "Новый сценарий"
    bl_description = "Создать пример сценария в текстовом редакторе"

    def execute(self, context):
        t = bpy.data.texts.new("ukl_сценарий.py")
        t.write(TEMPLATE)
        S(context).script = t
        return {"FINISHED"}


TEMPLATE = '''# Сценарий «UV Укладки». Модуль ukl уже подключён.
# Пример: крупные острова — с двойной плотностью, потом упаковка.

isl = ukl.islands()
big = sorted(isl, key=lambda i: i.area3d, reverse=True)[:3]
for i in big:
    i.scale(1.4)
ukl.commit()

stats = ukl.pack(margin_px=8, texture=2048, rot_step=90, fixed_scale=False)
print("Упаковка:", stats)
print("Проверка:", ukl.check())
'''


# ------------------------------------------------------------------ панели

class _Panel:
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Укладка"


class UKL_PT_main(_Panel, bpy.types.Panel):
    bl_label = "UV Укладка"

    def draw(self, context):
        s = S(context)
        col = self.layout.column(align=True)
        col.scale_y = 1.6
        col.operator(UKL_OT_pack.bl_idname, icon="UV_ISLANDSEL")
        if s.last:
            box = self.layout.box()
            box.scale_y = 0.8
            for part in s.last.split(" · "):
                box.label(text=part)


class UKL_PT_pack(_Panel, bpy.types.Panel):
    bl_label = "Упаковка"
    bl_parent_id = "UKL_PT_main"

    def draw(self, context):
        s = S(context)
        L = self.layout
        L.use_property_split = True
        L.use_property_decorate = False
        c = L.column()
        c.prop(s, "texture")
        c.prop(s, "margin_px")
        c.prop(s, "quality")
        c.prop(s, "rot_step")
        c.prop(s, "tile")
        c.prop(s, "threads")
        # галочки — на всю ширину, чтобы подписи не обрезались
        b = L.column(align=True)
        b.use_property_split = False
        b.prop(s, "rotate")
        r = b.row()
        r.active = s.rotate
        r.separator()
        r.prop(s, "pre_orient")
        b.separator()
        b.prop(s, "selected_only")
        r = b.row()
        r.active = s.selected_only
        r.separator()
        r.prop(s, "avoid_fixed")
        b.prop(s, "lock_pinned")
        b.prop(s, "stack")
        b.prop(s, "normalize")
        b.prop(s, "fixed_scale")


class UKL_PT_align(_Panel, bpy.types.Panel):
    bl_label = "Выравнивание"
    bl_parent_id = "UKL_PT_main"

    def draw(self, context):
        s = S(context)
        L = self.layout
        r = L.row(align=True)
        r.operator(UKL_OT_orient.bl_idname, icon="DRIVER_ROTATIONAL_DIFFERENCE")
        r.prop(s, "vertical", text="", icon="SORT_DESC")
        L.operator(UKL_OT_straighten.bl_idname, icon="EDGESEL")
        g = L.grid_flow(columns=3, align=True)
        for m, ic in (("LEFT", "ALIGN_LEFT"), ("CENTER_H", "ALIGN_CENTER"), ("RIGHT", "ALIGN_RIGHT"),
                      ("TOP", "ALIGN_TOP"), ("CENTER_V", "ALIGN_MIDDLE"), ("BOTTOM", "ALIGN_BOTTOM")):
            g.operator(UKL_OT_align.bl_idname, text="", icon=ic).mode = m
        r = L.row(align=True)
        r.operator(UKL_OT_distribute.bl_idname, text="Распределить ↔").axis = "X"
        r.operator(UKL_OT_distribute.bl_idname, text="Распределить ↕").axis = "Y"


class UKL_PT_stack(_Panel, bpy.types.Panel):
    bl_label = "Одинаковые острова"
    bl_parent_id = "UKL_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        L = self.layout
        L.prop(S(context), "stack_tol")
        r = L.row(align=True)
        r.operator(UKL_OT_select_similar.bl_idname, icon="RESTRICT_SELECT_OFF")
        r.operator(UKL_OT_stack.bl_idname, icon="DUPLICATE")


class UKL_PT_texel(_Panel, bpy.types.Panel):
    bl_label = "Плотность текселей"
    bl_parent_id = "UKL_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = S(context)
        L = self.layout
        L.prop(s, "density")
        r = L.row(align=True)
        r.operator(UKL_OT_texel_get.bl_idname, icon="EYEDROPPER")
        r.operator(UKL_OT_texel_set.bl_idname)
        r.operator(UKL_OT_texel_equalize.bl_idname)
        L.label(text=f"Текстура {s.texture} (из «Упаковки»)", icon="INFO")


class UKL_PT_check(_Panel, bpy.types.Panel):
    bl_label = "Проверка и сценарии"
    bl_parent_id = "UKL_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        s = S(context)
        L = self.layout
        L.operator(UKL_OT_check.bl_idname, icon="VIEWZOOM")
        L.separator()
        r = L.row(align=True)
        r.prop(s, "script", text="")
        r.operator(UKL_OT_script_template.bl_idname, text="", icon="FILE_NEW")
        L.operator(UKL_OT_run_script.bl_idname, icon="PLAY")


classes = (UKL_Settings, UKL_OT_pack, UKL_OT_orient, UKL_OT_straighten, UKL_OT_align,
           UKL_OT_distribute, UKL_OT_stack, UKL_OT_select_similar, UKL_OT_texel_get,
           UKL_OT_texel_set, UKL_OT_texel_equalize, UKL_OT_check, UKL_OT_run_script,
           UKL_OT_script_template, UKL_PT_main, UKL_PT_pack, UKL_PT_align, UKL_PT_stack,
           UKL_PT_texel, UKL_PT_check)


def _menu(self, context):
    self.layout.separator()
    self.layout.operator(UKL_OT_pack.bl_idname, text="Упаковать (UV Укладка)", icon="UV_ISLANDSEL")


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.uv_ukladka = PointerProperty(type=UKL_Settings)
    bpy.types.IMAGE_MT_uvs.append(_menu)


def unregister():
    bpy.types.IMAGE_MT_uvs.remove(_menu)
    del bpy.types.Scene.uv_ukladka
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
