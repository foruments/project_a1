"""
Генерация MJCF-сцен с разными типами рельефа для тестирования робота.

Доступные полигоны:
    "flat"   — плоская поверхность (контроль)
    "rough"  — неровный рельеф из множества низких "кочек"
    "slope"  — подъём с заданным углом
    "stairs" — ступеньки

Каждая функция возвращает строку XML-сцены, которую можно загрузить через
    mujoco.MjModel.from_xml_string(xml_str)

Робот всегда стартует в (0, 0, 0.32) — высота trunk над уровнем земли в
данной точке.
"""

from pathlib import Path
import os

def _find_a1_dir() -> str:
    env = os.environ.get("A1_DIR")
    if env and Path(env).exists():
        return env

    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "mujoco_menagerie" / "unitree_a1",
        project_root.parent / "mujoco_menagerie" / "unitree_a1",
    ]
    for c in candidates:
        if (c / "a1.xml").exists():
            return str(c)

    raise FileNotFoundError(
        "Папка unitree_a1 не найдена.\n"
        "Запусти: bash install.sh\n"
        "Или: export A1_DIR=/path/to/unitree_a1"
    )

A1_DIR = _find_a1_dir()


# --- Общая преамбула + ассеты -------------------------------------------
PREAMBLE = f"""
<mujoco model="a1 + custom terrain">
  <include file="{A1_DIR}/a1.xml"/>

  <statistic center="0 0 0.1" extent="2.0"/>
  <option timestep="0.002"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
    <material name="rock" rgba="0.5 0.4 0.35 1"/>
    <material name="wood" rgba="0.7 0.5 0.3 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
"""

POSTFIX = """
  </worldbody>
</mujoco>
"""


def build_flat() -> str:
    """Плоская поверхность."""
    body = """
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
"""
    return PREAMBLE + body + POSTFIX


def build_rough(seed: int = 0, n_bumps: int = 80,
                bump_height_max: float = 0.025) -> str:
    """
    Неровная поверхность: пол + куча маленьких "кочек" (capsules).
    Кочки разбросаны псевдослучайно, но детерминированно.
    """
    import random
    rng = random.Random(seed)
    bumps = []
    # площадь — прямоугольник 8x4 м, начиная с x=0.5 (чтобы робот не стартовал на кочке)
    for i in range(n_bumps):
        x = rng.uniform(0.5, 8.0)
        y = rng.uniform(-2.0, 2.0)
        h = rng.uniform(0.005, bump_height_max)
        r = rng.uniform(0.04, 0.08)
        bumps.append(
            f'    <geom name="bump_{i}" type="cylinder" '
            f'pos="{x:.3f} {y:.3f} {h/2:.3f}" '
            f'size="{r:.3f} {h/2:.3f}" material="rock"/>'
        )
    body = """
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
""" + "\n".join(bumps)
    return PREAMBLE + body + POSTFIX


def build_slope(angle_deg: float = 10.0) -> str:
    """
    Подъём с заданным углом. Склон делаем как наклонённый box большой
    толщины (1 м), который начинается в x=0.7 и идёт вверх.

    ВАЖНО: a1.xml использует compiler angle="radian", и эта настройка
    распространяется на всё включающее XML. Поэтому axisangle тоже
    должен задаваться в радианах.
    """
    import math
    angle_rad = math.radians(angle_deg)
    length = 6.0
    thickness = 1.0
    half_w = 5.0
    # См. вывод формулы в коммите: хотим, чтобы (0.7, 0, 0) было
    # передним нижним углом верхней грани.
    a = -angle_rad
    dx = (length/2) * math.cos(a) + (-thickness/2) * math.sin(a)
    dz = -(length/2) * math.sin(a) + (-thickness/2) * math.cos(a)
    cx = 0.7 + dx
    cz = 0.0 + dz
    body = f"""
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <body name="ramp" pos="{cx:.3f} 0 {cz:.3f}"
          axisangle="0 1 0 {-angle_rad:.4f}">
      <geom name="ramp_geom" type="box"
            size="{length/2:.3f} {half_w:.3f} {thickness/2:.3f}"
            material="wood"/>
    </body>
"""
    return PREAMBLE + body + POSTFIX


def build_stairs(n_steps: int = 6, step_h: float = 0.05,
                 step_d: float = 0.25) -> str:
    """
    Ступеньки. Каждая ступенька — короткий box. Начинаются в x=1.0.
    step_h — высота одной ступеньки, step_d — глубина.
    """
    geoms = []
    geoms.append(
        '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>'
    )
    start_x = 1.0
    for i in range(n_steps):
        x_center = start_x + (i + 0.5) * step_d
        z_top    = (i + 1) * step_h
        # box: половина-размеры (sx, sy, sz)
        sx = step_d / 2
        sy = 1.0
        sz = z_top / 2
        geoms.append(
            f'    <geom name="step_{i}" type="box" '
            f'pos="{x_center:.3f} 0 {z_top/2:.4f}" '
            f'size="{sx:.4f} {sy:.4f} {sz:.4f}" material="wood"/>'
        )
    body = "\n".join(geoms)
    return PREAMBLE + body + POSTFIX


# Ширина коридора прохождения (по Y). Робот стартует в центре (y=0).
# Все препятствия покрывают этот коридор полностью; по краям стоят
# невидимые стенки, не дающие сойти с трассы.
CORRIDOR_HALF_WIDTH = 1.0   # м → коридор шириной 2 м


def build_course(sections=("flat", "rough", "slope", "stairs"),
                 section_length: float = 3.0,
                 add_walls: bool = True) -> str:
    """
    Полоса препятствий: несколько секций друг за другом, ограниченная
    стенками по бокам (если add_walls=True).

    Все препятствия покрывают полную ширину коридора
    (y ∈ [-CORRIDOR_HALF_WIDTH, +CORRIDOR_HALF_WIDTH]), так что робот
    физически не может их обойти. Дополнительно по краям коридора стоят
    тонкие невидимые стенки, не дающие соскочить с трассы.

    sections: кортеж из {"flat", "rough", "slope", "stairs"}.
    section_length: длина каждой секции по X в метрах.

    Возвращает (xml_string, section_boundaries).
    """
    import math
    geoms = ['    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>']
    boundaries = []
    x_cursor = 0.0
    counter = 0
    H = CORRIDOR_HALF_WIDTH

    for sec_name in sections:
        x_start = x_cursor
        x_end = x_cursor + section_length

        if sec_name == "flat":
            # просто продолжение пола — ничего добавлять не нужно
            pass

        elif sec_name == "rough":
            # Кочки распределяем по всему коридору, плотнее (60 вместо 30),
            # чтобы робот не мог "проскользнуть между ними"
            import random
            rng = random.Random(counter * 1000 + 42)
            n_bumps = 60
            for i in range(n_bumps):
                x = rng.uniform(x_start + 0.1, x_end - 0.1)
                y = rng.uniform(-H + 0.05, H - 0.05)
                h = rng.uniform(0.005, 0.025)
                r = rng.uniform(0.04, 0.08)
                geoms.append(
                    f'    <geom name="bump_{counter}_{i}" type="cylinder" '
                    f'pos="{x:.3f} {y:.3f} {h/2:.3f}" '
                    f'size="{r:.3f} {h/2:.3f}" material="rock"/>'
                )

        elif sec_name == "slope":
            # СИММЕТРИЧНАЯ ГОРКА: пандус вверх → плато → пандус вниз → пол.
            # Это нужно, чтобы метрика проходимости была честной: робот должен
            # реально перейти препятствие и встать на ровный пол, а не падать
            # с обрыва за подъёмом. Пиковая высота умышленно невелика
            # (≈ L_ramp_h·tan α), чтобы A1 мог её преодолеть с обычными CPG-
            # параметрами.
            angle_deg = 8.0
            angle_rad = math.radians(angle_deg)
            half_w = H                  # точно по ширине коридора
            thickness = 1.0             # толстый box, нижняя грань уходит под пол
 
            # Распределяем секцию: пандус вверх / плато / пандус вниз
            L_ramp_h = 1.0              # горизонтальная проекция каждого пандуса
            L_plat = section_length - 2.0 * L_ramp_h   # плато по остатку
            if L_plat < 0.1:            # на случай очень короткой секции
                L_plat = 0.1
                L_ramp_h = (section_length - L_plat) / 2.0
            L_ramp = L_ramp_h / math.cos(angle_rad)     # реальная длина наклонной
            peak_h = L_ramp_h * math.tan(angle_rad)     # высота плато
 
            # --- Пандус вверх. Pivot верхней грани: (x_start, 0, 0) ---
            a_up = -angle_rad
            dx = (L_ramp/2) * math.cos(a_up) + (-thickness/2) * math.sin(a_up)
            dz = -(L_ramp/2) * math.sin(a_up) + (-thickness/2) * math.cos(a_up)
            cx_up = x_start + dx
            cz_up = 0.0 + dz
            geoms.append(
                f'    <body name="ramp_up_{counter}" pos="{cx_up:.3f} 0 {cz_up:.3f}" '
                f'axisangle="0 1 0 {a_up:.4f}">\n'
                f'      <geom name="ramp_up_geom_{counter}" type="box" '
                f'size="{L_ramp/2:.3f} {half_w:.3f} {thickness/2:.3f}" '
                f'material="wood"/>\n'
                f'    </body>'
            )
 
            # --- Плато. Толстый box, верхняя грань на peak_h ---
            plat_x_start = x_start + L_ramp_h
            plat_x_end = plat_x_start + L_plat
            plat_cx = (plat_x_start + plat_x_end) / 2.0
            plat_cz = peak_h - thickness/2.0
            geoms.append(
                f'    <geom name="ramp_plat_{counter}" type="box" '
                f'pos="{plat_cx:.3f} 0 {plat_cz:.4f}" '
                f'size="{L_plat/2:.3f} {half_w:.3f} {thickness/2:.3f}" '
                f'material="wood"/>'
            )
 
            # --- Пандус вниз. Pivot верхней грани: (x_end, 0, 0) ---
            # Симметрично подъёму, но угол положительный (верх клонится в -z
            # с ростом x), и опорная точка — задний нижний угол верхней грани.
            a_dn = +angle_rad
            cx_dn = x_end - math.cos(angle_rad) * (L_ramp/2.0) \
                          - math.sin(angle_rad) * (thickness/2.0)
            cz_dn = math.sin(angle_rad) * (L_ramp/2.0) \
                          - math.cos(angle_rad) * (thickness/2.0)
            geoms.append(
                f'    <body name="ramp_dn_{counter}" pos="{cx_dn:.3f} 0 {cz_dn:.3f}" '
                f'axisangle="0 1 0 {a_dn:.4f}">\n'
                f'      <geom name="ramp_dn_geom_{counter}" type="box" '
                f'size="{L_ramp/2:.3f} {half_w:.3f} {thickness/2:.3f}" '
                f'material="wood"/>\n'
                f'    </body>'
            )

        elif sec_name == "stairs":
            # СИММЕТРИЧНАЯ ЛЕСТНИЦА: n_up ступеней вверх → плато → n_down вниз.
            # Самая правая «ступенька вниз» имеет z_top = 0, то есть это просто
            # пол — поэтому соответствующий box не создаётся, и в конце секции
            # робот оказывается на ровной поверхности.
            n_up = 3                    # число ступеней на подъём
            n_down = 3                  # число ступеней на спуск (последняя = пол)
            plat_segments = 2           # ширина плато в единицах step_d
            total_segments = n_up + plat_segments + n_down
            step_d = section_length / total_segments
            step_h = 0.04
            peak_h = n_up * step_h
 
            # Подъём: каждый box стоит на полу (z от 0 до z_top)
            for i in range(n_up):
                xc = x_start + (i + 0.5) * step_d
                z_top = (i + 1) * step_h
                geoms.append(
                    f'    <geom name="step_up_{counter}_{i}" type="box" '
                    f'pos="{xc:.3f} 0 {z_top/2:.4f}" '
                    f'size="{step_d/2:.4f} {H:.4f} {z_top/2:.4f}" material="wood"/>'
                )
 
            # Плато: один широкий box высотой peak_h
            plat_x_start = x_start + n_up * step_d
            plat_w = plat_segments * step_d
            plat_cx = plat_x_start + plat_w / 2.0
            geoms.append(
                f'    <geom name="step_plat_{counter}" type="box" '
                f'pos="{plat_cx:.3f} 0 {peak_h/2:.4f}" '
                f'size="{plat_w/2:.4f} {H:.4f} {peak_h/2:.4f}" material="wood"/>'
            )
 
            # Спуск: ступени высотой (n_down-1)·step_h, ..., 1·step_h, 0
            # Box с z_top=0 пропускаем — это просто пол.
            down_x_start = plat_x_start + plat_w
            for i in range(n_down):
                z_top = (n_down - 1 - i) * step_h
                if z_top <= 0:
                    continue
                xc = down_x_start + (i + 0.5) * step_d
                geoms.append(
                    f'    <geom name="step_dn_{counter}_{i}" type="box" '
                    f'pos="{xc:.3f} 0 {z_top/2:.4f}" '
                    f'size="{step_d/2:.4f} {H:.4f} {z_top/2:.4f}" material="wood"/>'
                )
        else:
            raise ValueError(f"Неизвестная секция: {sec_name}")

        boundaries.append((x_start, x_end, sec_name))
        x_cursor = x_end
        counter += 1

    # --- Стенки по краям коридора (по всей длине полосы + запас) ----------
    if add_walls:
        total_length = x_cursor + 2.0     # с запасом 2 м после последней секции
        wall_h = 0.4                       # высота стенки (выше робота)
        wall_t = 0.05                      # толщина (тонкая)
        wall_x = total_length / 2 - 1.0    # центр стенки по X (со сдвигом -1 чтобы покрывала и старт)
        wall_xlen = total_length / 2 + 1.0 # полудлина
        # Полупрозрачные стенки чтобы виден коридор в viewer
        for side, y in [("left", +H), ("right", -H)]:
            geoms.append(
                f'    <geom name="wall_{side}" type="box" '
                f'pos="{wall_x:.3f} {y:.3f} {wall_h/2:.3f}" '
                f'size="{wall_xlen:.3f} {wall_t/2:.4f} {wall_h/2:.3f}" '
                f'rgba="0.7 0.7 0.85 0.3"/>'
            )

    body = "\n".join(geoms)
    return PREAMBLE + body + POSTFIX, boundaries


def build_scene_xml(terrain: str, **kwargs) -> str:
    """Универсальная фабрика."""
    if terrain == "flat":
        return build_flat()
    if terrain == "rough":
        return build_rough(**kwargs)
    if terrain == "slope":
        return build_slope(**kwargs)
    if terrain == "stairs":
        return build_stairs(**kwargs)
    if terrain == "course":
        # course возвращает (xml, boundaries) — здесь нужен только xml
        return build_course(**kwargs)[0]
    raise ValueError(f"Неизвестный полигон: {terrain}. "
                     "Доступно: flat / rough / slope / stairs / course")


def load_model_for_terrain(terrain: str, **kwargs):
    """
    Загружает модель MuJoCo с заданным рельефом.

    Возвращает (model, boundaries) где boundaries — список секций
    [(x_start, x_end, name), ...] для terrain="course", иначе None.
    """
    import mujoco
    import os
    boundaries = None
    if terrain == "course":
        xml, boundaries = build_course(**kwargs)
    else:
        xml = build_scene_xml(terrain, **kwargs)
    tmp_path = os.path.join(A1_DIR, "_tmp_scene.xml")
    with open(tmp_path, "w") as f:
        f.write(xml)
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return model, boundaries


if __name__ == "__main__":
    """Самотест: проверяем, что все 4 сцены загружаются в MuJoCo без ошибок."""
    import mujoco
    print("=== Тестирование MJCF-сцен ===")
    for terrain in ["flat", "rough", "slope", "stairs"]:
        try:
            xml = build_scene_xml(terrain)
            # Записываем во временный файл в директории A1, чтобы относительный
            # meshdir="assets" в a1.xml корректно резолвился.
            import tempfile, os
            tmp_path = os.path.join(A1_DIR, "_tmp_scene.xml")
            with open(tmp_path, "w") as f:
                f.write(xml)
            try:
                model = mujoco.MjModel.from_xml_path(tmp_path)
                print(f"  ✓ {terrain:8s}: nq={model.nq}, ngeom={model.ngeom}, "
                      f"nbody={model.nbody}")
            finally:
                os.remove(tmp_path)
        except Exception as e:
            print(f"  ✗ {terrain}: {e}")
