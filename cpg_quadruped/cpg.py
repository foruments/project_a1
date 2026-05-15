"""
CPG-контроллер для четвероногого робота Unitree A1.

Структура (см. ЦЗП, рис. 1):
    Верхний уровень — CPG: фазовые осцилляторы для каждой из 4 ног.
                      Фаза каждой ноги φ_i(t) = ω·t + ψ_i (mod 2π).
    Средний уровень — Генерация траектории стопы p_foot(t) = f(φ, H, W).
                      Используется semi-эллиптическая траектория:
                        - в фазе переноса (swing, φ ∈ [0, π)) стопа поднимается
                          по полу-эллипсу высотой H и шириной W;
                        - в фазе опоры (stance, φ ∈ [π, 2π)) стопа идёт
                          по прямой назад вдоль земли (отрицательное смещение по X).
    Нижний уровень — IK (см. kinematics.py) + PD-регулирование на приводах
                     (PD реализуется внутри MuJoCo через kp/kd актуаторов
                     или вручную в torque-actuated режиме).

Параметры оптимизации (8 непрерывных, согласно ЦЗП):
    H              — высота траектории стопы (m)
    W              — ширина шага (m)
    omega          — частота CPG (рад/с)
    psi_FL, psi_FR, psi_HL, psi_HR — фазовые сдвиги ног (рад)
    K_p, K_d       — коэффициенты PD (управляются на уровне симуляции)

Типы походок задаются фазовыми сдвигами:
    trot  (рысь):    FL и RR в фазе; FR и RL в противофазе
    pace  (иноходь): FL и RL в фазе; FR и RR в противофазе
    bound (галоп):   передние в фазе, задние в фазе, между ними сдвиг π
"""
from dataclasses import dataclass, field
import numpy as np

from .kinematics import leg_ik, LEG_SIDE_SIGN, D_HIP

LEG_NAMES = ["FR", "FL", "RR", "RL"]


@dataclass
class CPGParams:
    """Параметры CPG-контроллера. Это и есть оптимизируемые переменные."""
    # --- геометрия траектории стопы ---
    H: float = 0.06          # высота подъёма стопы в swing (m)
    W: float = 0.12          # длина шага вдоль X (m)
    # --- темп ---
    omega: float = 2.0 * np.pi * 2.0   # 2 Hz по умолчанию
    # --- фазовые сдвиги (rad) ---  trot по умолчанию
    psi_FR: float = 0.0
    psi_FL: float = np.pi
    psi_RR: float = np.pi
    psi_RL: float = 0.0
    # --- PD на приводах ---
    Kp: float = 60.0
    Kd: float = 2.0
    # --- номинальная поза (точка, относительно которой "колеблется" стопа) ---
    nominal_height: float = 0.27   # глубина стопы под hip (m, положительное число)

    def phase_for(self, leg: str) -> float:
        return {"FR": self.psi_FR, "FL": self.psi_FL,
                "RR": self.psi_RR, "RL": self.psi_RL}[leg]


# --- Пресеты походок (для удобства экспериментов) ------------------------
GAIT_PRESETS = {
    "trot":  {"psi_FR": 0.0,    "psi_FL": np.pi, "psi_RR": np.pi, "psi_RL": 0.0},
    "pace":  {"psi_FR": 0.0,    "psi_FL": np.pi, "psi_RR": 0.0,    "psi_RL": np.pi},
    "bound": {"psi_FR": 0.0,    "psi_FL": 0.0,    "psi_RR": np.pi, "psi_RL": np.pi},
}


def foot_trajectory(phase: float, p: CPGParams) -> np.ndarray:
    """
    Эталонная траектория одной стопы (в локальной СК hip body) для данной фазы.
    phase ∈ [0, 2π).

    Конвенция:
        phase = 0       — середина фазы STANCE (нога ровно под hip, x=0)
        phase ∈ [π/2, 3π/2)  — STANCE: стопа идёт назад (продвигая корпус вперёд)
        phase ∈ [3π/2, 5π/2)  — SWING: стопа летит вперёд по полу-эллипсу

    Такой выбор гарантирует, что при t=0 и нулевых фазовых сдвигах стопа
    стоит ровно под hip-суставом — нет начального крутящего момента,
    робот идёт прямо.

    Длительность stance = swing = π (50/50 duty cycle).

    Возвращает (x, y, z) в локальной СК hip body. Y-составляющая = 0
    (вынос D_HIP добавляется в cpg_step).
    """
    phi = phase % (2.0 * np.pi)
    nominal_z = -p.nominal_height

    # Перенесём начало координат:
    # пусть psi = phi - pi/2 (так stance начинается в psi=0)
    psi = (phi - np.pi / 2.0) % (2.0 * np.pi)

    if psi < np.pi:
        # --- STANCE ---  psi ∈ [0, π)
        # Стопа идёт от +W/2 (только что приземлилась впереди) до -W/2 (вот-вот
        # оторвётся сзади), линейно. Z = nominal_z (на земле).
        s = psi / np.pi   # [0, 1)
        x = p.W / 2.0 - p.W * s        # +W/2 → -W/2
        z = nominal_z
    else:
        # --- SWING ---  psi ∈ [π, 2π)
        # Стопа летит назад→вперёд по полу-эллипсу.
        # x от -W/2 (отрыв сзади) до +W/2 (приземление впереди).
        s = (psi - np.pi) / np.pi   # [0, 1)
        x = -p.W / 2.0 + p.W * s       # -W/2 → +W/2
        z = nominal_z + p.H * np.sin(np.pi * s)  # подъём по полу-эллипсу

    return np.array([x, 0.0, z])


def cpg_step(t: float, p: CPGParams) -> dict:
    """Получить целевые углы суставов для всех 4 ног в момент времени t."""
    out = {}
    for leg in LEG_NAMES:
        phase = p.omega * t + p.phase_for(leg)
        p_foot_in_hip_frame = foot_trajectory(phase, p)
        # Добавляем вынос ноги вдоль Y (sgn*D_HIP)
        sgn = LEG_SIDE_SIGN[leg]
        p_foot = p_foot_in_hip_frame + np.array([0.0, sgn * D_HIP, 0.0])
        q = leg_ik(p_foot, leg)
        out[leg] = q
    return out


# --- Параметры обратной связи по курсу --------------------------------
# Дифференциальный контроль: если робот отклонился от курса (yaw ≠ 0)
# или сместился вбок (y ≠ 0), регулируем длину шага левой и правой
# сторон по аналогии с дифференциальным приводом колёсных роботов.
#
# Эти коэффициенты подобраны эмпирически.
HEADING_KP_YAW = 0.30      # сила коррекции по углу yaw (рад → отн. длины шага)
HEADING_KP_Y   = 0.50      # сила коррекции по боковому смещению (м → отн. длины шага)
HEADING_MAX_CORR = 0.5     # максимальная относительная коррекция (50% от W)


def cpg_step_with_feedback(t: float, p: CPGParams,
                           base_yaw: float = 0.0,
                           base_y: float = 0.0) -> dict:
    """
    CPG-контроллер с обратной связью по курсу.

    Принцип: если робот повернулся влево (yaw > 0) или сместился влево
    (y > 0), то левая сторона должна делать шаги КОРОЧЕ, а правая —
    ДЛИННЕЕ. Это разворачивает робота вправо к нулевому курсу.

    Эквивалент дифференциального управления для колёсного робота.

    Аргументы:
        t        — время симуляции
        p        — параметры CPG
        base_yaw — текущий угол рыскания (рад)
        base_y   — текущее боковое смещение (м)

    Возвращает: словарь {имя ноги: q (3 угла)}.
    """
    # P-регулятор. Знаки коррекции подобраны эмпирически для конвенции
    # MJCF Unitree A1 (см. tests/test_heading.py).
    # При yaw < 0 (робот повернулся вправо) или y < 0 (сместился вправо)
    # коррекция положительна → правые ноги (FR, RR) шагают длиннее, левые
    # короче → робот разворачивается влево к нулевому курсу.
    correction = -HEADING_KP_YAW * base_yaw - HEADING_KP_Y * base_y
    correction = float(np.clip(correction, -HEADING_MAX_CORR, HEADING_MAX_CORR))

    # Локальные множители длины шага для каждой стороны
    W_mult = {
        "FR": 1.0 + correction,   # правая (sgn=-1)
        "RR": 1.0 + correction,
        "FL": 1.0 - correction,   # левая  (sgn=+1)
        "RL": 1.0 - correction,
    }

    out = {}
    for leg in LEG_NAMES:
        phase = p.omega * t + p.phase_for(leg)
        # Базовая траектория, но с скорректированной длиной шага.
        # Делаем это, временно "подменив" p.W через локальное преобразование.
        p_loc = foot_trajectory(phase, p)
        # масштабируем только X-компоненту относительно центра
        p_loc_scaled = np.array([p_loc[0] * W_mult[leg], p_loc[1], p_loc[2]])
        sgn = LEG_SIDE_SIGN[leg]
        p_foot = p_loc_scaled + np.array([0.0, sgn * D_HIP, 0.0])
        q = leg_ik(p_foot, leg)
        out[leg] = q
    return out


if __name__ == "__main__":
    print("=== Тест CPG-контроллера ===\n")
    p = CPGParams()
    # Проверим за один период: фаза должна возвращаться к началу
    T = 2 * np.pi / p.omega
    print(f"Период походки T = {T:.3f} s, частота {1/T:.2f} Hz")
    print(f"H = {p.H} m, W = {p.W} m")
    print(f"Походка: trot (FR↔RL в фазе, FL↔RR в фазе)\n")

    # Покажем траектории стопы для разных фаз
    print("Траектория стопы FR за период (X, Z в локальной СК hip):")
    print(f"{'t':>6s} {'phase/π':>9s} {'фаза':>10s} {'x':>9s} {'z':>9s}")
    for k in range(9):
        t = k * T / 8
        phase = p.omega * t + p.phase_for("FR")
        traj_phase = phase % (2 * np.pi)
        which = "swing" if traj_phase < np.pi else "stance"
        p_loc = foot_trajectory(phase, p)
        print(f"{t:6.3f} {phase/np.pi:9.3f}  {which:>9s}  {p_loc[0]:+8.4f} {p_loc[2]:+8.4f}")

    # Проверим, что IK выдаёт разумные углы для всех ног
    print("\nЦелевые углы суставов в момент t=0:")
    q_all = cpg_step(0.0, p)
    for leg, q in q_all.items():
        print(f"  {leg}: q = [{q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}]")
