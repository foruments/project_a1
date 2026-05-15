"""
Симулятор походки Unitree A1 в MuJoCo.

Содержит:
- Базовые константы (порядок суставов, пути)
- make_initial_qpos        — расставить робота в стояк
- run_episode              — один полный эпизод (опционально с MuJoCo viewer)
- compute_metrics          — посчитать J1..J6 из лога

Это движок проекта. UI находится в run.py.
"""
import time

import mujoco
import numpy as np

from .cpg import (
    CPGParams, GAIT_PRESETS,
    cpg_step, cpg_step_with_feedback,
    LEG_NAMES,
)
from .terrain import load_model_for_terrain, CORRIDOR_HALF_WIDTH

from pathlib import Path

# --- Автоматический поиск модели Unitree A1 --------------------------
# Ищем в следующем порядке:
#   1. <папка проекта>/mujoco_menagerie/unitree_a1/  (стандартный install.sh)
#   2. <папка проекта>/../mujoco_menagerie/unitree_a1/
#   3. Переменная окружения A1_XML (если хочешь указать вручную)
import os

def _find_a1_xml() -> str:
    # Явное указание через переменную окружения — высший приоритет
    env = os.environ.get("A1_XML")
    if env and Path(env).exists():
        return env

    # Ищем относительно корня проекта (родителя пакета cpg_quadruped)
    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "mujoco_menagerie" / "unitree_a1" / "scene.xml",
        project_root.parent / "mujoco_menagerie" / "unitree_a1" / "scene.xml",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        "Модель Unitree A1 не найдена.\n"
        "Запусти установку:\n"
        "  bash install.sh\n"
        "Или укажи путь вручную через переменную окружения:\n"
        "  export A1_XML=/path/to/unitree_a1/scene.xml"
    )

A1_XML = _find_a1_xml()

JOINT_ORDER = ["FR", "FL", "RR", "RL"]
QPOS_LEG_OFFSET = {leg: 7 + 3 * i for i, leg in enumerate(JOINT_ORDER)}
CTRL_LEG_OFFSET = {leg: 3 * i for i, leg in enumerate(JOINT_ORDER)}

TRUNK_MASS_TOTAL = 12.45   # кг, из калибровки
G = 9.81


def make_initial_qpos(model: mujoco.MjModel, params: CPGParams) -> np.ndarray:
    """Расставить робота в начальной позе по фазам CPG в t=0."""
    qpos = np.zeros(model.nq)
    qpos[2] = 0.32
    qpos[3] = 1.0
    q_targets = cpg_step(0.0, params)
    for leg in JOINT_ORDER:
        off = QPOS_LEG_OFFSET[leg]
        qpos[off:off+3] = q_targets[leg]
    return qpos


def _get_base_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))


def run_episode(params: CPGParams,
                course_sections,
                use_viewer: bool = False,
                max_duration: float = 30.0,
                realtime: bool = True,
                heading_feedback: bool = True,
                verbose: bool = True):
    """
    Один полный эпизод симуляции.

    Возвращает (log_dict, metrics_dict, boundaries_list).
    """
    if len(course_sections) == 1 and course_sections[0] == "flat":
        model = mujoco.MjModel.from_xml_path(A1_XML)
        boundaries = [(0.0, 1e6, "flat")]
    else:
        model, boundaries = load_model_for_terrain(
            "course", sections=tuple(course_sections), section_length=3.0
        )
    data = mujoco.MjData(model)
    data.qpos[:] = make_initial_qpos(model, params)
    mujoco.mj_forward(model, data)

    sim_dt = model.opt.timestep
    course_length = boundaries[-1][1] if boundaries else 1e6

    log = {k: [] for k in
           ["t", "qpos", "qvel", "ctrl", "actuator_force",
            "base_pos", "base_quat"]}
    fell = False
    fall_t = None
    completed = False
    completion_t = None

    def step_once(t_sim: float):
        nonlocal fell, fall_t, completed, completion_t
        if heading_feedback:
            yaw_now = _get_base_yaw(data.qpos[3:7])
            y_now = float(data.qpos[1])
            q_targets = cpg_step_with_feedback(
                t_sim, params, base_yaw=yaw_now, base_y=y_now
            )
        else:
            q_targets = cpg_step(t_sim, params)
        for leg in JOINT_ORDER:
            off = CTRL_LEG_OFFSET[leg]
            data.ctrl[off:off+3] = q_targets[leg]
        mujoco.mj_step(model, data)

        log["t"].append(t_sim)
        log["qpos"].append(data.qpos.copy())
        log["qvel"].append(data.qvel.copy())
        log["ctrl"].append(data.ctrl.copy())
        log["actuator_force"].append(data.actuator_force.copy())
        log["base_pos"].append(data.qpos[:3].copy())
        log["base_quat"].append(data.qpos[3:7].copy())

        if data.qpos[2] < 0.1 and not fell:
            fell = True
            fall_t = t_sim
        if data.qpos[0] > course_length and not completed:
            completed = True
            completion_t = t_sim

    if verbose:
        course_str = " → ".join(course_sections)
        print(f"\n{'═'*70}")
        print(f"  Полоса препятствий:  {course_str}")
        print(f"  Длина полосы:        {course_length:.1f} м")
        print(f"  Параметры CPG:       H={params.H} W={params.W} "
              f"f={params.omega/(2*np.pi):.2f} Hz")
        print(f"  Heading feedback:    "
              f"{'ВКЛ' if heading_feedback else 'ВЫКЛ (open-loop)'}")
        print(f"{'═'*70}\n")

    t_sim = 0.0
    n_steps = int(max_duration / sim_dt)

    if use_viewer:
        from mujoco import viewer as mj_viewer
        if verbose:
            print("  Окно MuJoCo открыто. Esc или закрытие — выход.")
            print("  ПКМ: вращать камеру, прокрутка: zoom, Space: пауза.\n")
        with mj_viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 110

            step = 0
            while viewer.is_running() and step < n_steps:
                step_start = time.time()
                step_once(t_sim)
                t_sim += sim_dt
                step += 1
                viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()
                if realtime:
                    elapsed = time.time() - step_start
                    if elapsed < sim_dt:
                        time.sleep(sim_dt - elapsed)
                if (fell or completed) and step % 50 == 0:
                    if t_sim - (fall_t or completion_t) > 1.0:
                        break
    else:
        for step in range(n_steps):
            step_once(t_sim)
            t_sim += sim_dt
            if (fell or completed) and \
               (t_sim - (fall_t or completion_t)) > 0.5:
                break

    for k in list(log.keys()):
        log[k] = np.array(log[k])
    log["fell"] = fell
    log["fall_t"] = fall_t
    log["completed"] = completed
    log["completion_t"] = completion_t

    metrics = compute_metrics(log, params)

    if verbose:
        _print_summary(log, metrics, course_length)

    return log, metrics, boundaries


def _print_summary(log, metrics, course_length):
    base_final = log["base_pos"][-1]
    print(f"\n{'─'*70}")
    print(f"  РЕЗУЛЬТАТЫ")
    print(f"{'─'*70}")
    if log["fell"]:
        print(f"  ⚠ УПАЛ в момент t={log['fall_t']:.2f}с "
              f"на x={log['base_pos'][-1, 0]:.2f}м")
    elif metrics.get("out_of_corridor"):
        print(f"  ⚠ ВЫШЕЛ ИЗ КОРИДОРА (макс. |y|={metrics['max_lateral']:.2f}м)")
    elif log["completed"]:
        print(f"  ✓ ПОЛОСА ПРОЙДЕНА! Время: {log['completion_t']:.2f}с")
    else:
        print(f"  ⏱ Время вышло. Прошёл {base_final[0]:.2f}м из {course_length:.1f}м")

    print(f"\n  J₁ (средняя скорость):    {metrics['J1_speed']:+.4f} м/с")
    print(f"  J₂ (CoT):                  {metrics['J2_CoT']:.4f}")
    print(f"  J₃ (RMS z̈):                {metrics['J3_smooth']:.4f} м/с²")
    print(f"  J₄ (макс. крен):           {np.degrees(metrics['J4_roll']):.2f}°")
    print(f"  J₅ (макс. дифферент):      {np.degrees(metrics['J5_pitch']):.2f}°")
    print(f"  J₆ (проходимость):          {metrics['J6_traverse']}  "
          f"{'✓' if metrics['J6_traverse'] else '✗'}")
    print(f"  Боковое смещение (max):    {metrics['max_lateral']:.3f} м "
          f"(допуск < {CORRIDOR_HALF_WIDTH*0.95:.2f} м)")
    print(f"{'─'*70}\n")


def compute_metrics(log, params: CPGParams) -> dict:
    """Шесть метрик из ЦЗП плюс диагностика."""
    t = log["t"]
    base = log["base_pos"]
    quat = log["base_quat"]
    qvel = log["qvel"]
    tau = log["actuator_force"]
    fell = log["fell"]
    dt = t[1] - t[0] if len(t) > 1 else 0.001

    out_threshold = CORRIDOR_HALF_WIDTH * 0.95
    out_of_corridor = False
    out_of_corridor_t = None
    for i, p in enumerate(base):
        if abs(p[1]) > out_threshold:
            out_of_corridor = True
            out_of_corridor_t = float(t[i])
            break

    J6 = 1 if (not fell and not out_of_corridor) else 0

    end_idx = len(t)
    if fell and log["fall_t"] is not None:
        end_idx = min(end_idx, int(log["fall_t"] / dt))
    if out_of_corridor and out_of_corridor_t is not None:
        end_idx = min(end_idx, int(out_of_corridor_t / dt))

    if end_idx < 2:
        J1 = 0.0
    else:
        elapsed = t[end_idx-1] - t[0]
        J1 = (base[end_idx-1, 0] - base[0, 0]) / max(elapsed, 1e-3)

    joint_vels = qvel[:end_idx, 6:]
    joint_taus = tau[:end_idx, :]
    energy = np.sum(np.abs(joint_taus * joint_vels)) * dt
    dist = max(abs(base[end_idx-1, 0] - base[0, 0]), 1e-3)
    J2 = energy / (TRUNK_MASS_TOTAL * G * dist)

    if end_idx > 3:
        z = base[:end_idx, 2]
        zdd = np.gradient(np.gradient(z, dt), dt)
        J3 = float(np.sqrt(np.mean(zdd**2)))
    else:
        J3 = 0.0

    if end_idx > 1:
        q = quat[:end_idx]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        J4 = float(np.max(np.abs(roll)))
        J5 = float(np.max(np.abs(pitch)))
    else:
        J4 = J5 = 0.0

    return {
        "J1_speed":           float(J1),
        "J2_CoT":             float(J2),
        "J3_smooth":          float(J3),
        "J4_roll":            float(J4),
        "J5_pitch":           float(J5),
        "J6_traverse":        int(J6),
        "fell":               bool(fell),
        "fall_t":             log["fall_t"],
        "out_of_corridor":    bool(out_of_corridor),
        "out_of_corridor_t":  out_of_corridor_t,
        "distance":           float(base[end_idx-1, 0] - base[0, 0]),
        "max_lateral":        float(np.max(np.abs(base[:end_idx, 1]))),
        "final_z":            float(base[end_idx-1, 2]),
    }
