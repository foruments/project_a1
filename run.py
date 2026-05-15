"""
Параметрическая оптимизация походки Unitree A1 — главная точка входа.

Запуск:
    python run.py                          # интерактивное меню
    python run.py --help                   # справка по CLI
    python run.py --action view            # прямой запуск без меню
    python run.py --action optimize ...    # см. README

Действия:
    view       — посмотреть прохождение полосы (MuJoCo viewer)
    bruteforce — перебор по сетке (H, W)
    cma        — CMA-ES, эволюционная стратегия в 8-мерном пространстве
    nsga       — NSGA-II, многокритериальный фронт
    compare    — сравнение всех методов на одном графике
    run_best   — запуск с уже найденными оптимальными параметрами
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from cpg_quadruped import (
    CPGParams, GAIT_PRESETS,
    run_episode, evaluate, scalar_objective,
    PARAM_NAMES, PARAM_BOUNDS, PARAM_NOMINAL, N_PARAMS,
    vec_to_cpgparams, clip_to_bounds,
    CORRIDOR_HALF_WIDTH,
)

try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SECTIONS = ["flat", "rough", "slope", "stairs"]
SECTION_DESC = {
    "flat":   "Плоская поверхность",
    "rough":  "Неровный рельеф (кочки)",
    "slope":  "Подъём 8°",
    "stairs": "Ступеньки (6×4 см)",
}


# ════════════════════════════════════════════════════════════════════
#                    UI-ПРИМИТИВЫ (questionary / input)
# ════════════════════════════════════════════════════════════════════

def ask_select(prompt, choices, default=None):
    if HAS_QUESTIONARY:
        if choices and isinstance(choices[0], tuple):
            opts = [questionary.Choice(label, value=val) for val, label in choices]
        else:
            opts = choices
        return questionary.select(prompt, choices=opts, default=default).ask()
    print(f"\n{prompt}")
    for i, c in enumerate(choices, 1):
        label = c[1] if isinstance(c, tuple) else c
        print(f"  {i}. {label}")
    while True:
        try:
            i = int(input("Выбор: ").strip()) - 1
            return choices[i][0] if isinstance(choices[i], tuple) else choices[i]
        except (ValueError, IndexError):
            print("Введи корректный номер.")


def ask_checkbox(prompt, choices):
    if HAS_QUESTIONARY:
        return questionary.checkbox(prompt, choices=choices).ask()
    print(f"\n{prompt} (через пробел, например '1 3 4'):")
    for i, c in enumerate(choices, 1):
        print(f"  {i}. {c}")
    nums = input("Выбор: ").strip().split()
    return [choices[int(n) - 1] for n in nums if n.isdigit()]


def ask_text(prompt, default=""):
    if HAS_QUESTIONARY:
        return questionary.text(prompt, default=str(default)).ask()
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def ask_confirm(prompt, default=True):
    if HAS_QUESTIONARY:
        return questionary.confirm(prompt, default=default).ask()
    val = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not val:
        return default
    return val.startswith("y") or val.startswith("д")


def ask_course() -> list[str]:
    print("\n📍 Выбери секции полосы препятствий (можно несколько):")
    picked = ask_checkbox(
        "Секции (Space — отметить, Enter — подтвердить):",
        [f"{s} — {SECTION_DESC[s]}" for s in SECTIONS]
    )
    if not picked:
        print("  Ничего не выбрано, использую 'flat'.")
        return ["flat"]
    chosen = [s.split(" — ")[0] for s in picked]
    if len(chosen) > 1:
        print(f"\n📋 Выбрано: {', '.join(chosen)}")
        order_str = ask_text(
            "Порядок (через пробел) или Enter — оставить как есть",
            default=" ".join(chosen),
        )
        order = order_str.strip().split()
        valid = [s for s in order if s in chosen]
        if len(valid) == len(chosen):
            return valid
        print("  Введён некорректный порядок, оставляю исходный.")
    return chosen


def ask_cpg_params() -> CPGParams:
    print("\n⚙️  Параметры походки:")
    use_defaults = ask_confirm(
        "Использовать параметры по умолчанию (trot, H=0.06, W=0.12, f=2 Hz)?")
    if use_defaults:
        return CPGParams(**GAIT_PRESETS["trot"])

    gait = ask_select("Тип походки:",
                      [(g, f"{g}") for g in GAIT_PRESETS.keys()])
    H = float(ask_text("Высота подъёма стопы H (м) [0.02-0.12]", default="0.06"))
    W = float(ask_text("Ширина шага W (м) [0.06-0.20]", default="0.12"))
    freq = float(ask_text("Частота f (Hz) [1.0-3.5]", default="2.0"))
    H = float(np.clip(H, 0.02, 0.12))
    W = float(np.clip(W, 0.06, 0.20))
    freq = float(np.clip(freq, 1.0, 3.5))
    return CPGParams(H=H, W=W, omega=2*np.pi*freq, **GAIT_PRESETS[gait])


# ════════════════════════════════════════════════════════════════════
#                    ДЕЙСТВИЕ: ПРОСМОТР
# ════════════════════════════════════════════════════════════════════

def action_view(course=None, params=None, show_plots=True,
                heading_feedback=True, max_duration=60.0,
                use_viewer=True):
    if course is None:
        course = ask_course()
    if params is None:
        params = ask_cpg_params()

    log, metrics, boundaries = run_episode(
        params, course,
        use_viewer=use_viewer,
        max_duration=max_duration,
        heading_feedback=heading_feedback,
    )

    # Сохраняем лог последнего запуска (для повторного показа графиков)
    extra = {f"metric_{k}": v for k, v in metrics.items()
             if isinstance(v, (int, float, bool)) and v is not None}
    np.savez(f"{RESULTS_DIR}/last_run.npz",
             t=log["t"], qpos=log["qpos"], qvel=log["qvel"],
             ctrl=log["ctrl"], actuator_force=log["actuator_force"],
             base_pos=log["base_pos"], base_quat=log["base_quat"],
             fell=log["fell"], fall_t=log["fall_t"] or -1,
             completed=log["completed"],
             completion_t=log["completion_t"] or -1,
             course=course,
             H=params.H, W=params.W, omega=params.omega,
             nominal_height=params.nominal_height,
             psi_FR=params.psi_FR, psi_FL=params.psi_FL,
             psi_RR=params.psi_RR, psi_RL=params.psi_RL,
             **extra)

    if show_plots:
        _show_results_window(log, metrics, params, boundaries, course)


# ════════════════════════════════════════════════════════════════════
#                    ДЕЙСТВИЕ: BRUTE-FORCE
# ════════════════════════════════════════════════════════════════════

def action_bruteforce(course=None, n_grid=6, duration=20.0, interactive=True):
    if course is None:
        course = ask_course()
    if interactive and HAS_QUESTIONARY:
        n_grid = int(ask_text(f"Размер сетки N (NxN = {n_grid*n_grid} симуляций)",
                              default=str(n_grid)))
        duration = float(ask_text("Длительность каждой симуляции (с)",
                                  default=str(duration)))

    print(f"\n{'═'*70}")
    print(f"  BRUTE-FORCE по сетке (H × W)")
    print(f"  Полоса: {' → '.join(course)}  |  Сетка: {n_grid}×{n_grid}={n_grid*n_grid}")
    print(f"{'═'*70}\n")

    H_range = np.linspace(0.02, 0.12, n_grid)
    W_range = np.linspace(0.06, 0.20, n_grid)
    J1 = np.zeros((n_grid, n_grid))
    J2 = np.zeros((n_grid, n_grid))
    J6 = np.zeros((n_grid, n_grid), dtype=int)

    t_start = time.time()
    for i, H in enumerate(H_range):
        for j, W in enumerate(W_range):
            p = CPGParams(H=H, W=W, **GAIT_PRESETS["trot"])
            _, m, _ = run_episode(p, course, use_viewer=False,
                                  max_duration=duration, realtime=False,
                                  heading_feedback=True, verbose=False)
            J1[i, j] = m["J1_speed"]
            J2[i, j] = m["J2_CoT"]
            J6[i, j] = m["J6_traverse"]
            k = i*n_grid + j
            eta = (time.time()-t_start) / (k+1) * (n_grid*n_grid - k - 1)
            mark = "✓" if J6[i,j] else "✗"
            print(f"  [{k+1:3d}/{n_grid*n_grid}] {mark} "
                  f"H={H:.3f} W={W:.3f}  J1={J1[i,j]:+.3f}  ETA {eta:.0f}с")
    print(f"\nГотово за {time.time()-t_start:.1f}с.\n")

    passable = J6.astype(bool)
    if passable.any():
        i, j = np.unravel_index(np.argmax(J1 * passable), J1.shape)
        print(f"  🏆 Макс J1: H={H_range[i]:.3f}, W={W_range[j]:.3f}  →  "
              f"v={J1[i,j]:.3f} м/с")
        i, j = np.unravel_index(np.argmin(np.where(passable, J2, np.inf)), J2.shape)
        print(f"  🏆 Мин J2:  H={H_range[i]:.3f}, W={W_range[j]:.3f}  →  "
              f"CoT={J2[i,j]:.3f}")

    course_str = "_".join(course)
    np.savez(f"{RESULTS_DIR}/optimize_{course_str}.npz",
             H_range=H_range, W_range=W_range,
             J1=J1, J2=J2, J6=J6, course=course)
    _plot_bruteforce(H_range, W_range, J1, J2, J6, course)


# ════════════════════════════════════════════════════════════════════
#                    ДЕЙСТВИЕ: CMA-ES
# ════════════════════════════════════════════════════════════════════

def action_cma(course=None, budget=40, sigma=0.20, popsize=8,
               duration=20.0, interactive=True):
    import cma
    if course is None:
        course = ask_course()
    if interactive and HAS_QUESTIONARY:
        budget = int(ask_text("Бюджет (число оценок)", default=str(budget)))
        popsize = int(ask_text("Размер популяции", default=str(popsize)))
        duration = float(ask_text("Длительность каждой симуляции (с)",
                                  default=str(duration)))

    print(f"\n{'═'*70}")
    print(f"  CMA-ES в 8-мерном пространстве")
    print(f"  Полоса:  {' → '.join(course)}")
    print(f"  Бюджет:  {budget}, популяция {popsize}, σ={sigma}")
    print(f"{'═'*70}\n")

    lo, hi = PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1]
    x0_norm = (PARAM_NOMINAL - lo) / (hi - lo)
    es = cma.CMAEvolutionStrategy(
        x0_norm, sigma,
        {'bounds': [[0]*N_PARAMS, [1]*N_PARAMS],
         'popsize': popsize, 'maxfevals': budget, 'verbose': -9}
    )

    history = []
    t_start = time.time()
    n = 0
    while not es.stop() and n < budget:
        sols_norm = es.ask()
        fits = []
        for x_norm in sols_norm:
            x_real = lo + np.array(x_norm) * (hi - lo)
            m = evaluate(x_real, course, duration=duration)
            f = scalar_objective(m)
            fits.append(f)
            history.append({"params": x_real.tolist(),
                            "J1": m["J1_speed"], "J2": m["J2_CoT"],
                            "J6": m["J6_traverse"], "obj": f})
            n += 1
            eta = (time.time()-t_start) / n * (budget - n)
            mark = "✓" if m["J6_traverse"] else "✗"
            print(f"  [{n:3d}/{budget}] {mark} v={m['J1_speed']:+.3f} "
                  f"CoT={m['J2_CoT']:.2f}  obj={f:+.2f}  ETA {eta:.0f}с")
        es.tell(sols_norm, fits)

    best_x = clip_to_bounds(lo + np.array(es.result.xbest) * (hi - lo))
    print(f"\n  🏆 Лучшие параметры:")
    for nm, v in zip(PARAM_NAMES, best_x):
        print(f"      {nm:>12s} = {v:+.4f}")
    best_m = evaluate(best_x, course, duration=duration)
    print(f"\n  Проверка: J1={best_m['J1_speed']:+.4f} м/с,  "
          f"J2={best_m['J2_CoT']:.4f},  J6={best_m['J6_traverse']}")

    course_str = "_".join(course)
    np.savez(f"{RESULTS_DIR}/cma_{course_str}.npz",
             best_x=best_x, param_names=PARAM_NAMES,
             history_J1=np.array([h["J1"] for h in history]),
             history_J2=np.array([h["J2"] for h in history]),
             history_J6=np.array([h["J6"] for h in history]),
             history_obj=np.array([h["obj"] for h in history]),
             history_params=np.array([h["params"] for h in history]),
             course=course)
    _plot_cma(history, best_m, course)


# ════════════════════════════════════════════════════════════════════
#                    ДЕЙСТВИЕ: NSGA-II
# ════════════════════════════════════════════════════════════════════

def action_nsga(course=None, pop=10, n_gen=6, duration=18.0, interactive=True):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PolynomialMutation

    if course is None:
        course = ask_course()
    if interactive and HAS_QUESTIONARY:
        pop = int(ask_text("Размер популяции", default=str(pop)))
        n_gen = int(ask_text("Число поколений", default=str(n_gen)))
        duration = float(ask_text("Длительность каждой симуляции (с)",
                                  default=str(duration)))

    total = pop * n_gen
    print(f"\n{'═'*70}")
    print(f"  NSGA-II — многокритериальная оптимизация")
    print(f"  Полоса: {' → '.join(course)}")
    print(f"  Бюджет: {pop} × {n_gen} = {total} оценок")
    print(f"{'═'*70}\n")

    class CPGProblem(Problem):
        def __init__(self):
            super().__init__(n_var=N_PARAMS, n_obj=2, n_ieq_constr=1,
                             xl=PARAM_BOUNDS[:, 0], xu=PARAM_BOUNDS[:, 1])
            self.n = 0
            self.t0 = time.time()
        def _evaluate(self, X, out, *a, **kw):
            F = np.zeros((X.shape[0], 2))
            G = np.zeros((X.shape[0], 1))
            for i in range(X.shape[0]):
                m = evaluate(X[i], course, duration=duration)
                F[i, 0] = -m["J1_speed"]
                F[i, 1] = m["J2_CoT"]
                G[i, 0] = 1.0 - m["J6_traverse"]
                self.n += 1
                mark = "✓" if m["J6_traverse"] else "✗"
                eta = (time.time()-self.t0)/self.n * (total - self.n)
                print(f"  [{self.n:3d}/{total}] {mark} v={m['J1_speed']:+.3f} "
                      f"CoT={m['J2_CoT']:.2f}  ETA {eta:.0f}с")
            out["F"] = F
            out["G"] = G

    res = minimize(
        CPGProblem(),
        NSGA2(pop_size=pop,
              crossover=SBX(prob=0.9, eta=15),
              mutation=PolynomialMutation(eta=20)),
        ('n_gen', n_gen), seed=42, verbose=False,
    )

    if res.X is None or len(res.X) == 0:
        print("\n  ⚠ Парето-фронт пуст. Все решения провалили J6.")
        return

    J1p = -res.F[:, 0]
    J2p = res.F[:, 1]
    order = np.argsort(-J1p)
    print(f"\n  ✓ Парето-фронт: {len(J1p)} решений\n")
    print(f"  {'#':>3s} {'J1':>8s} {'J2':>8s} {'H':>6s} {'W':>6s} {'f':>5s}")
    for r, i in enumerate(order):
        print(f"  {r+1:>3d} {J1p[i]:>8.4f} {J2p[i]:>8.4f} "
              f"{res.X[i,0]:>6.3f} {res.X[i,1]:>6.3f} {res.X[i,2]:>5.2f}")

    course_str = "_".join(course)
    np.savez(f"{RESULTS_DIR}/nsga_{course_str}.npz",
             pareto_X=res.X, pareto_F=res.F,
             J1_pareto=J1p, J2_pareto=J2p,
             param_names=PARAM_NAMES, course=course)
    _plot_nsga(res, J1p, J2p, course)


# ════════════════════════════════════════════════════════════════════
#                    ДЕЙСТВИЕ: COMPARE
# ════════════════════════════════════════════════════════════════════

def action_compare(course=None):
    if course is None:
        course = ask_course()
    course_str = "_".join(course)
    found = []
    fig, ax = plt.subplots(figsize=(10, 7))

    bf = f"{RESULTS_DIR}/optimize_{course_str}.npz"
    if os.path.exists(bf):
        d = np.load(bf)
        J1 = d["J1"].flatten(); J2 = d["J2"].flatten(); J6 = d["J6"].flatten()
        passable = J6.astype(bool)
        ax.scatter(J1[~passable], J2[~passable], c='lightgrey', marker='x', s=20)
        ax.scatter(J1[passable], J2[passable], c='C0', s=40, alpha=0.4,
                   label=f'brute-force ({passable.sum()}/{len(J1)})')
        if passable.any():
            pareto = _pareto(J1[passable], J2[passable])
            ax.plot(J1[passable][pareto], J2[passable][pareto], 'C0-',
                    lw=2, marker='o', mec='black', ms=10,
                    label='Парето brute-force')
        found.append("brute-force")

    cma_path = f"{RESULTS_DIR}/cma_{course_str}.npz"
    if os.path.exists(cma_path):
        d = np.load(cma_path)
        J1 = d["history_J1"]; J2 = d["history_J2"]; J6 = d["history_J6"]
        passable = J6.astype(bool)
        ax.scatter(J1[passable], J2[passable], c='C1', s=40, alpha=0.4, marker='s',
                   label=f'CMA-ES ({passable.sum()}/{len(J1)})')
        if passable.any():
            pareto = _pareto(J1[passable], J2[passable])
            ax.plot(J1[passable][pareto], J2[passable][pareto], 'C1-',
                    lw=2, marker='s', mec='black', ms=10,
                    label='Парето CMA-ES')
        found.append("CMA-ES")

    nsga_path = f"{RESULTS_DIR}/nsga_{course_str}.npz"
    if os.path.exists(nsga_path):
        d = np.load(nsga_path)
        J1p = d["J1_pareto"]; J2p = d["J2_pareto"]
        order = np.argsort(J1p)
        ax.plot(J1p[order], J2p[order], 'C3-', lw=2.5,
                marker='D', mec='black', ms=12,
                label=f'NSGA-II ({len(J1p)} точек)')
        found.append("NSGA-II")

    if not found:
        plt.close(fig)
        print(f"\n  ⚠ Для полосы '{course_str}' не найдено результатов.")
        print(f"  Сначала запусти один из оптимизаторов.")
        return

    ax.set_xlabel("$J_1$: скорость, м/с (→ больше = лучше)", fontsize=12)
    ax.set_ylabel("$J_2$ CoT (↓ меньше = лучше)", fontsize=12)
    ax.set_title(f"Сравнение методов: «{' → '.join(course)}»",
                 fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    out_png = f"{RESULTS_DIR}/compare_{course_str}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches='tight')
    print(f"\n  ✓ Сравнение: {out_png}")
    print(f"  ✓ Методы: {', '.join(found)}\n")
    plt.show()


def _pareto(J1, J2):
    """Индексы точек на Парето-фронте (max J1, min J2)."""
    idx_sorted = np.argsort(-J1)
    res, best = [], float('inf')
    for i in idx_sorted:
        if J2[i] < best:
            res.append(i); best = J2[i]
    return np.array(sorted(res, key=lambda i: J1[i]))


# ════════════════════════════════════════════════════════════════════
#               ДЕЙСТВИЕ: ЗАПУСК С ОПТИМАЛЬНЫМИ ПАРАМЕТРАМИ
# ════════════════════════════════════════════════════════════════════

def action_run_best():
    files = sorted([f for f in os.listdir(RESULTS_DIR)
                    if f.endswith(".npz") and
                       (f.startswith("optimize_") or f.startswith("cma_")
                        or f.startswith("nsga_"))])
    if not files:
        print(f"\n  ⚠ В {RESULTS_DIR}/ нет npz с результатами.")
        return
    choice = ask_select("Файл с оптимумом:", files)
    full = os.path.join(RESULTS_DIR, choice)
    d = np.load(full, allow_pickle=True)

    course = [str(x) for x in d["course"]] if "course" in d.files else None
    if course is None:
        course = ask_course()

    if "best_x" in d.files:                   # CMA
        params = vec_to_cpgparams(d["best_x"])
        src = "CMA-ES (best_x)"
    elif "pareto_X" in d.files:                # NSGA
        J1 = d["J1_pareto"]
        i = int(np.argmax(J1))
        params = vec_to_cpgparams(d["pareto_X"][i])
        src = f"NSGA-II (#{i+1}, J1={J1[i]:.3f})"
    else:                                      # brute-force
        J1 = d["J1"]; J6 = d["J6"]
        passable = J6.astype(bool)
        if not passable.any():
            print("  ⚠ В brute-force нет валидных решений.")
            return
        idx = np.unravel_index(np.argmax(J1 * passable), J1.shape)
        H = float(d["H_range"][idx[0]])
        W = float(d["W_range"][idx[1]])
        params = CPGParams(H=H, W=W, **GAIT_PRESETS["trot"])
        src = f"brute-force (H={H:.3f}, W={W:.3f}, J1={J1[idx]:.3f})"

    print(f"\n  Запускаю с оптимумом: {src}")
    action_view(course=course, params=params)


# ════════════════════════════════════════════════════════════════════
#                          ГРАФИКИ
# ════════════════════════════════════════════════════════════════════

def _quat_to_rpy(quat):
    w, x, y, z = quat.T
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def _show_results_window(log, metrics, params, boundaries, course_sections):
    """6 панелей + таблица метрик."""
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    t = log["t"]
    base = log["base_pos"]
    qvel = log["qvel"]
    tau = log["actuator_force"]
    dt = (t[1] - t[0]) if len(t) > 1 else 0.001

    SCOLORS = {"flat": "#E0E0E0", "rough": "#A0D6A0",
               "slope": "#F0C8A0", "stairs": "#C8A0D6"}

    # 1. Траектория CoM
    ax = fig.add_subplot(gs[0, 0])
    if len(boundaries) > 1:
        for x_s, x_e, name in boundaries:
            ax.axvspan(x_s, x_e, color=SCOLORS.get(name, "#CCC"), alpha=0.5)
            ax.text((x_s+x_e)/2, base[:, 1].max() + 0.08, name, ha='center',
                    fontsize=8, fontweight='bold', alpha=0.7)
    ax.plot(base[:, 0], base[:, 1], 'b-', lw=2)
    ax.plot(base[0, 0], base[0, 1], 'go', ms=10, label='старт')
    ax.plot(base[-1, 0], base[-1, 1], 'rs', ms=10, label='финиш')
    ax.axhline(+CORRIDOR_HALF_WIDTH, color='red', ls=':', alpha=0.4)
    ax.axhline(-CORRIDOR_HALF_WIDTH, color='red', ls=':', alpha=0.4)
    ax.set_xlabel("x, м"); ax.set_ylabel("y, м")
    ax.set_title("Траектория CoM (вид сверху)")
    ax.grid(True, alpha=0.3); ax.legend(loc='upper left', fontsize=8)

    # 2. z(t)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, base[:, 2], 'b-', lw=1.5)
    ax.axhline(params.nominal_height, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel("t, с"); ax.set_ylabel("z, м")
    ax.set_title("Высота trunk"); ax.grid(True, alpha=0.3)

    # 3. Углы корпуса
    ax = fig.add_subplot(gs[0, 2])
    roll, pitch, yaw = _quat_to_rpy(log["base_quat"])
    ax.plot(t, np.degrees(roll), 'r-', lw=1.5,
            label=f'крен (max={np.degrees(metrics["J4_roll"]):.1f}°)')
    ax.plot(t, np.degrees(pitch), 'b-', lw=1.5,
            label=f'дифф. (max={np.degrees(metrics["J5_pitch"]):.1f}°)')
    ax.plot(t, np.degrees(yaw), 'g-', lw=1, alpha=0.6, label='рыскание')
    ax.set_xlabel("t, с"); ax.set_ylabel("угол, °")
    ax.set_title("Углы корпуса"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. v_x(t)
    ax = fig.add_subplot(gs[1, 0])
    vx = np.gradient(base[:, 0], dt)
    ax.plot(t, vx, 'b-', lw=1, alpha=0.5)
    w = max(int(0.5/dt), 1)
    vx_sm = np.convolve(vx, np.ones(w)/w, mode='same')
    ax.plot(t, vx_sm, 'r-', lw=2, label=f'avg={metrics["J1_speed"]:.3f} м/с')
    ax.set_xlabel("t, с"); ax.set_ylabel("v_x, м/с")
    ax.set_title("$J_1$: скорость")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 5. Мощность
    ax = fig.add_subplot(gs[1, 1])
    power = np.abs(tau * qvel[:, 6:]).sum(axis=1)
    ax.plot(t, power, 'g-', lw=1.2)
    ax.set_xlabel("t, с"); ax.set_ylabel("P, Вт")
    ax.set_title(f"Мощность приводов. avg={power.mean():.1f} Вт")
    ax.grid(True, alpha=0.3)

    # 6. z̈
    ax = fig.add_subplot(gs[1, 2])
    zdd = np.gradient(np.gradient(base[:, 2], dt), dt)
    ax.plot(t, zdd, 'm-', lw=1)
    rms = metrics["J3_smooth"]
    ax.axhline(rms, color='r', ls='--', alpha=0.6, label=f'RMS={rms:.2f}')
    ax.axhline(-rms, color='r', ls='--', alpha=0.6)
    ax.set_xlabel("t, с"); ax.set_ylabel("z̈, м/с²")
    ax.set_title("$J_3$: вертикальное ускорение")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 7. Таблица
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    course_str = " → ".join(course_sections)
    if log["fell"]:
        status = f"[X] УПАЛ при t={log['fall_t']:.2f}с"
    elif metrics.get("out_of_corridor"):
        status = "[X] ВЫШЕЛ ИЗ КОРИДОРА"
    elif log["completed"]:
        status = f"[OK] ПРОЙДЕНО за {log['completion_t']:.2f}с"
    else:
        status = f"[...] Прошёл {metrics['distance']:.2f}м"
    table_data = [
        ["Параметр", "Значение", "Цель"],
        ["Полоса", course_str, ""],
        ["CPG", f"H={params.H:.3f}, W={params.W:.3f}, f={params.omega/(2*np.pi):.2f} Hz", ""],
        ["", "", ""],
        ["J₁ — скорость",        f"{metrics['J1_speed']:+.4f} м/с",  "max"],
        ["J₂ — CoT",             f"{metrics['J2_CoT']:.4f}",         "min"],
        ["J₃ — RMS z̈",          f"{metrics['J3_smooth']:.4f} м/с²", "min"],
        ["J₄ — макс. крен",      f"{np.degrees(metrics['J4_roll']):.2f}°",  "min"],
        ["J₅ — макс. дифферент", f"{np.degrees(metrics['J5_pitch']):.2f}°", "min"],
        ["J₆ — проходимость",    status, "=1"],
        ["Боковое смещение",     f"{metrics['max_lateral']:.3f} м "
                                 f"(допуск < {CORRIDOR_HALF_WIDTH*0.95:.2f})", ""],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[0.30, 0.55, 0.15])
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.4)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#D5E8F0')
        elif r == 9:
            cell.set_facecolor('#D0F0C0' if metrics["J6_traverse"] else '#F0C0C0')

    fig.suptitle("Результаты прохождения полосы",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(f"{RESULTS_DIR}/last_run.png", dpi=110, bbox_inches='tight')
    plt.show()


def _plot_bruteforce(H, W, J1, J2, J6, course):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    extent = [W[0], W[-1], H[0], H[-1]]
    passable = J6.astype(bool)
    HH = np.meshgrid(H, W, indexing='ij')[0]

    ax = axes[0]
    M = J1.copy(); M[~passable] = np.nan
    im = ax.imshow(M, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    ax.set_xlabel("W, м"); ax.set_ylabel("H, м")
    ax.set_title("$J_1$: скорость (max)"); plt.colorbar(im, ax=ax)

    ax = axes[1]
    M = J2.copy(); M[~passable] = np.nan
    im = ax.imshow(M, origin='lower', extent=extent, aspect='auto', cmap='viridis_r')
    ax.set_xlabel("W, м"); ax.set_ylabel("H, м")
    ax.set_title("$J_2$: CoT (min)"); plt.colorbar(im, ax=ax)

    ax = axes[2]
    ax.scatter(J1[~passable], J2[~passable], c='lightgrey', marker='x', label='упал')
    if passable.any():
        sc = ax.scatter(J1[passable], J2[passable], c=HH[passable],
                        s=60, cmap='plasma', edgecolor='black', linewidth=0.5)
        plt.colorbar(sc, ax=ax, label="H, м")
        pareto = _pareto(J1[passable], J2[passable])
        ax.plot(J1[passable][pareto], J2[passable][pareto],
                'r-', lw=2, marker='o', mec='black', ms=10, label='Парето')
    ax.set_xlabel("$J_1$, м/с"); ax.set_ylabel("$J_2$ CoT")
    ax.set_title("Парето J₁ vs J₂"); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Brute-force на полосе {' → '.join(course)}", fontweight='bold')
    course_str = "_".join(course)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/optimize_{course_str}.png", dpi=110, bbox_inches='tight')
    plt.show()


def _plot_cma(history, best_m, course):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    J1_arr = np.array([h["J1"] for h in history])
    J2_arr = np.array([h["J2"] for h in history])
    J6_arr = np.array([h["J6"] for h in history])
    objs = np.array([h["obj"] for h in history])
    passable = J6_arr.astype(bool)

    ax = axes[0]
    iters = np.arange(len(objs))
    ax.scatter(iters[~passable], objs[~passable], c='lightgrey', marker='x',
               label='не прошёл')
    ax.scatter(iters[passable], objs[passable], c='C0', label='прошёл')
    running_min = np.minimum.accumulate(objs)
    ax.plot(iters, running_min, 'r-', lw=2, label='лучшее к моменту')
    ax.set_xlabel("номер оценки"); ax.set_ylabel("Φ (min)")
    ax.set_title("Сходимость CMA-ES"); ax.legend(); ax.grid(True, alpha=0.3)
    if (objs[~passable] > 100).any():
        ax.set_yscale("symlog")

    ax = axes[1]
    if passable.any():
        sc = ax.scatter(J1_arr[passable], J2_arr[passable], c=iters[passable],
                        s=50, cmap='viridis', edgecolor='black', linewidth=0.4)
        plt.colorbar(sc, ax=ax, label="номер оценки")
        pareto = _pareto(J1_arr[passable], J2_arr[passable])
        ax.plot(J1_arr[passable][pareto], J2_arr[passable][pareto],
                'r-', lw=2, marker='o', mec='black', ms=10, label='Парето')
    ax.set_xlabel("$J_1$, м/с"); ax.set_ylabel("$J_2$ CoT")
    ax.set_title("Валидные оценки CMA-ES")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f"CMA-ES на полосе {' → '.join(course)} "
                 f"(лучшее v={best_m['J1_speed']:.3f} м/с)", fontweight='bold')
    plt.tight_layout()
    course_str = "_".join(course)
    plt.savefig(f"{RESULTS_DIR}/cma_{course_str}.png", dpi=110, bbox_inches='tight')
    plt.show()


def _plot_nsga(res, J1p, J2p, course):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = np.argsort(J1p)

    ax = axes[0]
    try:
        F = res.pop.get("F"); G = res.pop.get("G")
        all_J1 = -F[:, 0]; all_J2 = F[:, 1]
        feas = (G[:, 0] <= 0)
        ax.scatter(all_J1[~feas], all_J2[~feas], c='lightgrey', marker='x')
        ax.scatter(all_J1[feas], all_J2[feas], c='C0', s=40, alpha=0.6,
                   label='финальная популяция')
    except Exception:
        pass
    ax.plot(J1p[order], J2p[order], 'r-o', lw=2, mec='black', ms=10,
            label='Парето-фронт')
    ax.set_xlabel("$J_1$, м/с"); ax.set_ylabel("$J_2$ CoT")
    ax.set_title(f"Парето-фронт ({len(J1p)} решений)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    sx = res.X[order]
    ax.plot(range(len(order)), sx[:, 0], 'o-', label='H')
    ax.plot(range(len(order)), sx[:, 1], 's-', label='W')
    ax.plot(range(len(order)), sx[:, 2]/4, '^-', label='freq/4')
    ax.set_xlabel("# на фронте"); ax.set_ylabel("значение")
    ax.set_title("Параметры вдоль фронта"); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f"NSGA-II на полосе {' → '.join(course)}", fontweight='bold')
    plt.tight_layout()
    course_str = "_".join(course)
    plt.savefig(f"{RESULTS_DIR}/nsga_{course_str}.png", dpi=110, bbox_inches='tight')
    plt.show()


# ════════════════════════════════════════════════════════════════════
#                  ИНТЕРАКТИВНОЕ МЕНЮ
# ════════════════════════════════════════════════════════════════════

ACTIONS = [
    ("view",       "🎬 Запустить симуляцию (MuJoCo viewer + графики)"),
    ("bruteforce", "📊 Brute-force оптимизация по сетке (H, W)"),
    ("cma",        "🧬 CMA-ES оптимизация (8 параметров)"),
    ("nsga",       "🎯 NSGA-II многокритериальная оптимизация"),
    ("compare",    "📈 Сравнить результаты всех методов"),
    ("run_best",   "🏆 Запустить с найденными оптимальными параметрами"),
    ("quit",       "❌ Выход"),
]


def interactive_menu():
    print(r"""
╔═══════════════════════════════════════════════════════════════════════╗
║   ПАРАМЕТРИЧЕСКАЯ ОПТИМИЗАЦИЯ ПОХОДКИ UNITREE A1 (CPG-контроллер)    ║
║   Дисциплина «Разработка и оптимизация мехатронных систем», ИТМО     ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
    if not HAS_QUESTIONARY:
        print("ℹ️  Совет: pip install questionary — даст красивое меню со стрелочками\n")

    while True:
        action = ask_select("Что делаем?", ACTIONS)
        if action == "quit" or action is None:
            print("\nДо встречи 👋")
            break
        print()
        try:
            if action == "view":
                action_view()
            elif action == "bruteforce":
                action_bruteforce()
            elif action == "cma":
                action_cma()
            elif action == "nsga":
                action_nsga()
            elif action == "compare":
                action_compare()
            elif action == "run_best":
                action_run_best()
        except KeyboardInterrupt:
            print("\n⏸  Прервано. Возвращаюсь в меню.\n")
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            import traceback; traceback.print_exc()
        print()
        if not ask_confirm("Вернуться в главное меню?", default=True):
            break


# ════════════════════════════════════════════════════════════════════
#                          CLI
# ════════════════════════════════════════════════════════════════════

def parse_cli():
    ap = argparse.ArgumentParser(
        description="Параметрическая оптимизация походки Unitree A1.")
    ap.add_argument("--action", choices=["view", "bruteforce", "cma", "nsga",
                                         "compare", "run_best"],
                    help="Если задан — запуск без меню")
    ap.add_argument("--course", nargs="+", choices=SECTIONS,
                    help="Список секций полосы (например: flat rough)")
    ap.add_argument("--H", type=float, help="Высота подъёма стопы (м)")
    ap.add_argument("--W", type=float, help="Ширина шага (м)")
    ap.add_argument("--freq", type=float, help="Частота CPG (Hz)")
    ap.add_argument("--gait", choices=list(GAIT_PRESETS.keys()), default="trot")
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--max-duration", type=float, default=60.0)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--gen", type=int, default=6)
    return ap.parse_args()


def main():
    args = parse_cli()
    if args.action is None:
        interactive_menu()
        return

    course = args.course or ["flat"]
    # В CLI-режиме всегда подставляем дефолты, чтобы не было интерактивных
    # запросов. Пользователь, если хочет интерактив — запускает без --action.
    H = args.H if args.H is not None else 0.06
    W = args.W if args.W is not None else 0.12
    freq = args.freq if args.freq is not None else 2.0
    params = CPGParams(H=H, W=W, omega=2*np.pi*freq, **GAIT_PRESETS[args.gait])

    if args.action == "view":
        action_view(course=course, params=params,
                    show_plots=not args.no_plots,
                    heading_feedback=not args.no_feedback,
                    max_duration=args.max_duration,
                    use_viewer=not args.no_viewer)
    elif args.action == "bruteforce":
        action_bruteforce(course=course, n_grid=args.n,
                          duration=args.max_duration, interactive=False)
    elif args.action == "cma":
        action_cma(course=course, budget=args.budget, popsize=args.popsize,
                   duration=args.max_duration, interactive=False)
    elif args.action == "nsga":
        action_nsga(course=course, pop=args.pop, n_gen=args.gen,
                    duration=args.max_duration, interactive=False)
    elif args.action == "compare":
        action_compare(course=course)
    elif args.action == "run_best":
        action_run_best()


if __name__ == "__main__":
    main()
