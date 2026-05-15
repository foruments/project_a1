"""
Универсальная функция оценки CPG-параметров для оптимизаторов.

Пространство оптимизации — 8 непрерывных параметров (ЦЗП):
    p[0] = H            — высота траектории стопы (м)
    p[1] = W            — ширина шага (м)
    p[2] = freq         — частота CPG (Hz)
    p[3..6] = psi_*     — фазовые сдвиги ног (рад)
    p[7] = nominal_h    — высота стояния trunk (м)
"""
import numpy as np

from .cpg import CPGParams


PARAM_NAMES = ["H", "W", "freq", "psi_FR", "psi_FL", "psi_RR", "psi_RL", "nominal_h"]

PARAM_BOUNDS = np.array([
    [0.02, 0.12],          # H
    [0.06, 0.20],          # W
    [1.0,  3.5],           # freq, Hz
    [0.0,  2*np.pi],       # psi_FR
    [0.0,  2*np.pi],       # psi_FL
    [0.0,  2*np.pi],       # psi_RR
    [0.0,  2*np.pi],       # psi_RL
    [0.22, 0.32],          # nominal_h
])

PARAM_NOMINAL = np.array([
    0.06, 0.12, 2.0,
    0.0, np.pi, np.pi, 0.0,
    0.27,
])

N_PARAMS = len(PARAM_NAMES)


def vec_to_cpgparams(p) -> CPGParams:
    p = np.asarray(p, dtype=float)
    return CPGParams(
        H=float(p[0]),
        W=float(p[1]),
        omega=2 * np.pi * float(p[2]),
        psi_FR=float(p[3]),
        psi_FL=float(p[4]),
        psi_RR=float(p[5]),
        psi_RL=float(p[6]),
        nominal_height=float(p[7]),
    )


def clip_to_bounds(p):
    return np.clip(p, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])


def evaluate(param_vec, course_sections, duration=25.0,
             heading_feedback=True):
    """Одна оценка вектора параметров. Возвращает словарь метрик."""
    from .simulator import run_episode
    params = vec_to_cpgparams(clip_to_bounds(param_vec))
    _, metrics, _ = run_episode(
        params,
        course_sections=course_sections,
        use_viewer=False,
        max_duration=duration,
        realtime=False,
        heading_feedback=heading_feedback,
        verbose=False,
    )
    return metrics


def scalar_objective(metrics, weights=None):
    """Свёртка для CMA-ES. Возвращает значение для МИНИМИЗАЦИИ."""
    if metrics["J6_traverse"] == 0:
        return 1e6
    if weights is None:
        weights = {"w1": 10.0, "w2": 0.2, "w3": 0.0, "w4": 0.0, "w5": 0.0}
    return (
        - weights.get("w1", 0) * metrics["J1_speed"]
        + weights.get("w2", 0) * metrics["J2_CoT"]
        + weights.get("w3", 0) * metrics["J3_smooth"]
        + weights.get("w4", 0) * metrics["J4_roll"]
        + weights.get("w5", 0) * metrics["J5_pitch"]
    )
