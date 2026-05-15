"""Пакет CPG-контроллера для квадропеда Unitree A1."""
from .kinematics import leg_fk, leg_ik, L1, L2, D_HIP, LEG_SIDE_SIGN
from .cpg import (
    CPGParams, GAIT_PRESETS, foot_trajectory,
    cpg_step, cpg_step_with_feedback, LEG_NAMES,
)
from .terrain import (
    load_model_for_terrain, build_course, CORRIDOR_HALF_WIDTH,
)
from .simulator import (
    run_episode, compute_metrics,
    JOINT_ORDER, CTRL_LEG_OFFSET, A1_XML,
)
from .objective import (
    PARAM_NAMES, PARAM_BOUNDS, PARAM_NOMINAL, N_PARAMS,
    vec_to_cpgparams, clip_to_bounds, evaluate, scalar_objective,
)

__all__ = [
    "leg_fk", "leg_ik", "L1", "L2", "D_HIP", "LEG_SIDE_SIGN",
    "CPGParams", "GAIT_PRESETS", "foot_trajectory",
    "cpg_step", "cpg_step_with_feedback", "LEG_NAMES",
    "load_model_for_terrain", "build_course", "CORRIDOR_HALF_WIDTH",
    "run_episode", "compute_metrics",
    "JOINT_ORDER", "CTRL_LEG_OFFSET", "A1_XML",
    "PARAM_NAMES", "PARAM_BOUNDS", "PARAM_NOMINAL", "N_PARAMS",
    "vec_to_cpgparams", "clip_to_bounds", "evaluate", "scalar_objective",
]
