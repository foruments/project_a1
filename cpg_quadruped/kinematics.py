"""
Обратная кинематика ноги Unitree A1, выведенная и сверенная с MuJoCo.

Конвенция углов (определена из калибровки на MJCF-модели):
    q_hip   — поворот вокруг оси X (тазобедренная абдукция).
              q_hip > 0 поворачивает ногу влево (для FR/RR — к центру тела).
    q_thigh — поворот бедра вокруг оси Y.
              q_thigh > 0 — нога идёт НАЗАД (положительный поворот от -Z к -X).
    q_calf  — поворот голени относительно бедра вокруг оси Y.
              q_calf < 0 — колено сгибается "назад" (стандарт для квадропеда).

Прямая кинематика (выведена и сверена с MuJoCo):
    x_loc = -L1*sin(q_thigh) - L2*sin(q_thigh + q_calf)
    y_loc = sgn * D_HIP                 (sgn = -1 для FR/RR, +1 для FL/RL)
    z_loc = -L1*cos(q_thigh) - L2*cos(q_thigh + q_calf)

    После поворота вокруг X на q_hip:
        x = x_loc
        y = y_loc*cos(q_hip) - z_loc*sin(q_hip)
        z = y_loc*sin(q_hip) + z_loc*cos(q_hip)

Здесь (x, y, z) — положение стопы в СК trunk относительно origin hip body.
"""
import numpy as np

L1 = 0.20        # длина бедра (m)
L2 = 0.20        # длина голени (m)
D_HIP = 0.0851   # вынос ноги по Y от origin hip body (из калибровки)

LEG_SIDE_SIGN = {"FR": -1, "FL": +1, "RR": -1, "RL": +1}


def leg_fk(q: np.ndarray, leg: str) -> np.ndarray:
    """Прямая кинематика — положение стопы в СК trunk относительно hip body."""
    q_hip, q_thigh, q_calf = q
    sgn = LEG_SIDE_SIGN[leg]
    x_loc = -L1 * np.sin(q_thigh) - L2 * np.sin(q_thigh + q_calf)
    y_loc = sgn * D_HIP
    z_loc = -L1 * np.cos(q_thigh) - L2 * np.cos(q_thigh + q_calf)
    x = x_loc
    y = y_loc * np.cos(q_hip) - z_loc * np.sin(q_hip)
    z = y_loc * np.sin(q_hip) + z_loc * np.cos(q_hip)
    return np.array([x, y, z])


def leg_ik(p_foot: np.ndarray, leg: str, knee_bent: str = "back") -> np.ndarray:
    """Аналитическая IK для одной ноги A1."""
    x, y, z = float(p_foot[0]), float(p_foot[1]), float(p_foot[2])
    sgn = LEG_SIDE_SIGN[leg]
    D = D_HIP

    s = y * y + z * z - D * D
    if s < 0:
        raise ValueError("Точка вне досягаемости по hip: y^2+z^2 < D^2")
    z_loc = -np.sqrt(s)

    q_hip = np.arctan2(z, y) - np.arctan2(z_loc, sgn * D)
    q_hip = (q_hip + np.pi) % (2 * np.pi) - np.pi

    u = -x
    v = -z_loc
    r2 = u * u + v * v
    r = np.sqrt(r2)
    if r > L1 + L2 + 1e-9:
        raise ValueError(f"Слишком далеко: r={r:.4f} > L1+L2={L1+L2:.4f}")
    if r < abs(L1 - L2) - 1e-9:
        raise ValueError(f"Слишком близко: r={r:.4f} < |L1-L2|={abs(L1-L2):.4f}")

    cos_calf = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_calf = np.clip(cos_calf, -1.0, 1.0)
    if knee_bent == "back":
        q_calf = -np.arccos(cos_calf)
    else:
        q_calf = +np.arccos(cos_calf)

    q_thigh = np.arctan2(u, v) - np.arctan2(
        L2 * np.sin(q_calf), L1 + L2 * np.cos(q_calf)
    )
    return np.array([q_hip, q_thigh, q_calf])


if __name__ == "__main__":
    print("=== Тест IK для всех ног Unitree A1 ===\n")
    test_cases = [
        ("FR", np.array([0.0,   0.0,   0.0])),
        ("FR", np.array([0.0,   0.5,   0.0])),
        ("FR", np.array([0.0,   0.0,  -0.5])),
        ("FR", np.array([0.3,   0.0,   0.0])),
        ("FR", np.array([0.0,   0.9,  -1.8])),
        ("FL", np.array([0.0,   0.9,  -1.8])),
        ("RR", np.array([0.1,   0.5,  -1.0])),
        ("RL", np.array([-0.1,  0.6,  -1.2])),
    ]
    n_ok = 0
    for leg, q_exp in test_cases:
        p_fk = leg_fk(q_exp, leg)
        try:
            q_ik = leg_ik(p_fk, leg)
            p_back = leg_fk(q_ik, leg)
            err = np.linalg.norm(p_back - p_fk)
            tag = "✓" if err < 1e-6 else "✗"
            if err < 1e-6: n_ok += 1
            print(f"{tag} {leg}: q_in={q_exp.round(3)}  p={p_fk.round(4)}")
            print(f"     q_ik={q_ik.round(3)}  err={err:.1e}")
        except ValueError as e:
            print(f"✗ {leg}: {e}")
    print(f"\nУспешно: {n_ok}/{len(test_cases)}")
