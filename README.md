# CPG-Gait Optimization for Unitree A1

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-orange.svg)](https://mujoco.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Параметрическая оптимизация походки четвероногого робота Unitree A1 в среде MuJoCo** с использованием CPG-контроллера и трёх методов оптимизации: brute-force, CMA-ES, NSGA-II.

Проект по дисциплине «Разработка и оптимизация мехатронных систем», Университет ИТМО, факультет систем управления и робототехники.

> Базовая работа: Owaki D., Kano T., Ishiguro A. *Foot trajectory as a key factor for diverse gait patterns in quadruped robot locomotion* // Scientific Reports, 2025. DOI: [10.1038/s41598-024-84060-5](https://doi.org/10.1038/s41598-024-84060-5)

---

## 📋 Содержание

- [Что внутри](#-что-внутри)
- [Быстрый старт](#-быстрый-старт)
- [Использование](#-использование)
- [Архитектура](#-архитектура)
- [Пространство оптимизации](#-пространство-оптимизации)
- [Метрики](#-метрики)
- [Результаты](#-результаты)
- [Полезные ссылки](#-полезные-ссылки)
- [Цитирование](#-цитирование)
- [Лицензия](#-лицензия)

---

## 🎯 Что внутри

- **CPG-контроллер** на основе фазовых осцилляторов с обратной связью по курсу
- **Аналитическая обратная кинематика** ноги, сверенная с MuJoCo до 10⁻⁶ м
- **Полоса препятствий** из 4 типов секций (flat, rough, slope, stairs), окружённая стенками для предотвращения обхода
- **Шесть метрик качества** J₁..J₆ (скорость, CoT, плавность, стабильность, проходимость)
- **Три алгоритма оптимизации**: brute-force, CMA-ES (`pycma`), NSGA-II (`pymoo`)
- **Интерактивное меню** (`questionary`)

---

## 🚀 Быстрый старт

### Linux / macOS

```bash
git clone https://github.com/foruments/project_a1.git
cd project_a1
bash install.sh
python run.py
```

### Windows

```powershell
git clone https://github.com/foruments/project_a1.git
cd project_a1
pip install -r requirements.txt

# Скачиваем модель Unitree A1
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie
git sparse-checkout set unitree_a1
cd ..

# В файлах cpg_quadruped/simulator.py и cpg_quadruped/terrain.py
# поправь пути A1_XML и A1_DIR на полный путь к unitree_a1

python run.py
```

После запуска откроется интерактивное меню:

```
   ПАРАМЕТРИЧЕСКАЯ ОПТИМИЗАЦИЯ ПОХОДКИ UNITREE A1

   ? Что делаем?
   ❯ 🎬 Запустить симуляцию (MuJoCo viewer + графики)
     📊 Brute-force оптимизация по сетке (H, W)
     🧬 CMA-ES оптимизация (8 параметров)
     🎯 NSGA-II многокритериальная оптимизация
     📈 Сравнить результаты всех методов
     🏆 Запустить с найденными оптимальными параметрами
     ❌ Выход
```

---

## Использование

### Интерактивный режим

Просто запусти `python run.py` и следуй меню.

### CLI (для скриптов)

```bash
# Прохождение полосы с просмотром
python run.py --action view --course flat rough --H 0.10 --W 0.18

# Brute-force оптимизация
python run.py --action bruteforce --course flat rough --n 6

# CMA-ES в 8-мерном пространстве
python run.py --action cma --course flat rough --budget 60 --popsize 10

# NSGA-II многокритериальный
python run.py --action nsga --course flat rough --pop 12 --gen 8

# Сравнение всех методов
python run.py --action compare --course flat rough

# Запуск с найденным оптимумом
python run.py --action run_best
```

Полная справка: `python run.py --help`

### Управление в окне MuJoCo

| Действие | Клавиша/кнопка |
|---|---|
| Вращать камеру | Правая кнопка мыши |
| Zoom | Прокрутка колесом |
| Применить силу к роботу | Левая кнопка мыши (тянуть) |
| Пауза/возобновить | Space |
| Свободная/автокамера | Tab |
| Выход | Esc или закрыть окно |

---

## Архитектура

```
project_a1/
├── run.py                       ← ЕДИНАЯ ТОЧКА ВХОДА (меню + CLI)
├── cpg_quadruped/               ← Пакет с движком
│   ├── kinematics.py            # IK/FK ноги A1 (откалибровано на MuJoCo)
│   ├── cpg.py                   # CPG-осцилляторы + heading feedback
│   ├── terrain.py               # Сцены MuJoCo (коридор + стенки)
│   ├── simulator.py             # run_episode + compute_metrics
│   └── objective.py             # Функция оценки для оптимизаторов
├── results/                     # Графики и сохранённые данные (gitignored)
├── install.sh                   # Скрипт автоустановки
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

### Трёхуровневая структура контроллера

```
┌──────────────────────────────────────┐
│   CPG: фазовые осцилляторы           │   φᵢ(t) = ωt + ψᵢ
└──────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│   Генератор траектории стопы         │   p_foot(t) = f(φ, H, W)
└──────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│   IK + PD на приводах                │   τᵢ(t)
└──────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│   Unitree A1 в MuJoCo                │
└──────────────────────────────────────┘
              │
              └─── обратная связь по yaw, y ──┐
                                              ▼
                                       коррекция Wₗ/Wᵣ
```

---

## Пространство оптимизации

8 непрерывных параметров:

| # | Параметр | Диапазон | Описание |
|---|---|---|---|
| 1 | `H` | 0.02–0.12 м | высота подъёма стопы |
| 2 | `W` | 0.06–0.20 м | ширина шага |
| 3 | `freq` | 1.0–3.5 Hz | частота CPG |
| 4 | `ψ_FR` | 0–2π | фаза правой передней |
| 5 | `ψ_FL` | 0–2π | фаза левой передней |
| 6 | `ψ_RR` | 0–2π | фаза правой задней |
| 7 | `ψ_RL` | 0–2π | фаза левой задней |
| 8 | `nominal_h` | 0.22–0.32 м | высота trunk |

---

## Метрики

| Метрика | Формула | Цель |
|---|---|---|
| J₁ — скорость | (1/T) ∫₀ᵀ vₓ(t) dt | **max** |
| J₂ — стоимость перемещения (CoT) | ∫₀ᵀ Σᵢ\|τᵢωᵢ\| dt / (m·g·d) | **min** |
| J₃ — плавность | √((1/T) ∫₀ᵀ z̈² dt) | **min** |
| J₄ — макс. крен | maxₜ \|φ(t)\| | **min** |
| J₅ — макс. дифферент | maxₜ \|θ(t)\| | **min** |
| J₆ — проходимость | не упал И не вышел из коридора | **=1** |

---

## Результаты

Лучшие результаты на полосе `flat → rough`:

| Метод | Оценок | Лучший J₁ | Размерность |
|---|---|---|---|
| Brute-force | 25 | 0.354 м/с | 2 (H, W) |
| CMA-ES | 20 | 0.402 м/с | 8 |
| **NSGA-II** | 32 | **0.548 м/с** | 8 |

**Ключевой вывод:** в полном 8-мерном пространстве NSGA-II находит конфигурацию в **3.4 раза быстрее** номинальной trot-походки. Парето-фронт «скорость↔энергия» управляется в первую очередь **частотой CPG**, а не геометрией стопы.

---

## Особенности реализации

### Защита от обхода препятствий
Все препятствия покрывают полную ширину коридора 2 м. По краям стоят полупрозрачные стенки. J₆ = 1 только если робот реально прошёл через препятствия, а не обошёл их сбоку.

### Обратная связь по курсу
P-регулятор по yaw и lateral position снижает дрейф с 14% до 1.7%. Эквивалент IMU-обратной связи на реальных квадропедах (Spot, ANYmal).

### Калибровка кинематики
Конвенция углов Unitree A1 неинтуитивна (положительный `q_thigh` поворачивает ногу назад). Кинематика откалибрована эмпирически на MJCF-модели, ошибка IK/FK < 10⁻⁶ м.

---

## Полезные ссылки

- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — модель Unitree A1
- [pycma](https://github.com/CMA-ES/pycma) — реализация CMA-ES
- [pymoo](https://pymoo.org/) — NSGA-II и другие многокритериальные алгоритмы
- [Owaki et al. 2025](https://doi.org/10.1038/s41598-024-84060-5) — базовая статья

---

## Цитирование

Если этот проект помог в исследованиях:

```bibtex
@software{vzglyadov2026cpg,
  author = {Vzglyadov, Z.E.},
  title = {CPG-Based Gait Optimization for Unitree A1 Quadruped Robot in MuJoCo},
  year = {2026},
  url = {https://github.com/foruments/project_a1},
  note = {ITMO University course project}
}
```

См. также [`CITATION.cff`](CITATION.cff).

---

## Лицензия

[MIT](LICENSE) — свободно используй, модифицируй, распространяй.

---

<div align="center">

Made with at [ITMO University](https://itmo.ru/) · Faculty of Control Systems and Robotics

</div>
