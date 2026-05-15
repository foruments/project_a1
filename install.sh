#!/usr/bin/env bash
# Быстрая установка проекта.
# Использование: bash install.sh
set -e

echo "▶ Установка зависимостей Python..."
pip install -r requirements.txt

echo ""
echo "▶ Скачиваю модель Unitree A1 из MuJoCo Menagerie..."
if [ ! -d "mujoco_menagerie" ]; then
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/google-deepmind/mujoco_menagerie.git
    cd mujoco_menagerie && git sparse-checkout set unitree_a1 && cd ..
    echo "  ✓ Модель скачана в ./mujoco_menagerie/unitree_a1/"
else
    echo "  ℹ Папка mujoco_menagerie уже существует, пропускаю"
fi

echo ""
echo "▶ Проверка что модель найдена..."
python3 -c "
from cpg_quadruped.simulator import A1_XML
from cpg_quadruped.terrain import A1_DIR
print('  ✓ A1_XML:', A1_XML)
print('  ✓ A1_DIR:', A1_DIR)
"

echo ""
echo "══════════════════════════════════════════════════"
echo "  Готово! Запусти: python run.py"
echo "══════════════════════════════════════════════════"
