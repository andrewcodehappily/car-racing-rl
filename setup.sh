#!/usr/bin/env bash
# ────────────────────────────────────────────────────────
#  CarRacing-v3 RL 專案環境建置腳本
#  用法：bash scripts/setup.sh
# ────────────────────────────────────────────────────────
set -e

# 不管 setup.sh 放在專案根目錄還是 scripts/ 子目錄，都能正確找到專案根
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    PROJECT_DIR="$SCRIPT_DIR"           # setup.sh 在根目錄
else
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"  # setup.sh 在 scripts/ 子目錄
fi
VENV_DIR="$PROJECT_DIR/.venv"

echo ""
echo "================================================"
echo "  CarRacing RL 環境建置"
echo "  專案目錄：$PROJECT_DIR"
echo "================================================"
echo ""

# ── 1. 確認 Python 版本 ──────────────────────────────────
PYTHON=$(command -v python3.11 || command -v python3.10 || command -v python3)
PY_VERSION=$($PYTHON --version 2>&1)
echo "使用 Python：$PY_VERSION  ($PYTHON)"

# ── 2. 建立 .venv ────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo ""
    echo ".venv 已存在，跳過建立。"
    echo "如需重建，請先刪除：rm -rf $VENV_DIR"
else
    echo ""
    echo "建立虛擬環境 .venv ..."
    $PYTHON -m venv "$VENV_DIR"
    echo ".venv 建立完成。"
fi

# ── 3. 啟動 venv ─────────────────────────────────────────
source "$VENV_DIR/bin/activate"
echo "已啟動 .venv"

# ── 4. 升級 pip ──────────────────────────────────────────
echo ""
echo "升級 pip ..."
pip install --upgrade pip --quiet

# ── 5. 安裝依賴 ──────────────────────────────────────────
echo ""
echo "安裝依賴套件（requirements.txt）..."
pip install -r "$PROJECT_DIR/requirements.txt"

# ── 6. 建立資料夾結構 ────────────────────────────────────
mkdir -p "$PROJECT_DIR/models/best"
mkdir -p "$PROJECT_DIR/logs"

# ── 7. 完成提示 ──────────────────────────────────────────
echo ""
echo "================================================"
echo "  安裝完成！"
echo ""
echo "  啟動環境："
echo "    source .venv/bin/activate"
echo ""
echo "  開始訓練："
echo "    python src/train.py"
echo ""
echo "  觀看訓練結果："
echo "    python src/play.py --model models/best/best_model"
echo ""
echo "  開啟 TensorBoard："
echo "    tensorboard --logdir logs"
echo "================================================"
echo ""