# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

CarRacing-v3 PPO 強化學習專案，使用 Stable-Baselines3 在 Gymnasium 的 CarRacing-v3 環境中訓練自動駕駛 AI。輸入為連續 4 幀 96x96 畫面，輸出連續動作（方向盤、油門、煞車），總參數約 210 萬。

## Python 環境

- **Python 3.11** required（3.12/3.13 不支援）
- 虛擬環境在 `.venv/`
- 安裝：`bash setup.sh` 或 `pip install -r requirements.txt`
- 啟動：`source .venv/bin/activate`

## 常用指令

```bash
# 訓練（從零開始）
python train.py

# 繼續訓練
python train.py --resume models/best/best_model --infinite

# 繼續訓練 + 自訂步數/學習率
python train.py --resume models/best/best_model --steps 500000 --lr 3e-5

# 凍結 CNN（只訓練 Actor/Critic 頭，速度快 3~4x）
python train.py --resume models/best/best_model --freeze-cnn

# 觀看 AI 跑車
python play.py

# 指定模型觀看
python play.py --model timebest/best_model

# 人類模式
python human_play.py

# 人機對戰
python versus.py --model timebest/best_model

# 自動找出最強模型（評估所有 session）
python find_best.py

# 啟動 TensorBoard
tensorboard --logdir logs
```

## 程式碼架構

### 核心檔案

| 檔案 | 功能 |
|------|------|
| `train.py` | 主訓練腳本，支援從零 / 繼續 / 無限模式 |
| `play.py` | 觀看 AI 跑車 + 即時儀表板（matplotlib MacOSX 後端） |
| `human_play.py` | 人類鍵盤控制模式 + 儀表板 |
| `versus.py` | 人機對戰（AI 背景執行緒 + 人類 pygame） |
| `find_best.py` | 自動評估所有 session 模型，選最強複製到 `timebest/` |
| `setup.sh` | 一鍵建立 .venv + 安裝依賴 + 建立資料夾 |

### 目錄結構

```
models/
├── best/            # 全域最佳模型（EvalCallback 自動儲存）
├── session_XX/      # 每輪訓練的 checkpoint + final
├── interrupted/     # Ctrl+C 緊急儲存
└── emergency/       # 未知錯誤緊急儲存
logs/                # TensorBoard 日誌 + Monitor CSV
timebest/            # find_best.py 找出的最強模型
```

### train.py 關鍵元件

- **RewardShapingWrapper**（第 151 行）：環境 wrapper，實作離軌漸進懲罰、急彎獎勵、平滑駕駛獎懲、打圈偵測、完賽大獎。單步最大正 reward ~+0.52，最大負向 ~-0.15，完賽大獎 50.0。
- **ProgressCallback**（第 267 行）：每 VIZ_FREQ 步印進度條、平均 reward、FPS、ETA。
- **StabilizeCallback**（第 326 行）：每 2048 步監控 action std，std 過高時自動降低 ent_coef / lr / clip，std 過低時提高 ent_coef 防過擬合。
- **LRHolder**（第 92 行）：可變 LR 容器，解決 SB3 lambda 閉包問題，支援訓練中動態調整學習率。
- **adaptive_lr_decay**（第 113 行）：無限模式的自適應衰減，越接近 LR_MIN 衰減越小。
- 使用 4 個並行環境（DummyVecEnv）+ VecFrameStack(4) + VecTransposeImage。

### play.py / human_play.py 儀表板

- matplotlib MacOSX 後端，GridSpec 5x2 佈局
- 即時曲線：速度、方向盤、油門、煞車、陀螺儀、累積 reward
- 大數字顯示 + ABS 四輪抓地力指示燈

### 動作後處理（所有檔案共用邏輯）

1. 方向盤 clip 到 [-1, 1]，油門/煞車 clip 到 [0, 1]
2. 油門 > 0.1 且 煞車 > 0.1 時互斥（只保留較大者）

## 訓練參數

| 參數 | 值 | 說明 |
|------|-----|------|
| N_ENVS | 4 | 並行環境數 |
| N_STACK | 4 | 堆疊幀數 |
| TOTAL_STEPS | 1,000,000 | 每輪步數 |
| EVAL_FREQ | 20,000 | 評估頻率 |
| SAVE_FREQ | 50,000 | checkpoint 頻率 |
| n_steps | 512 | 每步收集長度 |
| batch_size | 128 | 批次大小 |
| n_epochs | 10 | 每批次更新次數 |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE 參數 |
| ent_coef | 0.01(初始)/0.0005(resume) | 熵正則係數 |

## 注意事項

- play.py 需要顯示器（render_mode="human"），SSH 環境需 X11 forward
- MPS 訓練不穩定時自動 fallback 到 CPU
- 模型副檔名為 `.zip`，路徑傳入時可省略
