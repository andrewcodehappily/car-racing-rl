# 🏎 CarRacing-v3 PPO

> 🇹🇼 [繁體中文](#繁體中文) ｜ 🇺🇸 [English](#english)

---

## 繁體中文

### 目錄
- [專案簡介](#專案簡介)
- [成果展示](#成果展示)
- [環境需求](#環境需求)
- [安裝步驟](#安裝步驟)
- [使用方式](#使用方式)
- [專案結構](#專案結構)
- [訓練原理](#訓練原理)

---

### 專案簡介

使用 **PPO（近端策略優化）** 強化學習算法，在 Gymnasium 的 `CarRacing-v3` 環境中訓練一個自動駕駛 AI。

- 訓練硬體：Apple Silicon Mac mini（MPS GPU）
- 最佳成績：**941+ 分**（滿分約 900~1000）
- 訓練時間：約 2~3 小時 / 百萬步

---

### 成果展示

AI 能夠流暢地跑完隨機生成的賽道，平均得分 700~900 分。

| 指標 | 數值 |
|------|------|
| 最佳 reward | 941+ |
| 平均 reward | ~700 |
| 訓練步數 | 1,000,000+ |
| 訓練裝置 | Apple Silicon MPS |

---

### 環境需求

- macOS（Apple Silicon）或 Linux
- Python **3.11**（不支援 3.12/3.13）
- 約 2GB 磁碟空間

---

### 安裝步驟

```bash
# 1. 安裝 Python 3.11（macOS）
brew install python@3.11

# 2. Clone 專案
git clone https://github.com/andrewcodehappily/car-racing-rl.git
cd car-racing-rl

# 3. 一鍵安裝環境
bash setup.sh

# 4. 啟動虛擬環境
source .venv/bin/activate
```

---

### 使用方式

#### 訓練 AI

```bash
# 從零開始訓練
python train.py

# 繼續訓練已有模型
python train.py --resume models/best/best_model --infinite

# 自訂步數與學習率
python train.py --resume models/best/best_model --steps 500000 --lr 3e-5
```

#### 觀看 AI 跑車

```bash
# 預設模型，附即時儀表板
python play.py

# 指定模型
python play.py --model timebest/best_model

# 不限步數（跑完整圈）
python play.py --model timebest/best_model --max-steps 0
```

#### 自己玩

```bash
# 鍵盤控制（↑油門 ↓煞車 ←→轉向）
python human_play.py
```

#### 人機對戰

```bash
# 跟 AI 比分數
python versus.py --model timebest/best_model

# 調整靈敏度
python versus.py --model timebest/best_model --sensitivity 0.3
```

#### 找出最強模型

```bash
python find_best.py
# 自動評估所有 session 的模型，儲存最強的到 timebest/
```

---

### 專案結構

```
car-racing-rl/
├── train.py          # 主訓練腳本（支援從零/繼續/無限模式）
├── play.py           # 觀看 AI 跑車 + 即時儀表板
├── human_play.py     # 鍵盤控制人類模式
├── versus.py         # 人機對戰模式
├── find_best.py      # 自動找出最強模型
├── setup.sh          # 一鍵安裝腳本
├── requirements.txt  # Python 套件需求
└── models/           # 訓練產生的模型（不含在 repo 中）
    ├── best/         # 全域最佳模型
    ├── session_XX/   # 每輪訓練的模型
    └── timebest/     # find_best.py 找出的最強模型
```

---

### 訓練原理

使用 **PPO + CnnPolicy**，輸入為連續 4 幀的 96×96 畫面，輸出連續動作（方向盤、油門、煞車）。

- **Actor**：決定要做什麼動作
- **Critic**：評估當前狀態的價值，引導 Actor 改進
- **Reward Shaping**：離軌懲罰、急彎獎勵、完賽大獎

兩個網路共享同一個 CNN 特徵提取層，只在最後幾層分叉，總參數量約 210 萬。

---

---

## English

### Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)

---

### Overview

A **PPO (Proximal Policy Optimization)** reinforcement learning agent trained to drive autonomously in Gymnasium's `CarRacing-v3` environment.

- Hardware: Apple Silicon Mac mini (MPS GPU)
- Best score: **941+** (max ~900–1000)
- Training time: ~2–3 hours per million steps

---

### Results

The AI drives smoothly on randomly generated tracks, averaging 700–900 points per episode.

| Metric | Value |
|--------|-------|
| Best reward | 941+ |
| Mean reward | ~700 |
| Training steps | 1,000,000+ |
| Device | Apple Silicon MPS |

---

### Requirements

- macOS (Apple Silicon) or Linux
- Python **3.11** (3.12/3.13 not supported)
- ~2GB disk space

---

### Installation

```bash
# 1. Install Python 3.11 (macOS)
brew install python@3.11

# 2. Clone the repo
git clone https://github.com/andrewcodehappily/car-racing-rl.git
cd car-racing-rl

# 3. One-command setup
bash setup.sh

# 4. Activate virtual environment
source .venv/bin/activate
```

---

### Usage

#### Train the AI

```bash
# Train from scratch
python train.py

# Continue training from saved model
python train.py --resume models/best/best_model --infinite

# Custom steps and learning rate
python train.py --resume models/best/best_model --steps 500000 --lr 3e-5
```

#### Watch the AI drive

```bash
# Default model with live dashboard
python play.py

# Specify model
python play.py --model timebest/best_model

# No step limit (run full lap)
python play.py --model timebest/best_model --max-steps 0
```

#### Play yourself

```bash
# Keyboard control (↑ gas  ↓ brake  ← → steer)
python human_play.py
```

#### Human vs AI

```bash
# Compete against the AI
python versus.py --model timebest/best_model

# Adjust steering sensitivity
python versus.py --model timebest/best_model --sensitivity 0.3
```

#### Find the best model

```bash
python find_best.py
# Evaluates all session models and saves the best to timebest/
```

---

### Project Structure

```
car-racing-rl/
├── train.py          # Main training script (scratch / resume / infinite)
├── play.py           # Watch AI drive + live dashboard
├── human_play.py     # Human keyboard control mode
├── versus.py         # Human vs AI mode
├── find_best.py      # Auto-find best model across all sessions
├── setup.sh          # One-command environment setup
├── requirements.txt  # Python dependencies
└── models/           # Trained models (not included in repo)
    ├── best/         # Global best model
    ├── session_XX/   # Per-session checkpoints
    └── timebest/     # Best model found by find_best.py
```

---

### How It Works

Uses **PPO + CnnPolicy** with 4 stacked 96×96 frames as input, outputting continuous actions (steering, gas, brake).

- **Actor**: Decides what action to take
- **Critic**: Evaluates state value to guide the Actor
- **Reward Shaping**: Off-track penalties, sharp-turn bonuses, lap completion reward

Both networks share a CNN feature extractor and only branch at the final layers, totaling ~2.1M parameters.