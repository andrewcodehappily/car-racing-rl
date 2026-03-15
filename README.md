# 🏎 CarRacing-v3 PPO

A reinforcement learning agent trained to drive in [Gymnasium](https://gymnasium.farama.org/)'s CarRacing-v3 environment using Proximal Policy Optimization (PPO).

Trained on Apple Silicon (MPS) · Best reward: **941+**

---

## Demo

The AI learns to drive a randomly generated track from scratch using only raw pixels as input.

---

## Project Structure

```
carai/
├── train.py          # Train from scratch or resume
├── play.py           # Watch AI drive + live dashboard
├── human_play.py     # Play yourself with live dashboard
├── versus.py         # Human vs AI head-to-head
├── find_best.py      # Evaluate all saved models, pick the best
├── timebest/         # Best model saved here
├── models/           # All checkpoints
└── logs/             # TensorBoard logs
```

---

## Setup

Requires **Python 3.11** (not 3.12/3.13 — causes crashes with gymnasium).

```bash
# Install Python 3.11 via Homebrew (macOS)
brew install python@3.11

# Create virtual environment
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install "gymnasium[box2d]>=0.29.1" \
            "stable-baselines3[extra]>=2.3.2" \
            tensorboard>=2.17.0 \
            opencv-python>=4.10.0 \
            numpy>=2.0.0 \
            torch>=2.6.0 \
            matplotlib>=3.9.0
```

---

## Usage

### Train

```bash
# From scratch
python train.py

# Resume from best model (infinite mode with adaptive lr decay)
python train.py --resume timebest/best_model --infinite

# Custom learning rate and steps per session
python train.py --resume timebest/best_model --lr 3e-5 --steps 2000000
```

### Watch AI play

```bash
python play.py --model timebest/best_model
```

Opens two windows: the game + a live dashboard showing speed, steering, gas/brake, gyro, score, and ABS grip per wheel.

### Play yourself

```bash
python human_play.py
```

Controls: `← →` steer · `↑` gas · `↓` brake · `R` restart · `Q` quit

All sensitivity settings are at the top of the file.

### Human vs AI

```bash
python versus.py --model timebest/best_model --rounds 3
```

### Find best model across all sessions

```bash
python find_best.py
# Saves the winner to timebest/best_model.zip
```

---

## Architecture

| Component | Detail |
|-----------|--------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Policy | CnnPolicy (shared CNN backbone) |
| Input | 4 stacked frames · 12 × 96 × 96 |
| CNN | 3 conv layers (32/64/64 filters) |
| Output | 3 continuous actions: steering, gas, brake |
| Device | Apple MPS (Apple Silicon GPU) |
| Parameters | ~2.1M total |

### Reward Shaping

- Off-track progressive penalty (longer = heavier)
- Early termination after 50 consecutive off-track steps
- Smooth steering bonus / jerk penalty
- Sharp turn success bonus
- Speed reward
- Finish bonus (+50)
- Spin detection penalty

---

## Training Progress

| Steps | ep_rew_mean | explained_variance |
|-------|-------------|-------------------|
| 4k    | -112        | -0.01             |
| 50k   | -77         | 0.78              |
| 500k  | 500+        | 0.85              |
| 1M+   | 700–940     | 0.93+             |

---

## TensorBoard

```bash
tensorboard --logdir logs
```

---

## Requirements

```
gymnasium[box2d]>=0.29.1
stable-baselines3[extra]>=2.3.2
tensorboard>=2.17.0
opencv-python>=4.10.0
numpy>=2.0.0
torch>=2.6.0
matplotlib>=3.9.0
```

---

## Notes

- `models/` and `timebest/` are excluded from git (too large). Only source code is committed.
- SDL duplicate warning on macOS (`objc: Class SDLApplication...`) is harmless — ignore it.
- Always activate `.venv` before running: `source .venv/bin/activate`