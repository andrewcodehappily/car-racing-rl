"""
人類單人模式 + 即時儀表板
兩個視窗：左邊遊戲，右邊顯示所有數值

控制：
  ← →   轉向
  ↑     油門
  ↓     煞車
  Q     放棄這局
  R     重新開始
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import gymnasium as gym
import pygame
import threading
import matplotlib
matplotlib.use("MacOSX")  # macOS 原生後端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ╔══════════════════════════════════════════════════════╗
# ║              全部可調參數都在這裡                      ║
# ╠══════════════════════════════════════════════════════╣

EPISODES        = 5
MAX_STEPS       = 2000
MIN_SCORE       = -300
FPS             = 60

STEER_MAX       = 0.6
STEER_ACCEL     = 0.04
STEER_RETURN    = 0.18

GAS_STRENGTH    = 0.7
BRAKE_STRENGTH  = 0.6

DASHBOARD       = True    # 開啟儀表板視窗
HISTORY_LEN     = 200     # 儀表板歷史曲線長度（幾步）

# ╚══════════════════════════════════════════════════════╝


# ── 按鍵狀態 ─────────────────────────────────────────
class Keys:
    left = right = gas = brake = quit = restart = False

keys   = Keys()
_steer = 0.0


def get_action() -> np.ndarray:
    global _steer
    if keys.left:
        _steer = max(_steer - STEER_ACCEL, -STEER_MAX)
    elif keys.right:
        _steer = min(_steer + STEER_ACCEL,  STEER_MAX)
    else:
        _steer = (_steer - STEER_RETURN) if _steer > 0 else (
                  _steer + STEER_RETURN  if _steer < 0 else 0.0)
        _steer = max(-STEER_MAX, min(STEER_MAX, _steer))
        if abs(_steer) < STEER_RETURN:
            _steer = 0.0

    gas   = GAS_STRENGTH   if keys.gas   else 0.0
    brake = BRAKE_STRENGTH if keys.brake else 0.0
    if gas > 0.1 and brake > 0.1:
        if gas >= brake: brake = 0.0
        else:            gas   = 0.0

    return np.array([_steer, gas, brake], dtype=np.float32)


def reset_keys():
    global _steer
    keys.left = keys.right = keys.gas = keys.brake = False
    keys.quit = keys.restart = False
    _steer = 0.0


def process_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            keys.quit = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:  keys.left    = True
            if event.key == pygame.K_RIGHT: keys.right   = True
            if event.key == pygame.K_UP:    keys.gas     = True
            if event.key == pygame.K_DOWN:  keys.brake   = True
            if event.key == pygame.K_q:     keys.quit    = True
            if event.key == pygame.K_r:     keys.restart = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:  keys.left    = False
            if event.key == pygame.K_RIGHT: keys.right   = False
            if event.key == pygame.K_UP:    keys.gas     = False
            if event.key == pygame.K_DOWN:  keys.brake   = False


# ── 儀表板 ────────────────────────────────────────────
class Dashboard:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(7, 9), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("儀表板")
        gs = GridSpec(5, 2, figure=self.fig,
                      hspace=0.6, wspace=0.4,
                      left=0.12, right=0.95, top=0.93, bottom=0.06)

        clr = "#1a1a2e"
        self.axes = {
            "speed":    self.fig.add_subplot(gs[0, :]),
            "steer":    self.fig.add_subplot(gs[1, :]),
            "gas":      self.fig.add_subplot(gs[2, 0]),
            "brake":    self.fig.add_subplot(gs[2, 1]),
            "gyro":     self.fig.add_subplot(gs[3, :]),
            "reward":   self.fig.add_subplot(gs[4, :]),
        }

        titles = {
            "speed":  "Speed (white)",
            "steer":  "Steering (green)  -L / +R",
            "gas":    "Gas",
            "brake":  "Brake",
            "gyro":   "Gyro · rotation speed (red)",
            "reward": "Score",
        }
        colors = {
            "speed": "#e0e0e0", "steer": "#4caf50",
            "gas":   "#ffb300", "brake": "#ef5350",
            "gyro":  "#ef5350", "reward":"#42a5f5",
        }

        self.lines  = {}
        self.data   = {k: [] for k in titles}
        self.colors = colors

        for k, ax in self.axes.items():
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="#aaaaaa", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")
            ax.set_title(titles[k], color="#cccccc", fontsize=8, pad=3)
            self.lines[k], = ax.plot([], [], color=colors[k], lw=1.2)
            ax.set_xlim(0, HISTORY_LEN)

        # 固定 y 軸範圍
        self.axes["speed"].set_ylim(0, 120)
        self.axes["steer"].set_ylim(-1, 1)
        self.axes["gas"].set_ylim(0, 1)
        self.axes["brake"].set_ylim(0, 1)
        self.axes["gyro"].set_ylim(-300, 300)

        # 大數字顯示區
        self.texts = {}
        for k in ["speed", "steer", "gyro", "reward"]:
            self.texts[k] = self.axes[k].text(
                0.98, 0.75, "0", transform=self.axes[k].transAxes,
                color=colors[k], fontsize=13, fontweight="bold",
                ha="right", va="top"
            )

        # ABS 四輪指示燈
        self.abs_ax = self.fig.add_axes([0.1, 0.008, 0.8, 0.04],
                                         facecolor="#0d1117")
        self.abs_ax.set_xlim(0, 4)
        self.abs_ax.set_ylim(0, 1)
        self.abs_ax.axis("off")
        self.abs_ax.set_title("ABS grip (blue=grip  red=slip)",
                               color="#cccccc", fontsize=8, pad=1)
        self.abs_rects = []
        labels = ["FL", "FR", "RL", "RR"]
        for i in range(4):
            rect = mpatches.FancyBboxPatch(
                (i + 0.1, 0.1), 0.8, 0.8,
                boxstyle="round,pad=0.05",
                facecolor="#1565c0", edgecolor="#333", lw=0.5
            )
            self.abs_ax.add_patch(rect)
            self.abs_ax.text(i + 0.5, 0.5, labels[i],
                              ha="center", va="center",
                              color="white", fontsize=7)
            self.abs_rects.append(rect)

        self.abs_vals = [1.0, 1.0, 1.0, 1.0]
        plt.draw()
        plt.pause(0.001)

    def update(self, info: dict, action: np.ndarray, total_reward: float):
        """info 是 gymnasium step 回傳的 info dict"""
        # 從環境拿數值
        speed = info.get("speed", 0.0) * 100      # 歸一化 → km/h 近似
        steer = float(action[0])
        gas   = float(action[1])
        brake = float(action[2])
        gyro  = info.get("gyro", 0.0) * 1000

        # ABS 四輪
        abs_vals = info.get("ABS", [1.0, 1.0, 1.0, 1.0])
        if hasattr(abs_vals, '__len__') and len(abs_vals) == 4:
            self.abs_vals = [float(v) for v in abs_vals]

        vals = {
            "speed":  speed,
            "steer":  steer,
            "gas":    gas,
            "brake":  brake,
            "gyro":   gyro,
            "reward": total_reward,
        }

        for k, v in vals.items():
            self.data[k].append(v)
            if len(self.data[k]) > HISTORY_LEN:
                self.data[k].pop(0)
            x = list(range(len(self.data[k])))
            self.lines[k].set_data(x, self.data[k])

        # 更新 reward 的 y 軸動態縮放
        if self.data["reward"]:
            mn = min(self.data["reward"])
            mx = max(self.data["reward"])
            pad = max(abs(mx - mn) * 0.2, 10)
            self.axes["reward"].set_ylim(mn - pad, mx + pad)

        # 更新大數字
        self.texts["speed"].set_text(f"{speed:.0f}")
        self.texts["steer"].set_text(f"{steer:+.2f}")
        self.texts["gyro"].set_text(f"{gyro:+.0f}")
        self.texts["reward"].set_text(f"{total_reward:.0f}")

        # 更新 ABS 顏色（越低越紅，代表打滑）
        for i, (rect, v) in enumerate(zip(self.abs_rects, self.abs_vals)):
            r = int(255 * (1 - v))
            b = int(255 * v)
            rect.set_facecolor((r/255, 0, b/255))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def reset(self):
        for k in self.data:
            self.data[k].clear()

    def close(self):
        plt.close(self.fig)


# ── 主遊戲迴圈 ────────────────────────────────────────
def play_episode(env, ep_num: int, dash) -> dict:
    global _steer
    _steer = 0.0
    obs, info = env.reset()
    total  = 0.0
    steps  = 0
    clock  = pygame.time.Clock()
    limit  = MAX_STEPS if MAX_STEPS > 0 else 999999

    if dash:
        dash.reset()

    print(f"\n  ── 第 {ep_num} 局  (Q放棄 / R重來) ──")

    while steps < limit:
        process_events()

        if keys.quit:
            print("  放棄")
            break
        if keys.restart:
            keys.restart = False
            obs, info = env.reset()
            total = 0.0
            steps = 0
            _steer = 0.0
            if dash: dash.reset()
            continue

        action = get_action()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1

        if dash and steps % 3 == 0:   # 每 3 步更新一次儀表板
            dash.update(info, action, total)

        if total < MIN_SCORE:
            print(f"\n  低於 {MIN_SCORE} 分，結束")
            break
        if terminated or truncated:
            break

        clock.tick(FPS)

    print(f"  結果：{total:.0f} 分  {steps} 步")
    return {"reward": total, "steps": steps}


def main():
    pygame.init()

    limit = MAX_STEPS if MAX_STEPS > 0 else 999999
    env   = gym.make("CarRacing-v3", continuous=True,
                     render_mode="human",
                     max_episode_steps=limit)

    dash = Dashboard() if DASHBOARD else None

    print("\n" + "═" * 50)
    print("  🏎  CarRacing 人類模式 + 儀表板")
    print("─" * 50)
    print(f"  局數 {EPISODES}  |  步數上限 {'無限' if MAX_STEPS==0 else MAX_STEPS}")
    print(f"  轉向靈敏度 {STEER_ACCEL}  |  最大轉向 {STEER_MAX}")
    print("  ← → 轉向   ↑ 油門   ↓ 煞車   Q 放棄   R 重來")
    print("═" * 50)

    all_rewards = []

    for ep in range(1, EPISODES + 1):
        result = play_episode(env, ep, dash)
        all_rewards.append(result["reward"])
        reset_keys()
        if keys.quit:
            break
        if ep < EPISODES:
            input("  按 Enter 繼續...")

    env.close()
    if dash:
        dash.close()

    print("\n" + "═" * 50)
    print(f"  平均 {np.mean(all_rewards):.0f} 分  最高 {max(all_rewards):.0f} 分")
    print("═" * 50 + "\n")

    pygame.quit()


if __name__ == "__main__":
    main()