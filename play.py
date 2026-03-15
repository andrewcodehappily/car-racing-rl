"""
用訓練好的模型跑遊戲 + 即時儀表板
用法：
  python play.py                              # 預設 5 局，無步數限制
  python play.py --model timebest/best_model  # 指定模型
  python play.py --episodes 10               # 跑 10 局
  python play.py --max-steps 2000            # 每局最多 2000 步
  python play.py --no-dashboard              # 關閉儀表板
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage

HISTORY_LEN = 300


# ── 儀表板 ────────────────────────────────────────────
class Dashboard:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(7, 9), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title("AI Dashboard")
        gs = GridSpec(5, 2, figure=self.fig,
                      hspace=0.6, wspace=0.4,
                      left=0.12, right=0.95, top=0.93, bottom=0.06)

        self.axes = {
            "speed":  self.fig.add_subplot(gs[0, :]),
            "steer":  self.fig.add_subplot(gs[1, :]),
            "gas":    self.fig.add_subplot(gs[2, 0]),
            "brake":  self.fig.add_subplot(gs[2, 1]),
            "gyro":   self.fig.add_subplot(gs[3, :]),
            "reward": self.fig.add_subplot(gs[4, :]),
        }

        titles = {
            "speed":  "Speed (white)",
            "steer":  "Steering (green)  -L / +R",
            "gas":    "Gas",
            "brake":  "Brake",
            "gyro":   "Gyro · rotation (red)",
            "reward": "Score",
        }
        colors = {
            "speed": "#e0e0e0", "steer": "#4caf50",
            "gas":   "#ffb300", "brake": "#ef5350",
            "gyro":  "#ef5350", "reward":"#42a5f5",
        }

        self.lines = {}
        self.data  = {k: [] for k in titles}

        for k, ax in self.axes.items():
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="#aaaaaa", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")
            ax.set_title(titles[k], color="#cccccc", fontsize=8, pad=3)
            self.lines[k], = ax.plot([], [], color=colors[k], lw=1.2)
            ax.set_xlim(0, HISTORY_LEN)

        self.axes["speed"].set_ylim(0, 120)
        self.axes["steer"].set_ylim(-1, 1)
        self.axes["gas"].set_ylim(0, 1)
        self.axes["brake"].set_ylim(0, 1)
        self.axes["gyro"].set_ylim(-300, 300)

        self.texts = {}
        for k in ["speed", "steer", "gyro", "reward"]:
            self.texts[k] = self.axes[k].text(
                0.98, 0.75, "0", transform=self.axes[k].transAxes,
                color=colors[k], fontsize=13, fontweight="bold",
                ha="right", va="top"
            )

        # ABS 四輪
        self.abs_ax = self.fig.add_axes([0.1, 0.008, 0.8, 0.04],
                                         facecolor="#0d1117")
        self.abs_ax.set_xlim(0, 4)
        self.abs_ax.set_ylim(0, 1)
        self.abs_ax.axis("off")
        self.abs_ax.set_title("ABS grip  (blue=grip  red=slip)",
                               color="#cccccc", fontsize=8, pad=1)
        self.abs_rects = []
        for i, label in enumerate(["FL", "FR", "RL", "RR"]):
            rect = mpatches.FancyBboxPatch(
                (i + 0.1, 0.1), 0.8, 0.8,
                boxstyle="round,pad=0.05",
                facecolor="#1565c0", edgecolor="#333", lw=0.5
            )
            self.abs_ax.add_patch(rect)
            self.abs_ax.text(i + 0.5, 0.5, label,
                              ha="center", va="center",
                              color="white", fontsize=7)
            self.abs_rects.append(rect)

        plt.draw()
        plt.pause(0.001)

    def update(self, info: dict, action: np.ndarray, total_reward: float):
        speed = info.get("speed", 0.0) * 100
        steer = float(action[0])
        gas   = float(action[1])
        brake = float(action[2])
        gyro  = info.get("gyro", 0.0) * 1000
        abs_v = info.get("ABS", [1.0, 1.0, 1.0, 1.0])

        vals = {"speed": speed, "steer": steer, "gas": gas,
                "brake": brake, "gyro": gyro, "reward": total_reward}

        for k, v in vals.items():
            self.data[k].append(v)
            if len(self.data[k]) > HISTORY_LEN:
                self.data[k].pop(0)
            self.lines[k].set_data(range(len(self.data[k])), self.data[k])

        if self.data["reward"]:
            mn = min(self.data["reward"])
            mx = max(self.data["reward"])
            pad = max(abs(mx - mn) * 0.2, 10)
            self.axes["reward"].set_ylim(mn - pad, mx + pad)

        self.texts["speed"].set_text(f"{speed:.0f}")
        self.texts["steer"].set_text(f"{steer:+.2f}")
        self.texts["gyro"].set_text(f"{gyro:+.0f}")
        self.texts["reward"].set_text(f"{total_reward:.0f}")

        if hasattr(abs_v, '__len__') and len(abs_v) == 4:
            for rect, v in zip(self.abs_rects, abs_v):
                v = float(v)
                rect.set_facecolor((1 - v, 0, v))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def reset(self):
        for k in self.data:
            self.data[k].clear()

    def close(self):
        plt.close(self.fig)


# ── 環境 & 動作 ───────────────────────────────────────
def make_render_env(max_steps: int):
    def _init():
        return gym.make("CarRacing-v3", continuous=True,
                        render_mode="human",
                        max_episode_steps=max_steps)
    return _init


def postprocess_action(action: np.ndarray) -> np.ndarray:
    a = action.copy()
    a[0] = np.clip(a[0], -1.0,  1.0)
    a[1] = np.clip(a[1],  0.0,  1.0)
    a[2] = np.clip(a[2],  0.0,  1.0)
    if a[1] > 0.1 and a[2] > 0.1:
        if a[1] >= a[2]: a[2] = 0.0
        else:            a[1] = 0.0
    return a


# ── 主程式 ────────────────────────────────────────────
def play(model_path: str, n_episodes: int, max_steps: int, dashboard: bool):
    print(f"\n載入模型：{model_path}")
    print(f"每局最多 {max_steps if max_steps > 0 else '無限制'} 步\n")

    limit = max_steps if max_steps > 0 else 99999

    env = DummyVecEnv([make_render_env(limit)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    model = PPO.load(model_path, env=env)
    dash  = Dashboard() if dashboard else None
    all_rewards = []

    for ep in range(n_episodes):
        obs          = env.reset()
        done         = False
        total_reward = 0.0
        steps        = 0

        if dash:
            dash.reset()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action[0] = postprocess_action(action[0])
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps        += 1

            # 每 3 步更新一次儀表板
            if dash and steps % 3 == 0:
                raw_info = info[0] if isinstance(info, (list, tuple)) else info
                dash.update(raw_info, action[0], total_reward)

        all_rewards.append(total_reward)
        print(f"Episode {ep+1:2d}: reward = {total_reward:7.1f},  steps = {steps}")

    print(f"\n平均 reward : {np.mean(all_rewards):.1f}")
    print(f"標準差      : {np.std(all_rewards):.1f}")
    print(f"最高        : {max(all_rewards):.1f}")
    print(f"最低        : {min(all_rewards):.1f}")

    if dash:
        dash.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best/best_model",
                        help="模型路徑（不含 .zip）")
    parser.add_argument("--episodes", type=int, default=5,
                        help="跑幾局（預設 5）")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="每局最多幾步，0 = 無限制（預設 0）")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="關閉儀表板")
    args = parser.parse_args()
    play(args.model, args.episodes, args.max_steps, not args.no_dashboard)