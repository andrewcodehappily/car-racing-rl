"""
人機對戰模式！看誰分數高
用法：python versus.py --model timebest/best_model

控制：
  方向鍵左右  : 轉向
  方向鍵上    : 油門
  方向鍵下    : 煞車
  Q           : 放棄這局
"""

import argparse
import threading
import numpy as np
import gymnasium as gym
import pygame

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage


# ── 鍵盤狀態（全域，讓兩個執行緒都能讀）──────────────
class KeyState:
    def __init__(self):
        self.left   = False
        self.right  = False
        self.gas    = False
        self.brake  = False
        self.quit   = False

key_state = KeyState()


# 人類控制靈敏度設定
HUMAN_STEERING  = 0.5    # 最大轉向幅度
HUMAN_GAS       = 0.8    # 油門強度
HUMAN_BRAKE     = 0.6    # 煞車強度
STEER_ACCEL     = 0.03   # 按下時每幀增加多少（越小越慢轉）
STEER_RETURN    = 0.12   # 放開後每幀回正多少（越大回正越快）

_current_steer = 0.0

def get_human_action() -> np.ndarray:
    global _current_steer

    pressing = key_state.left or key_state.right

    if key_state.left:
        # 按住左：慢慢往左增加
        _current_steer = max(_current_steer - STEER_ACCEL, -HUMAN_STEERING)
    elif key_state.right:
        # 按住右：慢慢往右增加
        _current_steer = min(_current_steer + STEER_ACCEL,  HUMAN_STEERING)
    else:
        # 放開：自動回正
        if _current_steer > 0:
            _current_steer = max(_current_steer - STEER_RETURN, 0.0)
        elif _current_steer < 0:
            _current_steer = min(_current_steer + STEER_RETURN, 0.0)

    gas   = HUMAN_GAS   if key_state.gas   else 0.0
    brake = HUMAN_BRAKE if key_state.brake else 0.0
    return np.array([_current_steer, gas, brake], dtype=np.float32)


def postprocess(a: np.ndarray) -> np.ndarray:
    a = a.copy()
    a[0] = np.clip(a[0], -1.0, 1.0)
    a[1] = np.clip(a[1],  0.0, 1.0)
    a[2] = np.clip(a[2],  0.0, 1.0)
    if a[1] > 0.1 and a[2] > 0.1:
        if a[1] >= a[2]:
            a[2] = 0.0
        else:
            a[1] = 0.0
    return a


def run_ai_episode(model_path: str, result: dict, idx: int, max_steps: int):
    """AI 在背景跑，結果存進 result[idx]"""
    env = gym.make("CarRacing-v3", continuous=True,
                   render_mode=None, max_episode_steps=max_steps)
    vec = DummyVecEnv([lambda: env])
    vec = VecFrameStack(vec, n_stack=4)
    vec = VecTransposeImage(vec)

    model = PPO.load(model_path, env=vec)
    obs   = vec.reset()
    total = 0.0
    steps = 0
    done  = False

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        action[0] = postprocess(action[0])
        obs, reward, done, _ = vec.step(action)
        total += reward[0]
        steps += 1

    vec.close()
    result[idx] = {"reward": total, "steps": steps}


def run_human_episode(max_steps: int) -> dict:
    """人類用 pygame 視窗玩"""
    env  = gym.make("CarRacing-v3", continuous=True,
                    render_mode="human", max_episode_steps=max_steps)
    obs, _ = env.reset()
    total  = 0.0
    steps  = 0
    clock  = pygame.time.Clock()

    print("\n  🎮 你的回合！方向鍵控制，Q 放棄")
    print("  ↑油門  ↓煞車  ←→轉向\n")

    while steps < max_steps:
        # 處理 pygame 事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                key_state.quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:   key_state.left  = True
                if event.key == pygame.K_RIGHT:  key_state.right = True
                if event.key == pygame.K_UP:     key_state.gas   = True
                if event.key == pygame.K_DOWN:   key_state.brake = True
                if event.key == pygame.K_q:      key_state.quit  = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:   key_state.left  = False
                if event.key == pygame.K_RIGHT:  key_state.right = False
                if event.key == pygame.K_UP:     key_state.gas   = False
                if event.key == pygame.K_DOWN:   key_state.brake = False

        if key_state.quit:
            break

        action = get_human_action()
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps += 1

        if terminated or truncated:
            break

        clock.tick(60)

    env.close()
    key_state.quit  = False
    key_state.left  = False
    key_state.right = False
    key_state.gas   = False
    key_state.brake = False
    global _current_steer
    _current_steer  = 0.0   # 重置轉向平滑狀態
    return {"reward": total, "steps": steps}


def print_result(human: dict, ai: dict, round_n: int):
    h = human["reward"]
    a = ai["reward"]
    winner = "🏆 你贏了！" if h > a else ("🤖 AI 贏了！" if a > h else "🤝 平手！")

    print("\n" + "═" * 45)
    print(f"  第 {round_n} 局結果")
    print("─" * 45)
    print(f"  👤 你   : {h:7.1f} 分  ({human['steps']} 步)")
    print(f"  🤖 AI   : {a:7.1f} 分  ({ai['steps']} 步)")
    print(f"  {winner}")
    print("═" * 45 + "\n")


def versus(model_path: str, rounds: int, max_steps: int):
    pygame.init()

    human_total = 0.0
    ai_total    = 0.0

    print("\n" + "═" * 45)
    print("  🏎  人機對戰模式！")
    print(f"  模型：{model_path}")
    print(f"  共 {rounds} 局，每局最多 {max_steps} 步")
    print("═" * 45)

    for r in range(1, rounds + 1):
        print(f"\n  ── 第 {r} 局 ──")
        print("  先讓 AI 跑（背景執行中）...")

        # AI 在背景跑
        ai_result = {}
        ai_thread = threading.Thread(
            target=run_ai_episode,
            args=(model_path, ai_result, 0, max_steps),
            daemon=True,
        )
        ai_thread.start()

        # 人類玩
        human_result = run_human_episode(max_steps)

        # 等 AI 跑完
        ai_thread.join()
        ai_r = ai_result.get(0, {"reward": 0.0, "steps": 0})

        human_total += human_result["reward"]
        ai_total    += ai_r["reward"]

        print_result(human_result, ai_r, r)

        if r < rounds:
            input("  按 Enter 繼續下一局...")

    # 總結
    print("═" * 45)
    print("  最終總分")
    print("─" * 45)
    print(f"  👤 你   : {human_total:.1f} 分")
    print(f"  🤖 AI   : {ai_total:.1f} 分")
    if human_total > ai_total:
        print("  🏆 恭喜你贏了！")
    elif ai_total > human_total:
        print("  🤖 AI 獲勝！繼續訓練你自己吧 XD")
    else:
        print("  🤝 平手！")
    print("═" * 45 + "\n")

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="timebest/best_model",
                        help="AI 模型路徑（不含 .zip）")
    parser.add_argument("--rounds", type=int, default=3,
                        help="對戰幾局（預設 3）")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="每局最多幾步（預設 1000）")
    parser.add_argument("--sensitivity", type=float, default=0.4,
                        help="轉向靈敏度 0.1~1.0（預設 0.4，越小越穩）")
    args = parser.parse_args()
    import versus as _m
    _m.HUMAN_STEERING = args.sensitivity
    versus(args.model, args.rounds, args.max_steps)