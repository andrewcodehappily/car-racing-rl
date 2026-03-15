"""
自動找出所有 session 中最強的模型，複製到 timebest/
用法：python find_best.py
"""

import os
import re
import shutil
import glob
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
import gymnasium as gym

MODEL_DIR  = "models"
OUTPUT_DIR = "timebest"
N_EVAL_EPS = 5
N_STACK    = 4


def make_eval_env():
    def _init():
        return gym.make("CarRacing-v3", continuous=True, render_mode=None)
    return _init


def build_eval_env():
    env = DummyVecEnv([make_eval_env()])
    env = VecFrameStack(env, n_stack=N_STACK)
    env = VecTransposeImage(env)
    return env


def postprocess_action(action: np.ndarray) -> np.ndarray:
    a = action.copy()
    a[0] = np.clip(a[0], -1.0,  1.0)
    a[1] = np.clip(a[1],  0.0,  1.0)
    a[2] = np.clip(a[2],  0.0,  1.0)
    if a[1] > 0.1 and a[2] > 0.1:
        if a[1] >= a[2]:
            a[2] = 0.0
        else:
            a[1] = 0.0
    return a


def evaluate_model(model_path: str, n_eps: int = N_EVAL_EPS) -> float:
    """跑 n_eps 個 episode，回傳平均 reward"""
    try:
        env   = build_eval_env()
        model = PPO.load(model_path, env=env, device="cpu")
        rewards = []

        for _ in range(n_eps):
            obs  = env.reset()
            done = False
            total = 0.0
            steps = 0
            while not done and steps < 1000:
                action, _ = model.predict(obs, deterministic=True)
                action[0] = postprocess_action(action[0])
                obs, reward, done, _ = env.step(action)
                total += reward[0]
                steps += 1
            rewards.append(total)

        env.close()
        return float(np.mean(rewards))
    except Exception as e:
        print(f"    評估失敗：{e}")
        return -9999.0


def find_all_models() -> list:
    """找出所有候選模型路徑（不含 .zip）"""
    candidates = []

    # session_XX/best/best_model
    for p in glob.glob(os.path.join(MODEL_DIR, "session_*/best/best_model.zip")):
        candidates.append(p.replace(".zip", ""))

    # session_XX/final
    for p in glob.glob(os.path.join(MODEL_DIR, "session_*/final.zip")):
        candidates.append(p.replace(".zip", ""))

    # models/best/best_model
    p = os.path.join(MODEL_DIR, "best", "best_model.zip")
    if os.path.exists(p):
        candidates.append(p.replace(".zip", ""))

    # models/interrupted
    p = os.path.join(MODEL_DIR, "interrupted.zip")
    if os.path.exists(p):
        candidates.append(p.replace(".zip", ""))

    return sorted(set(candidates))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    candidates = find_all_models()
    if not candidates:
        print("找不到任何模型！請確認 models/ 資料夾存在。")
        return

    print(f"\n找到 {len(candidates)} 個候選模型，開始評估...\n")
    print(f"{'模型路徑':<55} {'平均 reward':>12}")
    print("─" * 70)

    results = []
    for path in candidates:
        score = evaluate_model(path, n_eps=N_EVAL_EPS)
        print(f"  {path:<53} {score:>10.1f}")
        results.append((score, path))

    # 找最強
    results.sort(reverse=True)
    best_score, best_path = results[0]

    print("\n" + "═" * 70)
    print(f"  最強模型：{best_path}")
    print(f"  平均 reward：{best_score:.1f}")
    print("═" * 70)

    # 複製到 timebest/
    dst_zip  = os.path.join(OUTPUT_DIR, "best_model.zip")
    dst_info = os.path.join(OUTPUT_DIR, "info.txt")

    shutil.copy2(best_path + ".zip", dst_zip)

    with open(dst_info, "w") as f:
        f.write(f"最強模型來源：{best_path}\n")
        f.write(f"平均 reward（{N_EVAL_EPS} eps）：{best_score:.1f}\n\n")
        f.write("所有候選模型排名：\n")
        for rank, (score, path) in enumerate(results, 1):
            f.write(f"  {rank}. {score:>8.1f}  {path}\n")

    print(f"\n  已儲存至 {dst_zip}")
    print(f"  排名報告 → {dst_info}")
    print(f"\n  使用方式：")
    print(f"    python play.py --model timebest/best_model")
    print(f"    python train.py --resume timebest/best_model --infinite\n")


if __name__ == "__main__":
    main()