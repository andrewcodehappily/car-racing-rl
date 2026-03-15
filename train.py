"""
CarRacing-v3 PPO 終極強化版
用法：
  從零開始   : python train.py
  繼續訓練   : python train.py --resume models/best/best_model
  無限訓練   : python train.py --resume models/best/best_model --infinite
  凍結CNN    : python train.py --resume models/best/best_model --freeze-cnn

強化項目：
  - Reward shaping：離軌懲罰、平滑駕駛、速度獎勵、急彎獎勵、完賽大獎
  - 在線穩定引導：自動監控 std，防止爆炸
  - 凍結 CNN 選項：只微調最後幾層，速度快 3~4 倍
  - n_steps 縮短為 256，更新更頻繁
  - 自動學習率衰減（自適應，後期越來越慢）
  - optimizer lr 直接同步，不再有顯示 bug
  - Ctrl+C 安全中斷儲存
"""

import os
import time
import argparse
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback, CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# ════════════════════════════════════════════════════════
#  全域設定
# ════════════════════════════════════════════════════════
SEED        = 42
N_STACK     = 4
N_ENVS      = 4
TOTAL_STEPS = 1_000_000
EVAL_FREQ   = 20_000
N_EVAL_EPS  = 5
SAVE_FREQ   = 50_000
VIZ_FREQ    = 50_000
LOG_DIR     = "logs"
MODEL_DIR   = "models"
LR_MIN      = 5e-6


def adaptive_lr_decay(lr: float, base_lr: float) -> float:
    headroom   = (lr - LR_MIN) / (base_lr - LR_MIN + 1e-10)
    headroom   = max(0.0, min(1.0, headroom))
    decay_rate = 0.30 * headroom
    return max(lr * (1.0 - decay_rate), LR_MIN)


def set_lr(model, lr: float):
    """同時更新 SB3 屬性和 optimizer 內部 lr，確保兩邊一致"""
    model.learning_rate = lambda _: lr   # 用 lambda 包住，讓 SB3 schedule 回傳固定值
    for pg in model.policy.optimizer.param_groups:
        pg["lr"] = lr
    # 驗證
    actual = model.policy.optimizer.param_groups[0]["lr"]
    print(f"  [lr 確認] optimizer 實際 lr = {actual:.2e}")


# ════════════════════════════════════════════════════════
#  Reward Shaping Wrapper（含急彎獎勵）
# ════════════════════════════════════════════════════════
class RewardShapingWrapper(gym.Wrapper):
    OFF_TRACK_LIMIT  = 50
    OFF_TRACK_COEF   = 0.05
    OFF_TRACK_TERM   = 5.0
    SMOOTH_BONUS     = 0.02
    JERK_PENALTY     = 0.05
    SMOOTH_THRESH    = 0.1
    JERK_THRESH      = 0.5
    SPEED_COEF       = 0.001
    SHARP_TURN_BONUS = 0.5    # 急彎還在賽道上額外加分
    SHARP_TURN_THRESH= 0.4    # 轉向 > 此值算急彎
    FINISH_BONUS     = 50.0

    def __init__(self, env):
        super().__init__(env)
        self._prev_action     = np.zeros(3)
        self._off_track_steps = 0
        self._spinning_steps  = 0   # 打圈偵測
        self._total_steps     = 0   # 總步數（算進度用）
        self._tiles_visited   = 0   # 已訪問賽道磚數

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action     = np.zeros(3)
        self._off_track_steps = 0
        self._spinning_steps  = 0
        self._total_steps     = 0
        self._tiles_visited   = 0
        return obs, info

    def step(self, action):
        # 動作後處理：油門煞車互斥 + 限制範圍
        # 訓練時就修正，讓模型學到正確的動作習慣
        if isinstance(action, np.ndarray):
            action = action.copy()
            action[0] = np.clip(action[0], -1.0, 1.0)   # steering
            action[1] = np.clip(action[1],  0.0, 1.0)   # gas
            action[2] = np.clip(action[2],  0.0, 1.0)   # brake
            # 油門煞車互斥：保留較大的那個
            if action[1] > 0.1 and action[2] > 0.1:
                if action[1] >= action[2]:
                    action[2] = 0.0
                else:
                    action[1] = 0.0

        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = reward

        # 1. 離軌漸進懲罰 + 早停
        if reward < 0:
            self._off_track_steps += 1
            shaped -= self.OFF_TRACK_COEF * self._off_track_steps
        else:
            self._off_track_steps = 0

        if self._off_track_steps > self.OFF_TRACK_LIMIT:
            terminated  = True
            shaped     -= self.OFF_TRACK_TERM

        if isinstance(action, np.ndarray):
            diff = abs(action[0] - self._prev_action[0])

            # 2. 平滑駕駛獎懲
            if diff < self.SMOOTH_THRESH:
                shaped += self.SMOOTH_BONUS
            elif diff > self.JERK_THRESH:
                shaped -= self.JERK_PENALTY

            # 3. 急彎獎勵：轉向大但還在賽道上 → 成功過急彎
            if reward > 0 and abs(action[0]) > self.SHARP_TURN_THRESH:
                shaped += self.SHARP_TURN_BONUS

        # 4. 速度獎勵
        if reward > 0:
            shaped += reward * self.SPEED_COEF
            self._tiles_visited += 1

        # 5. 打圈懲罰（溫和版，不改變 reward 數量級）
        self._total_steps += 1
        if isinstance(action, np.ndarray) and abs(action[0]) > 0.5 and reward <= 0:
            self._spinning_steps += 1
        else:
            self._spinning_steps = max(0, self._spinning_steps - 1)

        if self._spinning_steps > 40:
            shaped -= 0.1               # 每步只扣 0.1，不累加
        if self._spinning_steps > 100:
            terminated = True
            shaped    -= 3.0            # 強制結束只扣 3 分，跟 OFF_TRACK_TERM 同級

        # 6. 完賽大獎
        if terminated and self._off_track_steps == 0 and self._spinning_steps == 0:
            shaped += self.FINISH_BONUS

        self._prev_action = np.array(action) if isinstance(action, np.ndarray) else np.zeros(3)
        return obs, shaped, terminated, truncated, info


# ════════════════════════════════════════════════════════
#  進度 Callback
# ════════════════════════════════════════════════════════
class ProgressCallback(BaseCallback):
    def __init__(self, viz_freq=50_000, total_steps=TOTAL_STEPS, session=1, verbose=0):
        super().__init__(verbose)
        self.viz_freq    = viz_freq
        self.total_steps = total_steps
        self.session     = session
        self.ep_rewards  = []
        self.ep_lengths  = []
        self._last_viz   = 0
        self._t0         = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self.ep_rewards.append(ep["r"])
                self.ep_lengths.append(ep["l"])
                if len(self.ep_rewards) % 10 == 0:
                    self.logger.record("custom/mean_reward_last10", np.mean(self.ep_rewards[-10:]))
                    self.logger.record("custom/mean_length_last10", np.mean(self.ep_lengths[-10:]))
                    self.logger.record("custom/best_reward", max(self.ep_rewards))
                    self.logger.record("custom/session", self.session)

        if self.num_timesteps - self._last_viz >= self.viz_freq:
            self._last_viz = self.num_timesteps
            self._print()
        return True

    def _print(self):
        elapsed = time.time() - self._t0
        steps   = self.num_timesteps
        pct     = min(steps / self.total_steps * 100, 100)
        fps     = int(steps / elapsed) if elapsed > 0 else 0
        eta     = int((self.total_steps - steps) / fps) if fps > 0 else 0
        recent  = self.ep_rewards[-20:] if self.ep_rewards else [0]
        bar     = "█" * int(30 * pct / 100) + "░" * (30 - int(30 * pct / 100))

        print("\n" + "═" * 58)
        print(f"  🏎  Session {self.session}  |  {steps:,} / {self.total_steps:,} steps")
        print(f"  [{bar}] {pct:.1f}%")
        print("─" * 58)
        print(f"  {'已跑 episodes':<22} {len(self.ep_rewards)}")
        print(f"  {'近20eps平均reward':<22} {np.mean(recent):.1f}")
        print(f"  {'歷史最佳reward':<22} {max(self.ep_rewards) if self.ep_rewards else 0:.1f}")
        print(f"  {'訓練速度':<22} {fps} steps/s")
        print(f"  {'已花時間':<22} {int(elapsed//60)}m {int(elapsed%60)}s")
        print(f"  {'預計剩餘':<22} {eta//3600}h {(eta%3600)//60}m")
        print("═" * 58 + "\n")


# ════════════════════════════════════════════════════════
#  穩定引導 Callback
# ════════════════════════════════════════════════════════
class StabilizeCallback(BaseCallback):
    CHECK_FREQ = 2048

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._last_check = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check < self.CHECK_FREQ:
            return True
        self._last_check = self.num_timesteps

        try:
            std = self.model.policy.action_dist.distribution.scale.mean().item()
        except Exception:
            return True

        ent = self.model.ent_coef

        cur_lr = self.model.policy.optimizer.param_groups[0]["lr"]

        if std > 6.0:
            new_ent = max(ent * 0.5, 1e-4)
            new_lr  = max(cur_lr * 0.5, LR_MIN)
            print(f"  [穩定] std={std:.2f} 危險！ent→{new_ent:.4f} lr {cur_lr:.2e}→{new_lr:.2e} clip→0.08")
            self.model.ent_coef   = new_ent
            self.model.clip_range = lambda _: 0.08
            set_lr(self.model, new_lr)
        elif std > 4.0:
            new_ent = max(ent * 0.7, 1e-4)
            new_lr  = max(cur_lr * 0.7, LR_MIN)
            print(f"  [穩定] std={std:.2f} 偏高 → ent→{new_ent:.4f} lr {cur_lr:.2e}→{new_lr:.2e}")
            self.model.ent_coef = new_ent
            set_lr(self.model, new_lr)
        elif std < 1.5 and ent < 0.005:
            new_ent = min(ent * 1.2, 0.005)
            print(f"  [穩定] std={std:.2f} 偏低 → ent→{new_ent:.4f}")
            self.model.ent_coef = new_ent

        self.logger.record("custom/std_live",      std)
        self.logger.record("custom/ent_coef_live", self.model.ent_coef)
        return True


# ════════════════════════════════════════════════════════
#  環境建立
# ════════════════════════════════════════════════════════
def make_env(rank=0, seed=0):
    def _init():
        env = gym.make("CarRacing-v3", continuous=True,
                       lap_complete_percent=0.95,   # 跑完 95% 才算完賽，逼它處理急彎
                       render_mode=None)
        env = RewardShapingWrapper(env)
        env = Monitor(env, filename=os.path.join(LOG_DIR, f"monitor_{rank}"))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def build_vec_env(n_envs=N_ENVS):
    envs = DummyVecEnv([make_env(i, SEED) for i in range(n_envs)])
    envs = VecFrameStack(envs, n_stack=N_STACK)
    envs = VecTransposeImage(envs)
    return envs


# ════════════════════════════════════════════════════════
#  單輪訓練
# ════════════════════════════════════════════════════════
def run_session(model, train_env, eval_env, steps, session, reset_timesteps):
    session_dir = os.path.join(MODEL_DIR, f"session_{session:02d}")
    best_dir    = os.path.join(session_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    callbacks = CallbackList([
        StabilizeCallback(),
        CheckpointCallback(
            save_freq   = SAVE_FREQ // N_ENVS,
            save_path   = session_dir,
            name_prefix = f"s{session:02d}_ppo",
            verbose     = 1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path = best_dir,
            log_path             = LOG_DIR,
            eval_freq            = EVAL_FREQ // N_ENVS,
            n_eval_episodes      = N_EVAL_EPS,
            deterministic        = True,
            render               = False,
            verbose              = 1,
        ),
        ProgressCallback(viz_freq=VIZ_FREQ, total_steps=steps, session=session),
    ])

    model.learn(
        total_timesteps     = steps,
        callback            = callbacks,
        progress_bar        = True,
        reset_num_timesteps = reset_timesteps,
    )

    out = os.path.join(session_dir, "final")
    model.save(out)
    print(f"\n  Session {session} 完成 → {out}.zip")
    return out


# ════════════════════════════════════════════════════════
#  主程式
# ════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="CarRacing-v3 PPO 終極強化版",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--resume", type=str, default=None, metavar="PATH",
        help="繼續訓練：模型路徑（不含 .zip）")
    parser.add_argument("--infinite", action="store_true",
        help="無限模式：跑完自動繼續，學習率自適應衰減")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS,
        help=f"每輪步數（預設 {TOTAL_STEPS:,}）")
    parser.add_argument("--lr", type=float, default=None,
        help="覆寫學習率（預設：新訓練 3e-4 / 繼續 5e-5）")
    parser.add_argument("--device", type=str, default="mps",
        help="訓練裝置：mps / cuda / cpu")
    parser.add_argument("--freeze-cnn", action="store_true",
        help="凍結 CNN，只微調 Actor/Critic 頭（速度快 3~4 倍）")
    args = parser.parse_args()

    os.makedirs(LOG_DIR,   exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 根據狀況決定學習率
    # 你現在的情況：std 偏高、reward 退步，用更小的 lr
    if args.lr is not None:
        base_lr = args.lr
    elif args.resume:
        base_lr = 5e-5   # 比之前更保守（之前 1e-4 還是太大）
    else:
        base_lr = 3e-4

    print("\n" + "═" * 58)
    print("  🏎  CarRacing-v3 PPO 終極強化版")
    print("─" * 58)
    print(f"  模式        : {'無限' if args.infinite else '單輪'}")
    print(f"  起點        : {'繼續 (' + args.resume + ')' if args.resume else '從零'}")
    print(f"  每輪步數    : {args.steps:,}")
    print(f"  起始學習率  : {base_lr:.2e}")
    print(f"  訓練裝置    : {args.device}")
    print(f"  凍結CNN     : {'是（只微調頭部）' if args.freeze_cnn else '否'}")
    print("═" * 58 + "\n")

    train_env = build_vec_env(N_ENVS)
    eval_env  = build_vec_env(1)

    if args.resume:
        model = PPO.load(args.resume, env=train_env, device=args.device)
        set_lr(model, base_lr)
        model.clip_range = lambda _: 0.10   # 更保守
        model.ent_coef   = 0.0005           # 抑制 std
        # 注意：n_steps 不能在 resume 後改，會導致 rollout buffer 大小不符
        # 維持載入模型的原始 n_steps

        # 凍結 CNN
        if args.freeze_cnn:
            for p in model.policy.features_extractor.parameters():
                p.requires_grad = False
            print("  CNN 已凍結，只訓練 Actor/Critic 頭")

        print(f"  lr={base_lr:.2e}  clip=0.10  ent=0.0005  n_steps=256")
        session_start = 2
        reset_ts      = False
    else:
        model = PPO(
            env             = train_env,
            policy          = "CnnPolicy",
            n_steps         = 256,       # 比預設 512 更頻繁
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            learning_rate   = base_lr,
            verbose         = 1,
            tensorboard_log = LOG_DIR,
            seed            = SEED,
            device          = args.device,
        )
        session_start = 1
        reset_ts      = True

    session = session_start
    lr      = base_lr

    try:
        while True:
            print(f"\n  ▶ Session {session}  目標lr={lr:.2e}")
            set_lr(model, lr)

            run_session(
                model           = model,
                train_env       = train_env,
                eval_env        = eval_env,
                steps           = args.steps,
                session         = session,
                reset_timesteps = reset_ts,
            )
            reset_ts = False

            if not args.infinite:
                break

            session += 1
            lr       = adaptive_lr_decay(lr, base_lr)
            print(f"\n  ♻  Session {session}，新學習率 {lr:.2e}")

    except KeyboardInterrupt:
        print("\n  ⏹  中斷！儲存中...")
        model.save(os.path.join(MODEL_DIR, "interrupted"))
        print(f"  已儲存 → models/interrupted.zip")

    finally:
        train_env.close()
        eval_env.close()
        print("  完成。\n")


if __name__ == "__main__":
    main()