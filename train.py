"""
CarRacing-v3 PPO 最終版
════════════════════════════════════════════════════════
用法：
  從零開始   : python train.py
  繼續訓練   : python train.py --resume models/best/best_model
  無限訓練   : python train.py --resume models/best/best_model --infinite
  凍結CNN    : python train.py --resume models/best/best_model --freeze-cnn
  指定裝置   : python train.py --device cpu

════════════════════════════════════════════════════════
"""

import os
import sys
import re
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

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


# ════════════════════════════════════════════════════════
#  工具函式
# ════════════════════════════════════════════════════════
def resolve_device(requested: str) -> str:
    """[修正10] 自動 fallback：mps → cuda → cpu"""
    import torch
    order = [requested, "mps", "cuda", "cpu"]
    seen  = set()
    for dev in order:
        if dev in seen:
            continue
        seen.add(dev)
        try:
            if dev == "mps"  and not torch.backends.mps.is_available():
                continue
            if dev == "cuda" and not torch.cuda.is_available():
                continue
            print(f"  [裝置] 使用 {dev}")
            return dev
        except Exception:
            continue
    return "cpu"


def detect_session_start(model_dir: str, is_resume: bool) -> int:
    """[修正9] 自動偵測目前最大 session 號碼，避免重複"""
    if not is_resume:
        return 1
    max_s = 1
    if os.path.isdir(model_dir):
        for name in os.listdir(model_dir):
            m = re.match(r"session_(\d+)$", name)
            if m:
                max_s = max(max_s, int(m.group(1)))
    return max_s + 1


class LRHolder:
    """可變的 lr 容器，當 SB3 schedule function 使用
    用可變物件解決 lambda 閉包問題：
    每次 SB3 呼叫 schedule(progress) 都會讀 self.value，
    所以只要改 holder.value 就能真正影響訓練。
    """
    def __init__(self, lr: float):
        self.value = lr
    def __call__(self, _progress: float) -> float:
        return self.value

# 全域 holder，整個訓練過程共用同一個物件
_lr_holder = LRHolder(3e-4)

def set_lr(model: PPO, lr: float) -> None:
    """[修正1] 透過 LRHolder 確保 lr 真正生效"""
    _lr_holder.value = lr
    # 確保 model 的 schedule 指向同一個 holder
    if not isinstance(model.learning_rate, LRHolder):
        model.learning_rate = _lr_holder
    # 同時直接更新 optimizer（雙保險）
    for pg in model.policy.optimizer.param_groups:
        pg["lr"] = lr
    actual = model.policy.optimizer.param_groups[0]["lr"]
    print(f"  [lr] optimizer 實際 lr = {actual:.2e}  holder = {_lr_holder.value:.2e}")


def adaptive_lr_decay(lr: float, base_lr: float) -> float:
    """[修正8] 自適應衰減：越接近 LR_MIN 衰減越小"""
    headroom   = (lr - LR_MIN) / max(base_lr - LR_MIN, 1e-10)
    headroom   = max(0.0, min(1.0, headroom))
    decay_rate = 0.30 * headroom
    return max(lr * (1.0 - decay_rate), LR_MIN)


def verify_model_path(path: str) -> str:
    """[修正11] 接受有無 .zip 的路徑，確認檔案存在"""
    clean = path.strip().rstrip("/")
    if clean.endswith(".zip"):
        clean = clean[:-4]
    if not os.path.exists(clean + ".zip"):
        print(f"\n  ❌ 找不到模型：{clean}.zip")
        print(f"     可能的路徑：")
        print(f"       models/best/best_model")
        print(f"       timebest/best_model")
        print(f"       models/session_02/best/best_model\n")
        sys.exit(1)
    return clean


def safe_save(model: PPO, path: str) -> bool:
    """安全儲存，回傳是否成功"""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        model.save(path)
        print(f"  ✅ 已儲存 → {path}.zip")
        return True
    except Exception as e:
        print(f"  ❌ 儲存失敗：{e}")
        return False


# ════════════════════════════════════════════════════════
#  Reward Shaping Wrapper
# ════════════════════════════════════════════════════════
class RewardShapingWrapper(gym.Wrapper):
    """
    [修正3] 所有 shaping 數量級控制：
      單步最大正向  ≈ +0.52（急彎 0.5 + 速度 0.001 + 平滑 0.02）
      單步最大負向  ≈ -0.15（離軌漸進 0.05 × n + 打圈 0.1）
      一次性懲罰    5.0（早停）/ 3.0（打圈）
      完賽大獎      50.0（一局只觸發一次）
    → 不改變 reward 數量級，value_loss 不會爆炸
    """
    OFF_TRACK_LIMIT   = 50
    OFF_TRACK_COEF    = 0.05
    OFF_TRACK_TERM    = 5.0
    SMOOTH_BONUS      = 0.02
    JERK_PENALTY      = 0.05
    SMOOTH_THRESH     = 0.1
    JERK_THRESH       = 0.5
    SPEED_COEF        = 0.001
    SHARP_TURN_BONUS  = 0.5
    SHARP_TURN_THRESH = 0.4
    SPIN_WARN         = 40
    SPIN_TERM         = 100
    SPIN_PENALTY      = 0.1
    SPIN_TERM_PENALTY = 3.0
    FINISH_BONUS      = 50.0

    def __init__(self, env):
        super().__init__(env)
        self._reset_state()

    def _reset_state(self):
        self._prev_action     = np.zeros(3, dtype=np.float32)
        self._off_track_steps = 0
        self._spinning_steps  = 0
        self._total_steps     = 0
        self._tiles_visited   = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_state()
        return obs, info

    def step(self, action):
        # [修正5] 動作後處理：訓練時也互斥，讓模型學到正確習慣
        if isinstance(action, np.ndarray):
            action = action.copy().astype(np.float32)
            action[0] = np.clip(action[0], -1.0,  1.0)
            action[1] = np.clip(action[1],  0.0,  1.0)
            action[2] = np.clip(action[2],  0.0,  1.0)
            if action[1] > 0.1 and action[2] > 0.1:
                if action[1] >= action[2]:
                    action[2] = 0.0
                else:
                    action[1] = 0.0

        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = float(reward)

        # 1. 離軌漸進懲罰 + 早停
        if reward < 0:
            self._off_track_steps += 1
            shaped -= self.OFF_TRACK_COEF * self._off_track_steps
        else:
            self._off_track_steps = 0
            self._tiles_visited  += 1

        if self._off_track_steps > self.OFF_TRACK_LIMIT:
            terminated = True
            shaped    -= self.OFF_TRACK_TERM

        if isinstance(action, np.ndarray):
            diff = abs(float(action[0]) - float(self._prev_action[0]))

            # 2. 平滑駕駛獎懲
            if diff < self.SMOOTH_THRESH:
                shaped += self.SMOOTH_BONUS
            elif diff > self.JERK_THRESH:
                shaped -= self.JERK_PENALTY

            # 3. 急彎獎勵：轉向大但在賽道上
            if reward > 0 and abs(float(action[0])) > self.SHARP_TURN_THRESH:
                shaped += self.SHARP_TURN_BONUS

        # 4. 速度獎勵
        if reward > 0:
            shaped += reward * self.SPEED_COEF

        # 5. 打圈偵測（固定懲罰，不累加，避免數量級問題）
        self._total_steps += 1
        if isinstance(action, np.ndarray) and abs(float(action[0])) > 0.5 and reward <= 0:
            self._spinning_steps += 1
        else:
            self._spinning_steps = max(0, self._spinning_steps - 1)

        if self._spinning_steps > self.SPIN_WARN:
            shaped -= self.SPIN_PENALTY
        if self._spinning_steps > self.SPIN_TERM:
            terminated = True
            shaped    -= self.SPIN_TERM_PENALTY

        # 6. 完賽大獎
        # [修正17] 確認真的跑完，不是因為任何懲罰結束
        if terminated and not truncated \
                and self._off_track_steps == 0 \
                and self._spinning_steps  == 0:
            shaped += self.FINISH_BONUS

        self._prev_action = (
            action.copy() if isinstance(action, np.ndarray)
            else np.zeros(3, dtype=np.float32)
        )
        return obs, shaped, terminated, truncated, info


# ════════════════════════════════════════════════════════
#  進度 Callback
# ════════════════════════════════════════════════════════
class ProgressCallback(BaseCallback):
    def __init__(self, viz_freq=50_000, total_steps=TOTAL_STEPS,
                 session=1, verbose=0):
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
                self.ep_rewards.append(float(ep["r"]))
                self.ep_lengths.append(int(ep["l"]))
                if len(self.ep_rewards) % 10 == 0:
                    self.logger.record("custom/mean_reward_last10",
                                       float(np.mean(self.ep_rewards[-10:])))
                    self.logger.record("custom/mean_length_last10",
                                       float(np.mean(self.ep_lengths[-10:])))
                    self.logger.record("custom/best_reward",
                                       float(max(self.ep_rewards)))
                    self.logger.record("custom/session", self.session)

        if self.num_timesteps - self._last_viz >= self.viz_freq:
            self._last_viz = self.num_timesteps
            self._print()
        return True

    def _print(self):
        elapsed = time.time() - self._t0
        steps   = self.num_timesteps
        pct     = min(steps / max(self.total_steps, 1) * 100, 100)
        # [修正16] elapsed > 1 才計算，避免 fps=0 除以零
        fps     = int(steps / elapsed) if elapsed > 1 else 0
        eta     = int((self.total_steps - steps) / fps) if fps > 0 else 0
        recent  = self.ep_rewards[-20:] if self.ep_rewards else [0]
        bar     = "█" * int(30 * pct / 100) + "░" * (30 - int(30 * pct / 100))
        best    = max(self.ep_rewards) if self.ep_rewards else 0

        print("\n" + "═" * 60)
        print(f"  🏎  Session {self.session}  |  {steps:,} / {self.total_steps:,} steps")
        print(f"  [{bar}] {pct:.1f}%")
        print("─" * 60)
        print(f"  {'已跑 episodes':<24} {len(self.ep_rewards)}")
        print(f"  {'近20eps平均reward':<24} {np.mean(recent):.1f}")
        print(f"  {'歷史最佳reward':<24} {best:.1f}")
        print(f"  {'訓練速度':<24} {fps} steps/s")
        print(f"  {'已花時間':<24} {int(elapsed//60)}m {int(elapsed%60)}s")
        print(f"  {'預計剩餘':<24} {eta//3600}h {(eta%3600)//60}m")
        print("═" * 60 + "\n")


# ════════════════════════════════════════════════════════
#  穩定引導 Callback
# ════════════════════════════════════════════════════════
class StabilizeCallback(BaseCallback):
    """
    [修正4] 每 CHECK_FREQ 步監控 std：
      std > 6   → 危險，同時降 ent_coef / lr / clip
      std > 4   → 偏高，降 ent_coef + lr
      std < 1.5 → 偏低，稍微提高 ent_coef 防過擬合
    """
    CHECK_FREQ = 2048

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._last_check = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check < self.CHECK_FREQ:
            return True
        self._last_check = self.num_timesteps

        try:
            std = float(
                self.model.policy.action_dist.distribution.scale.mean().item()
            )
        except Exception:
            return True

        ent    = float(self.model.ent_coef)
        cur_lr = float(self.model.policy.optimizer.param_groups[0]["lr"])

        if std > 6.0:
            new_ent = max(ent * 0.5, 1e-4)
            new_lr  = max(cur_lr * 0.5, LR_MIN)
            print(f"\n  [穩定] ⚠️  std={std:.2f} 危險！"
                  f"  ent {ent:.4f}→{new_ent:.4f}"
                  f"  lr {cur_lr:.2e}→{new_lr:.2e}  clip→0.08")
            self.model.ent_coef   = new_ent
            self.model.clip_range = lambda _: 0.08
            set_lr(self.model, new_lr)

        elif std > 4.0:
            new_ent = max(ent * 0.7, 1e-4)
            new_lr  = max(cur_lr * 0.7, LR_MIN)
            print(f"\n  [穩定] std={std:.2f} 偏高"
                  f"  ent {ent:.4f}→{new_ent:.4f}"
                  f"  lr {cur_lr:.2e}→{new_lr:.2e}")
            self.model.ent_coef = new_ent
            set_lr(self.model, new_lr)

        elif std < 1.5 and ent < 0.005:
            new_ent = min(ent * 1.2, 0.005)
            print(f"\n  [穩定] std={std:.2f} 偏低  ent {ent:.4f}→{new_ent:.4f}")
            self.model.ent_coef = new_ent

        self.logger.record("custom/std_live",      std)
        self.logger.record("custom/ent_coef_live", float(self.model.ent_coef))
        self.logger.record("custom/lr_live",       cur_lr)
        return True


# ════════════════════════════════════════════════════════
#  環境建立
# ════════════════════════════════════════════════════════
def make_env(rank: int = 0, seed: int = 0, session: int = 1):
    def _init():
        env = gym.make(
            "CarRacing-v3",
            continuous           = True,
            lap_complete_percent = 0.95,
            render_mode          = None,
        )
        env = RewardShapingWrapper(env)
        # [修正18] monitor 檔名加 session，避免多次 resume 後資料混雜
        monitor_path = os.path.join(
            LOG_DIR, f"s{session:02d}_monitor_{rank}"
        )
        env = Monitor(env, filename=monitor_path)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def build_vec_env(n_envs: int = N_ENVS, session: int = 1):
    """[修正13] 包 try/except，印出友善錯誤"""
    try:
        envs = DummyVecEnv([make_env(i, SEED, session) for i in range(n_envs)])
        envs = VecFrameStack(envs, n_stack=N_STACK)
        envs = VecTransposeImage(envs)
        return envs
    except Exception as e:
        print(f"\n  ❌ 建立環境失敗：{e}")
        print("  請確認已執行 bash setup.sh 且虛擬環境已啟動")
        sys.exit(1)


# ════════════════════════════════════════════════════════
#  單輪訓練
# ════════════════════════════════════════════════════════
def run_session(model: PPO, train_env, eval_env,
                steps: int, session: int, reset_timesteps: bool) -> str:
    # [修正15] 每個 session 獨立資料夾
    session_dir = os.path.join(MODEL_DIR, f"session_{session:02d}")
    best_dir    = os.path.join(session_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    callbacks = CallbackList([
        StabilizeCallback(),
        CheckpointCallback(
            save_freq   = max(SAVE_FREQ // N_ENVS, 1),
            save_path   = session_dir,
            name_prefix = f"s{session:02d}_ppo",
            verbose     = 1,
        ),
        # [修正14] verbose=1 讓 EvalCallback 的錯誤可以被看到
        EvalCallback(
            eval_env,
            best_model_save_path = best_dir,
            log_path             = LOG_DIR,
            eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
            n_eval_episodes      = N_EVAL_EPS,
            deterministic        = True,
            render               = False,
            verbose              = 1,
            warn                 = True,
        ),
        ProgressCallback(
            viz_freq    = VIZ_FREQ,
            total_steps = steps,
            session     = session,
        ),
    ])

    model.learn(
        total_timesteps     = steps,
        callback            = callbacks,
        progress_bar        = True,
        reset_num_timesteps = reset_timesteps,
    )

    out = os.path.join(session_dir, "final")
    safe_save(model, out)
    return out


# ════════════════════════════════════════════════════════
#  主程式
# ════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="CarRacing-v3 PPO 最終版",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--resume", type=str, default=None, metavar="PATH",
        help="繼續訓練：模型路徑（有無 .zip 都可以）\n"
             "例：--resume models/best/best_model")
    parser.add_argument("--infinite", action="store_true",
        help="無限模式：跑完自動繼續，學習率自適應衰減")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS,
        help=f"每輪步數（預設 {TOTAL_STEPS:,}）")
    parser.add_argument("--lr", type=float, default=None,
        help="覆寫學習率（預設：從零 3e-4 / 繼續 5e-5）")
    parser.add_argument("--device", type=str, default="mps",
        help="訓練裝置：mps / cuda / cpu（自動 fallback）")
    parser.add_argument("--freeze-cnn", action="store_true",
        help="凍結 CNN 層，只訓練 Actor/Critic 頭（速度快 3~4x）")
    args = parser.parse_args()

    # ── [修正12] 啟動前防呆檢查 ───────────────────────
    errors = []
    if args.steps < 1000:
        errors.append(f"--steps {args.steps} 太少，至少需要 1000")
    if args.lr is not None and not (1e-7 <= args.lr <= 1.0):
        errors.append(f"--lr {args.lr} 不合理，請設在 1e-7 ~ 1.0 之間")
    if errors:
        for e in errors:
            print(f"  ❌ {e}")
        sys.exit(1)

    # ── 建立必要資料夾 ─────────────────────────────────
    os.makedirs(LOG_DIR,   exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── [修正10] 裝置自動 fallback ────────────────────
    device = resolve_device(args.device)

    # ── 學習率 ─────────────────────────────────────────
    if args.lr is not None:
        base_lr = args.lr
    elif args.resume:
        base_lr = 5e-5
    else:
        base_lr = 3e-4

    # ── [修正11] 驗證 resume 路徑 ─────────────────────
    resume_path = verify_model_path(args.resume) if args.resume else None

    # ── [修正9] Session 號碼偵測 ──────────────────────
    session_start = detect_session_start(MODEL_DIR, bool(resume_path))

    # ── 啟動資訊 ───────────────────────────────────────
    print("\n" + "═" * 60)
    print("  🏎  CarRacing-v3 PPO 最終版")
    print("─" * 60)
    print(f"  模式         : {'無限' if args.infinite else '單輪'}")
    src = f"繼續 ({resume_path})" if resume_path else "從零開始"
    print(f"  起點         : {src}")
    print(f"  每輪步數     : {args.steps:,}")
    print(f"  起始學習率   : {base_lr:.2e}")
    print(f"  訓練裝置     : {device}")
    print(f"  凍結 CNN     : {'是' if args.freeze_cnn else '否'}")
    print(f"  Session 起點 : {session_start}")
    print("═" * 60 + "\n")

    # ── 建立環境 ───────────────────────────────────────
    train_env = build_vec_env(N_ENVS, session_start)
    eval_env  = build_vec_env(1,      session_start)

    # ── 建立 / 載入模型 ────────────────────────────────
    if resume_path:
        print(f"  載入模型：{resume_path}.zip")
        model = PPO.load(resume_path, env=train_env, device=device)

        # [修正1] 同步 lr
        _lr_holder.value = base_lr
        set_lr(model, base_lr)

        # resume 保守設定
        model.clip_range = lambda _: 0.10
        model.ent_coef   = 0.0005
        # [修正2] 不動 n_steps

        if args.freeze_cnn:
            frozen = 0
            for p in model.policy.features_extractor.parameters():
                p.requires_grad = False
                frozen += p.numel()
            print(f"  CNN 已凍結（{frozen:,} 個參數），只訓練 Actor/Critic 頭")

        reset_ts = False

    else:
        model = PPO(
            env             = train_env,
            policy          = "CnnPolicy",
            n_steps         = 512,
            batch_size      = 128,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.01,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            learning_rate   = _lr_holder,          # [修正1] 用 LRHolder
            verbose         = 1,
            tensorboard_log = LOG_DIR,
            seed            = SEED,
            device          = device,
        )
        reset_ts = True

    # ── 訓練迴圈 ───────────────────────────────────────
    session = session_start
    lr      = base_lr

    try:
        while True:
            print(f"\n  ▶  Session {session}  lr={lr:.2e}")
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
            print(f"\n  ♻  無限模式 → Session {session}，新學習率 {lr:.2e}")

    except KeyboardInterrupt:
        # [修正7] Ctrl+C 安全儲存
        print("\n\n  ⏹  中斷！儲存中...")
        safe_save(model, os.path.join(MODEL_DIR, "interrupted"))

    except Exception as e:
        # [修正7] 未知錯誤緊急儲存
        print(f"\n\n  ❌ 未預期錯誤：{type(e).__name__}: {e}")
        print("  嘗試緊急儲存...")
        safe_save(model, os.path.join(MODEL_DIR, "emergency"))
        raise   # 重新拋出讓 traceback 可見

    finally:
        for env in [train_env, eval_env]:
            try:
                env.close()
            except Exception:
                pass
        print("  環境已關閉。\n")


if __name__ == "__main__":
    main()