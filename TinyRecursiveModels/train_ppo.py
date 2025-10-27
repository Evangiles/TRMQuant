import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from rl.data.preprocess import load_market_csv, impute_forward_back_fill, fit_standardizer, impute_series
from rl.envs.market_env import MarketEnv, MarketEnvConfig
from rl.ppo import RolloutBuffer, PPOConfig, ppo_update_seq
from rl.metrics import adjusted_sharpe
from rl.utils.ema import EMAHelper
from models.recursive_reasoning.trm_ppo import TRMPPOTemporalEncoder, TRMPPOConfig


def main():
    parser = argparse.ArgumentParser(description="Train TRM-PPO with configurable preprocessing")
    parser.add_argument("--path", type=str, default=os.path.join("TinyRecursiveModels", "train.csv"))
    parser.add_argument("--impute", type=str, default="legacy", choices=["legacy", "diffusion"],
                        help="legacy: forward+back fill | diffusion: ffill+bfill+median (same as diffusion model)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size for TRM model")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--H_cycles", type=int, default=3, help="Number of H cycles")
    parser.add_argument("--L_cycles", type=int, default=4, help="Number of L cycles")
    parser.add_argument("--warm_start", type=str, default="", help="Path to supervised checkpoint for warm start")
    parser.add_argument("--reward_mode", type=str, default="alpha", choices=["alpha", "sharpe"], help="Reward function mode")
    parser.add_argument("--risk_penalty_lambda", type=float, default=0.0, help="Risk penalty weight")
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    csv_path = args.path

    if args.impute == "diffusion":
        # Use same preprocessing as diffusion model
        df = pd.read_csv(csv_path)
        cols = list(df.columns)
        date_idx = cols.index("date_id")
        target_cols = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]
        first_target = min(cols.index(c) for c in target_cols)
        feature_cols = cols[date_idx + 1:first_target]

        date_id = df["date_id"].to_numpy()
        X_df = df[feature_cols].copy()
        X_df = X_df.ffill().bfill()
        median = X_df.median(numeric_only=True)
        X_df = X_df.fillna(median)
        X = X_df.to_numpy(dtype=np.float32)
        feat_names = feature_cols

        fwd = impute_series(pd.to_numeric(df["forward_returns"], errors="coerce").to_numpy())
        rf = impute_series(pd.to_numeric(df["risk_free_rate"], errors="coerce").to_numpy())
        mkt_excess = impute_series(pd.to_numeric(df["market_forward_excess_returns"], errors="coerce").to_numpy())

        # standardize on train split only
        split_idx = int(len(X) * 0.8)
        stdzr = fit_standardizer(X[:split_idx])
        X = stdzr.transform(X)
    else:
        # Legacy preprocessing
        date_id, X, fwd, rf, mkt_excess, feat_names = load_market_csv(csv_path)
        X = impute_forward_back_fill(X)
        split_idx = int(len(X) * 0.8)
        stdzr = fit_standardizer(X[:split_idx])
        X = stdzr.transform(X)

    # Impute targets, too, to avoid NaNs in reward/metric
    fwd = impute_series(fwd)
    rf = impute_series(rf)
    mkt_excess = impute_series(mkt_excess)

    env = MarketEnv(
        features=X,
        market_excess_returns=mkt_excess,
        config=MarketEnvConfig(
            window_size=60,
            train_fraction=0.8,
            reward_mode=args.reward_mode,
            risk_penalty_lambda=args.risk_penalty_lambda,
            risk_cap_ratio=1.2,
            vol_window=60,
        ),
        forward_returns=fwd,
        risk_free_rate=rf,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = TRMPPOTemporalEncoder(
        TRMPPOConfig(
            window_size=env.config.window_size,
            num_features=X.shape[1],
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            expansion=4.0,
            pos_encodings="rope",
            H_cycles=args.H_cycles,
            L_cycles=args.L_cycles,
            L_layers=1,
        )
    ).to(device)

    # Warm start from supervised checkpoint if provided
    if args.warm_start and os.path.exists(args.warm_start):
        print(f"Loading warm start checkpoint from {args.warm_start}...")
        ckpt = torch.load(args.warm_start, map_location=device)
        state = ckpt.get('model', ckpt)
        policy.load_state_dict(state, strict=False)
        print("Warm start loaded successfully")

    cfg = PPOConfig(
        tbptt_len=32,
        lr=args.lr,
        ent_coef=0.05,  # Increase entropy to prevent collapse
        clip_eps=0.2,
        vf_coef=0.5,
    )
    optimizer = Adam(policy.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # EMA toggles via env vars or constants
    USE_EMA = bool(int(os.environ.get("USE_EMA", "1")))
    EMA_MU = float(os.environ.get("EMA_MU", "0.999"))
    ema = EMAHelper(mu=EMA_MU) if USE_EMA else None
    if ema is not None:
        ema.register(policy)
    buffer = RolloutBuffer(obs_shape=env.observation_shape, capacity=cfg.rollout_steps)

    # training loop
    best_val = -1e9
    best_path = os.path.join("TinyRecursiveModels", "checkpoints_trmppo.pt")

    epoch_pbar = tqdm(total=args.epochs, desc="Epochs", leave=True, dynamic_ncols=True, mininterval=0.2)
    for epoch in range(args.epochs):
        for split in ("train", "val"):
            obs = env.reset(split=split)
            done = False
            carry = policy.initial_carry(batch_size=1)

            step = 0
            allocations = []
            fwd_list = []
            rf_list = []
            buffer.ptr = 0
            # progress for current split
            try:
                total_steps_split = int(env._splits[split][1] - env._splits[split][0] + 1)
            except Exception:
                total_steps_split = 0
            split_pbar = tqdm(
                total=total_steps_split if total_steps_split>0 else None,
                desc=f"Epoch {epoch+1} {split}",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.1,
            )

            while not done:
                # avoid reallocating tensors each step
                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                with torch.no_grad():
                    carry, action, logprob, value = policy.act(carry, obs_t, deterministic=False)
                a = float(action.squeeze(0).cpu().numpy())

                next_obs, reward, done, info = env.step(a)
                allocations.append(a)
                # align fwd/rf at current t
                t_idx = env._t - 1
                fwd_list.append(float(fwd[t_idx]))
                rf_list.append(float(rf[t_idx]))

                # Only add to buffer during training
                if split == "train":
                    buffer.add(
                        torch.from_numpy(obs),
                        torch.tensor(a, dtype=torch.float32),
                        torch.tensor(reward, dtype=torch.float32),
                        torch.tensor(done, dtype=torch.bool),
                        value.squeeze(0).cpu(),
                        logprob.squeeze(0).cpu(),
                    )

                obs = next_obs
                step += 1
                split_pbar.update(1)

                if split == "train" and (buffer.ptr >= cfg.rollout_steps or done):
                    # bootstrap with last value
                    if done:
                        last_value = 0.0
                    else:
                        with torch.no_grad():
                            _, _, _, last_value_t = policy.act(carry, torch.as_tensor(obs, device=device).unsqueeze(0), deterministic=True)
                            last_value = float(last_value_t.squeeze(0).cpu())
                    buffer.compute_gae(last_value=last_value, gamma=cfg.gamma, lam=cfg.gae_lambda)
                    metrics = ppo_update_seq(policy, optimizer, buffer, cfg)
                    # print short training metrics summary every PPO update
                    if metrics:
                        print(f"PPO: pol_loss={metrics['policy_loss']:.4f} val_loss={metrics['value_loss']:.4f} ent={metrics['entropy']:.4f} clip={metrics['clip_frac']:.3f}")
                    if ema is not None:
                        ema.update(policy)
                    buffer.ptr = 0

            split_pbar.close()
            # Evaluate adjusted sharpe at split end
            try:
                df_eval = pd.DataFrame({
                    'forward_returns': np.array(fwd_list, dtype=np.float64),
                    'risk_free_rate': np.array(rf_list, dtype=np.float64),
                })
                # evaluate with EMA copy if enabled on val split
                eval_model = ema.ema_copy(policy) if (ema is not None and split == "val") else policy
                score = adjusted_sharpe(df_eval, np.array(allocations, dtype=np.float64))
                print(f"Epoch {epoch+1} Split {split} steps={step} AdjustedSharpe={score:.6f}")
                if split == "val" and score > best_val:
                    best_val = score
                    torch.save({'model': (eval_model.state_dict()), 'score': best_val}, best_path)
                    print(f"Saved best checkpoint to {best_path} (score={best_val:.6f})")
            except Exception as e:
                print(f"Metric computation failed ({split}): {e}")

        scheduler.step()
        epoch_pbar.update(1)
    epoch_pbar.close()


if __name__ == "__main__":
    main()


