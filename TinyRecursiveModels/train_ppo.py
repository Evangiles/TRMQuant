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
from rl.envs.market_env import MarketEnv, VectorizedMarketEnv, MarketEnvConfig
from rl.ppo import RolloutBuffer, PPOConfig, ppo_update_seq
from rl.metrics import adjusted_sharpe
from rl.utils.ema import EMAHelper
from models.recursive_reasoning.trm_ppo import TRMPPOTemporalEncoder, TRMPPOConfig, TRMPPOCarry


def main():
    parser = argparse.ArgumentParser(description="Train TRM-PPO with configurable preprocessing")
    parser.add_argument("--path", type=str, default=os.path.join("TinyRecursiveModels", "train_denoised.csv"))
    parser.add_argument("--impute", type=str, default="legacy", choices=["legacy", "diffusion"],
                        help="legacy: forward+back fill | diffusion: ffill+bfill+median (same as diffusion model)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size for TRM model")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--H_cycles", type=int, default=4, help="Number of H cycles")
    parser.add_argument("--L_cycles", type=int, default=6, help="Number of L cycles")
    parser.add_argument("--warm_start", type=str, default="", help="Path to supervised checkpoint for warm start")
    parser.add_argument("--reward_mode", type=str, default="alpha", choices=["alpha", "sharpe"], help="Reward function mode")
    parser.add_argument("--risk_penalty_lambda", type=float, default=0.0, help="Risk penalty weight")
    parser.add_argument("--tbptt_len", type=int, default=32, help="TBPTT sequence length (higher = better GPU util, more VRAM)")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments (>1 for vectorized rollout, 5-10x speedup)")
    parser.add_argument("--rollout_steps", type=int, default=1024, help="Steps per PPO update per environment (e.g., 256, 512, 1024)")
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

    # Create vectorized or single environment
    env_config = MarketEnvConfig(
        window_size=60,
        train_fraction=0.8,
        reward_mode=args.reward_mode,
        risk_penalty_lambda=args.risk_penalty_lambda,
        risk_cap_ratio=1.2,
        vol_window=60,
    )

    if args.num_envs > 1:
        env = VectorizedMarketEnv(
            num_envs=args.num_envs,
            features=X,
            market_excess_returns=mkt_excess,
            config=env_config,
            forward_returns=fwd,
            risk_free_rate=rf,
        )
        print(f"Using vectorized environment with {args.num_envs} parallel envs")
    else:
        env = MarketEnv(
            features=X,
            market_excess_returns=mkt_excess,
            config=env_config,
            forward_returns=fwd,
            risk_free_rate=rf,
        )
        print("Using single environment (consider --num_envs 8 for 5-10x speedup)")

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
        tbptt_len=args.tbptt_len,
        lr=args.lr,
        ent_coef=0.001,  # Reduced from 0.05 to preserve pretrain behavior
        clip_eps=0.1,    # Reduced from 0.2 for more conservative updates
        vf_coef=0.5,
        update_epochs=3,  # Reduced from 6 to avoid over-fitting
        rollout_steps=args.rollout_steps,
    )
    optimizer = Adam(policy.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # EMA toggles via env vars or constants
    USE_EMA = bool(int(os.environ.get("USE_EMA", "1")))
    EMA_MU = float(os.environ.get("EMA_MU", "0.999"))
    ema = EMAHelper(mu=EMA_MU) if USE_EMA else None
    if ema is not None:
        ema.register(policy)
    # Buffer capacity scales with num_envs: update after (rollout_steps * num_envs) transitions
    buffer_capacity = args.rollout_steps * args.num_envs if args.num_envs > 1 else args.rollout_steps

    buffer = RolloutBuffer(
        obs_shape=env.observation_shape,
        capacity=buffer_capacity,
        hidden_size=args.hidden_size,
        window_size=env.config.window_size
    )

    # training loop
    best_val = -1e9
    best_path = os.path.join("TinyRecursiveModels", "checkpoints_trmppo.pt")

    epoch_pbar = tqdm(total=args.epochs, desc="Epochs", leave=True, dynamic_ncols=True, mininterval=0.2)
    for epoch in range(args.epochs):
        for split in ("train", "val"):
            # Switch train/eval mode
            if split == "train":
                policy.train()
            else:
                policy.eval()

            obs = env.reset(split=split)  # [N, window_size, F] if vectorized else [window_size, F]

            # Handle vectorized vs single environment
            if args.num_envs > 1:
                done = np.zeros(args.num_envs, dtype=bool)
                carry = policy.initial_carry(batch_size=args.num_envs)
            else:
                done = False
                carry = policy.initial_carry(batch_size=1)
                obs = obs[np.newaxis, ...]  # Add batch dim: [1, window_size, F]

            step = 0
            allocations = []
            fwd_list = []
            rf_list = []
            buffer.clear()  # Properly reset buffer including episode counter

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

            # Track processed samples to avoid overshooting dataset steps
            processed = 0
            if args.num_envs > 1:
                finished_mask = np.zeros(args.num_envs, dtype=bool)

            # Use EMA model for validation actions
            actor_for_eval = ema.ema_copy(policy) if (ema and split == "val") else policy
            if split == "val":
                actor_for_eval.eval()

            while True:
                # obs is now always [N, window_size, F]
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)

                with torch.no_grad():
                    carry, action, logprob, value = actor_for_eval.act(carry, obs_t, deterministic=(split == "val"))

                # Convert actions to numpy
                actions_np = action.cpu().numpy()  # [N]

                # Step environment(s)
                if args.num_envs > 1:
                    # Mask out finished envs to avoid counting/reset loops
                    active_mask = ~finished_mask
                    if not active_mask.any():
                        break

                    # Zero out actions for finished envs to keep API shape
                    actions_np_masked = actions_np.copy()
                    actions_np_masked[~active_mask] = 0.0
                    next_obs, rewards, dones, infos = env.step(actions_np_masked)

                    # Metrics collection (light loop; optional)
                    active_idx = np.where(active_mask)[0]
                    for i in active_idx:
                        allocations.append(float(actions_np[i]))
                        t_idx = env.envs[i]._t - 1
                        fwd_list.append(float(fwd[t_idx]))
                        rf_list.append(float(rf[t_idx]))

                    # Vectorized buffer add during training
                    if split == "train":
                        obs_b = torch.from_numpy(obs[active_mask])
                        act_b = torch.from_numpy(actions_np[active_mask]).to(torch.float32)
                        rew_b = torch.from_numpy(rewards[active_mask]).to(torch.float32)
                        done_b = torch.from_numpy(dones[active_mask])
                        val_b = value[active_mask].cpu()
                        logp_b = logprob[active_mask].cpu()
                        carry_b = TRMPPOCarry(
                            z_H=carry.z_H[active_mask],
                            z_L=carry.z_L[active_mask]
                        )
                        buffer.add_batch(obs_b, act_b, rew_b, done_b, val_b, logp_b, carry_b)

                    # Update finished mask and advance only active envs' obs
                    finished_mask = finished_mask | dones
                    obs[active_mask] = next_obs[active_mask]

                    # Progress accounting: count only active envs this step
                    processed += int(active_mask.sum())
                else:
                    # Single environment (backward compatibility)
                    a = float(actions_np[0])
                    next_obs, reward, done_single, info = env.step(a)

                    allocations.append(a)
                    t_idx = env._t - 1
                    fwd_list.append(float(fwd[t_idx]))
                    rf_list.append(float(rf[t_idx]))

                    if split == "train":
                        obs_b = torch.from_numpy(obs)              # [1, T, F]
                        act_b = torch.tensor([a], dtype=torch.float32)
                        rew_b = torch.tensor([reward], dtype=torch.float32)
                        done_b = torch.tensor([done_single], dtype=torch.bool)
                        val_b = value.cpu()
                        logp_b = logprob.cpu()
                        carry_b = TRMPPOCarry(z_H=carry.z_H, z_L=carry.z_L)
                        buffer.add_batch(obs_b, act_b, rew_b, done_b, val_b, logp_b, carry_b)

                    done = done_single
                    obs = next_obs[np.newaxis, ...]  # Keep batch dim
                    processed += 1

                step += 1
                # Update progress bar
                inc = int(active_mask.sum()) if args.num_envs > 1 else 1
                split_pbar.update(inc)

                # Stop when we've consumed the split's steps
                if total_steps_split and processed >= total_steps_split:
                    break

                # PPO update (only during training)
                # Update every 1024 steps regardless of num_envs
                if split == "train" and (buffer.is_full() or (done.all() if args.num_envs > 1 else done)):
                    # Compute GAE with episode boundaries (no manual bootstrapping needed)
                    buffer.compute_gae(gamma=cfg.gamma, lam=cfg.gae_lambda)
                    metrics = ppo_update_seq(policy, optimizer, buffer, cfg)

                    if metrics:
                        print(f"PPO: pol_loss={metrics['policy_loss']:.4f} val_loss={metrics['value_loss']:.4f} ent={metrics['entropy']:.4f} clip={metrics['clip_frac']:.3f}")

                    if ema is not None:
                        ema.update(policy)

                    buffer.clear()
                    step = 0

            split_pbar.close()
            # Evaluate adjusted sharpe at split end
            try:
                df_eval = pd.DataFrame({
                    'forward_returns': np.array(fwd_list, dtype=np.float64),
                    'risk_free_rate': np.array(rf_list, dtype=np.float64),
                })
                # allocations already generated by actor_for_eval (EMA model for val)
                score = adjusted_sharpe(df_eval, np.array(allocations, dtype=np.float64))
                # Save the model that generated these allocations
                eval_model = actor_for_eval
                print(f"Epoch {epoch+1} Split {split} steps={len(allocations)} AdjustedSharpe={score:.6f}")
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


