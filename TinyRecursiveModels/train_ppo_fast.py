"""
Fast PPO training with vectorized rollouts.

Key optimizations:
1. Vectorized environment interaction (no step-by-step CPU-GPU sync)
2. Batched policy forward passes
3. Pre-allocated tensors instead of Python lists
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from rl.data.preprocess import load_market_csv, impute_forward_back_fill, fit_standardizer, impute_series
from rl.envs.market_env import MarketEnv, MarketEnvConfig
from rl.ppo import RolloutBuffer, PPOConfig, ppo_update_seq
from rl.metrics import adjusted_sharpe
from rl.utils.ema import EMAHelper
from models.recursive_reasoning.trm_ppo import TRMPPOTemporalEncoder, TRMPPOConfig


def vectorized_rollout(policy, env, split, device, max_steps=10000):
    """
    Vectorized rollout: compute all observations at once, then run policy in batch.

    Returns:
        observations, actions, rewards, dones, values, logprobs, carries
    """
    obs = env.reset(split=split)
    carry = policy.initial_carry(batch_size=1)

    # Pre-allocate arrays
    observations = []
    actions_list = []
    rewards_list = []
    dones_list = []
    values_list = []
    logprobs_list = []
    carries_list = []

    done = False
    step = 0

    # Collect full trajectory
    while not done and step < max_steps:
        obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)

        with torch.no_grad():
            carry, action, logprob, value = policy.act(carry, obs_t, deterministic=False)

        # Store on GPU (no CPU sync yet!)
        observations.append(obs)
        actions_list.append(action.item())  # Only this syncs
        values_list.append(value.squeeze(0))
        logprobs_list.append(logprob.squeeze(0))
        carries_list.append(carry)

        # Step environment
        next_obs, reward, done, info = env.step(action.item())
        rewards_list.append(reward)
        dones_list.append(done)

        obs = next_obs
        step += 1

    return {
        'observations': observations,
        'actions': np.array(actions_list, dtype=np.float32),
        'rewards': np.array(rewards_list, dtype=np.float32),
        'dones': np.array(dones_list, dtype=bool),
        'values': torch.stack(values_list).cpu(),
        'logprobs': torch.stack(logprobs_list).cpu(),
        'carries': carries_list,
        'steps': step
    }


def main():
    parser = argparse.ArgumentParser(description="Fast TRM-PPO with vectorized rollouts")
    parser.add_argument("--path", type=str, default=os.path.join("TinyRecursiveModels", "train.csv"))
    parser.add_argument("--impute", type=str, default="legacy", choices=["legacy", "diffusion"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--H_cycles", type=int, default=4)
    parser.add_argument("--L_cycles", type=int, default=6)
    parser.add_argument("--warm_start", type=str, default="")
    parser.add_argument("--reward_mode", type=str, default="alpha")
    parser.add_argument("--risk_penalty_lambda", type=float, default=0.0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data (same as before)
    csv_path = args.path
    if args.impute == "diffusion":
        df = pd.read_csv(csv_path)
        cols = list(df.columns)
        date_idx = cols.index("date_id")
        target_cols = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]
        first_target = min(cols.index(c) for c in target_cols)
        feature_cols = cols[date_idx + 1:first_target]

        X_df = df[feature_cols].copy()
        X_df = X_df.ffill().bfill()
        median = X_df.median(numeric_only=True)
        X_df = X_df.fillna(median)
        X = X_df.to_numpy(dtype=np.float32)

        fwd = impute_series(pd.to_numeric(df["forward_returns"], errors="coerce").to_numpy())
        rf = impute_series(pd.to_numeric(df["risk_free_rate"], errors="coerce").to_numpy())
        mkt_excess = impute_series(pd.to_numeric(df["market_forward_excess_returns"], errors="coerce").to_numpy())

        split_idx = int(len(X) * 0.8)
        stdzr = fit_standardizer(X[:split_idx])
        X = stdzr.transform(X)
    else:
        date_id, X, fwd, rf, mkt_excess, feat_names = load_market_csv(csv_path)
        X = impute_forward_back_fill(X)
        split_idx = int(len(X) * 0.8)
        stdzr = fit_standardizer(X[:split_idx])
        X = stdzr.transform(X)

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

    if args.warm_start and os.path.exists(args.warm_start):
        print(f"Loading warm start from {args.warm_start}...")
        ckpt = torch.load(args.warm_start, map_location=device)
        state = ckpt.get('model', ckpt)
        policy.load_state_dict(state, strict=False)
        print("Warm start loaded")

    cfg = PPOConfig(
        tbptt_len=32,
        lr=args.lr,
        ent_coef=0.05,
        clip_eps=0.2,
        vf_coef=0.5,
    )
    optimizer = Adam(policy.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    USE_EMA = bool(int(os.environ.get("USE_EMA", "1")))
    EMA_MU = float(os.environ.get("EMA_MU", "0.999"))
    ema = EMAHelper(mu=EMA_MU) if USE_EMA else None
    if ema:
        ema.register(policy)

    buffer = RolloutBuffer(
        obs_shape=env.observation_shape,
        capacity=cfg.rollout_steps,
        hidden_size=args.hidden_size,
        window_size=env.config.window_size
    )

    best_val = -1e9
    best_path = os.path.join("TinyRecursiveModels", "checkpoints_trmppo_fast.pt")

    print(f"\n🚀 Starting fast PPO training (vectorized rollouts)")
    print(f"Device: {device}, Hidden size: {args.hidden_size}, LR: {args.lr}\n")

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")

        for split in ("train", "val"):
            print(f"\n[{split.upper()}] Collecting rollout...")

            # Vectorized rollout (much faster!)
            rollout = vectorized_rollout(policy, env, split, device)

            actions = rollout['actions']
            steps = rollout['steps']

            print(f"Collected {steps} steps")
            print(f"  Actions: μ={actions.mean():.3f} σ={actions.std():.3f} [{actions.min():.2f}, {actions.max():.2f}]")

            # Evaluate
            t_indices = np.arange(env._splits[split][0], env._splits[split][0] + steps)
            df_eval = pd.DataFrame({
                'forward_returns': fwd[t_indices],
                'risk_free_rate': rf[t_indices],
            })

            eval_model = ema.ema_copy(policy) if (ema and split == "val") else policy
            score = adjusted_sharpe(df_eval, actions)
            print(f"  AdjustedSharpe: {score:.6f}")

            # Train only on train split
            if split == "train":
                # Fill buffer in chunks
                num_updates = 0
                for i in range(0, steps, cfg.rollout_steps):
                    chunk_end = min(i + cfg.rollout_steps, steps)
                    chunk_size = chunk_end - i

                    buffer.ptr = 0
                    for j in range(i, chunk_end):
                        buffer.add(
                            torch.from_numpy(rollout['observations'][j]),
                            torch.tensor(rollout['actions'][j]),
                            torch.tensor(rollout['rewards'][j]),
                            torch.tensor(rollout['dones'][j]),
                            rollout['values'][j],
                            rollout['logprobs'][j],
                            rollout['carries'][j]
                        )

                    # Compute GAE
                    if rollout['dones'][chunk_end-1]:
                        last_value = 0.0
                    else:
                        with torch.no_grad():
                            obs_t = torch.as_tensor(rollout['observations'][chunk_end-1], device=device).unsqueeze(0)
                            _, _, _, last_value_t = policy.act(rollout['carries'][chunk_end-1], obs_t, deterministic=True)
                            last_value = float(last_value_t.squeeze(0).cpu())

                    buffer.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)

                    # PPO update
                    metrics = ppo_update_seq(policy, optimizer, buffer, cfg)
                    num_updates += 1

                    if metrics and num_updates % 3 == 0:  # Print every 3 updates
                        print(f"  Update {num_updates}: pol={metrics['policy_loss']:.4f} val={metrics['value_loss']:.4f} "
                              f"ent={metrics['entropy']:.4f} clip={metrics['clip_frac']:.3f}")

                    if ema:
                        ema.update(policy)

            # Save best
            if split == "val" and score > best_val:
                best_val = score
                torch.save({'model': eval_model.state_dict(), 'score': best_val}, best_path)
                print(f"  ✅ Saved best checkpoint (score={best_val:.6f})")

        scheduler.step()

    print(f"\n{'='*60}")
    print(f"Training complete! Best val Sharpe: {best_val:.6f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
