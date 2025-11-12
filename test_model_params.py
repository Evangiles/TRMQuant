"""Test if model parameters change with different n_features"""
import sys
sys.path.append('TinyRecursiveModels')

from models.diffusion_mamba.denoiser import MultivariateMambaDenoiser

# Test different feature counts
feature_counts = [1, 3, 8, 14, 22, 40]

print("Testing model parameter counts:")
print("="*50)

for n_feat in feature_counts:
    model = MultivariateMambaDenoiser(
        n_features=n_feat,
        window_size=60,
        d_model=128,
        n_layers=4
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"n_features={n_feat:2d} → params={n_params:,}")

    # Calculate expected difference
    # Input proj: n_feat × 128
    # Output proj: 128 × n_feat
    input_params = n_feat * 128
    output_params = 128 * n_feat
    feature_params = input_params + output_params
    print(f"  Feature-dependent params: {feature_params:,}")
    print()
