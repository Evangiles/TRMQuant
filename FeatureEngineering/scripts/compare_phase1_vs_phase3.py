"""
Compare Phase 1 vs Phase 3 performance.

Phase 1: 394 features (baseline + 12 core)
Phase 3: 59 features (after PCA + stability filter)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.models.lightgbm_model import cross_validate_lightgbm
import json


def main():
    print('='*80)
    print('Comparing Phase 1 vs Phase 3 (denoised data)')
    print('='*80)

    # Phase 1
    print('\n--- Phase 1: 394 features ---')
    df1 = pd.read_csv('data/features/train_phase1_denoised.csv')
    print(f'Shape: {df1.shape}')
    model1, cv1 = cross_validate_lightgbm(df1, n_splits=5, embargo=21)

    # Phase 3
    print('\n--- Phase 3: 59 features ---')
    df3 = pd.read_csv('data/features/train_phase3_denoised.csv')
    print(f'Shape: {df3.shape}')
    model3, cv3 = cross_validate_lightgbm(df3, n_splits=5, embargo=21)

    # Compare
    print('\n' + '='*80)
    print('COMPARISON')
    print('='*80)

    print(f'\nAdjusted Sharpe:')
    p1_sharpe = cv1['adjusted_sharpe'].mean()
    p3_sharpe = cv3['adjusted_sharpe'].mean()
    print(f'  Phase 1: {p1_sharpe:.4f} ± {cv1["adjusted_sharpe"].std():.4f}')
    print(f'  Phase 3: {p3_sharpe:.4f} ± {cv3["adjusted_sharpe"].std():.4f}')
    print(f'  Change: {(p3_sharpe - p1_sharpe) / p1_sharpe * 100:+.1f}%')

    print(f'\nIC:')
    p1_ic = cv1['ic'].mean()
    p3_ic = cv3['ic'].mean()
    print(f'  Phase 1: {p1_ic:.4f} ± {cv1["ic"].std():.4f}')
    print(f'  Phase 3: {p3_ic:.4f} ± {cv3["ic"].std():.4f}')
    print(f'  Change: {(p3_ic - p1_ic) / p1_ic * 100:+.1f}%')

    print(f'\nMax Drawdown:')
    p1_dd = cv1['max_drawdown'].mean()
    p3_dd = cv3['max_drawdown'].mean()
    print(f'  Phase 1: {p1_dd:.4f} ± {cv1["max_drawdown"].std():.4f}')
    print(f'  Phase 3: {p3_dd:.4f} ± {cv3["max_drawdown"].std():.4f}')
    print(f'  Change: {(p3_dd - p1_dd) / abs(p1_dd) * 100:+.1f}%')

    print(f'\nHit Rate:')
    p1_hr = cv1['hit_rate'].mean()
    p3_hr = cv3['hit_rate'].mean()
    print(f'  Phase 1: {p1_hr:.2%} ± {cv1["hit_rate"].std():.2%}')
    print(f'  Phase 3: {p3_hr:.2%} ± {cv3["hit_rate"].std():.2%}')
    print(f'  Change: {(p3_hr - p1_hr) / p1_hr * 100:+.1f}%')

    print(f'\nFeatures:')
    print(f'  Phase 1: 394 features')
    print(f'  Phase 3: 59 features (-85% reduction)')

    # Save
    results = {
        'phase1': {
            'adjusted_sharpe': float(p1_sharpe),
            'adjusted_sharpe_std': float(cv1['adjusted_sharpe'].std()),
            'ic': float(p1_ic),
            'ic_std': float(cv1['ic'].std()),
            'max_drawdown': float(p1_dd),
            'hit_rate': float(p1_hr),
            'n_features': 394
        },
        'phase3': {
            'adjusted_sharpe': float(p3_sharpe),
            'adjusted_sharpe_std': float(cv3['adjusted_sharpe'].std()),
            'ic': float(p3_ic),
            'ic_std': float(cv3['ic'].std()),
            'max_drawdown': float(p3_dd),
            'hit_rate': float(p3_hr),
            'n_features': 59
        }
    }

    os.makedirs('results', exist_ok=True)
    with open('results/phase1_vs_phase3.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n' + '='*80)
    print('Results saved to: results/phase1_vs_phase3.json')
    print('='*80)


if __name__ == "__main__":
    main()
