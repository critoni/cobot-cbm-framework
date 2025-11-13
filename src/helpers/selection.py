"""
Uncertainty multiplier selection utilities for experiment optimization.
"""

import pandas as pd
import numpy as np


def calculate_ranking_metrics(df, eff_min, ecr_min):
    """Calculate normalized ranking metrics for uncertainty multiplier selection"""
    df = df.dropna(subset=['error_catch_rate', 'workload_efficiency'])

    qualified_mask = (df['workload_efficiency'] >= eff_min) & (df['error_catch_rate'] >= ecr_min)
    qualified_df = df[qualified_mask].copy()

    if qualified_df.empty:
        df['ErrRank'] = np.nan
        df['EffRank'] = np.nan
        return df, qualified_df

    ecr_max = qualified_df['error_catch_rate'].max()
    ecr_min_val = qualified_df['error_catch_rate'].min()
    we_max = qualified_df['workload_efficiency'].max()
    we_min_val = qualified_df['workload_efficiency'].min()

    ecr_range = ecr_max - ecr_min_val
    we_range = we_max - we_min_val

    if ecr_range == 0:
        df['ErrRank'] = 0.0
    else:
        df['ErrRank'] = (ecr_max - df['error_catch_rate']) / ecr_range

    if we_range == 0:
        df['EffRank'] = 0.0
    else:
        df['EffRank'] = (we_max - df['workload_efficiency']) / we_range

    qualified_df['ErrRank'] = df.loc[qualified_mask, 'ErrRank']
    qualified_df['EffRank'] = df.loc[qualified_mask, 'EffRank']

    return df, qualified_df


def select_optimal_multipliers(qualified_df):
    """Select three optimal uncertainty multipliers based on ranking criteria"""
    if qualified_df.empty:
        return None, None, None

    min_err_rank = qualified_df['ErrRank'].min()
    best_err_candidates = qualified_df[qualified_df['ErrRank'] == min_err_rank]
    best_err = best_err_candidates.loc[best_err_candidates['EffRank'].idxmin()]

    min_eff_rank = qualified_df['EffRank'].min()
    best_eff_candidates = qualified_df[qualified_df['EffRank'] == min_eff_rank]
    best_eff = best_eff_candidates.loc[best_eff_candidates['ErrRank'].idxmin()]

    qualified_df = qualified_df.copy()
    qualified_df['rank_diff'] = (qualified_df['ErrRank'] - qualified_df['EffRank']).abs()
    min_diff = qualified_df['rank_diff'].min()
    balance_candidates = qualified_df[qualified_df['rank_diff'] == min_diff].copy()
    balance_candidates['rank_sum'] = balance_candidates['ErrRank'] + balance_candidates['EffRank']
    best_balance = balance_candidates.loc[balance_candidates['rank_sum'].idxmin()]

    return best_err, best_eff, best_balance


def get_multiplier_selections(df, seed_size, base_multiplier_focus, eff_min, ecr_min):
    """Get optimal multiplier selections for a specific training size"""
    seed_size_data = df[(df['seed_size'] == seed_size) &
                        (np.isclose(df['base_multiplier'], base_multiplier_focus))].copy()

    if seed_size_data.empty:
        return None

    all_data, qualified_data = calculate_ranking_metrics(seed_size_data, eff_min, ecr_min)
    best_err, best_eff, best_balance = select_optimal_multipliers(qualified_data)

    if best_err is None:
        return None

    return {
        'seed_size': int(seed_size),
        'all_data': all_data.sort_values('uncertainty_multiplier').reset_index(drop=True),
        'selections': {
            'best_err': {
                'uncertainty_multiplier': float(best_err['uncertainty_multiplier']),
                'err_rank': float(best_err['ErrRank']),
                'eff_rank': float(best_err['EffRank'])
            },
            'best_eff': {
                'uncertainty_multiplier': float(best_eff['uncertainty_multiplier']),
                'err_rank': float(best_eff['ErrRank']),
                'eff_rank': float(best_eff['EffRank'])
            },
            'best_balance': {
                'uncertainty_multiplier': float(best_balance['uncertainty_multiplier']),
                'err_rank': float(best_balance['ErrRank']),
                'eff_rank': float(best_balance['EffRank'])
            }
        }
    }


def load_selected_multiplier(csv_path, seed_size, selection_strategy):
    """Load selected uncertainty multiplier from experiment 2 CSV results"""
    df = pd.read_csv(csv_path)
    row = df[df['seed_size'] == seed_size]

    if row.empty:
        raise ValueError(f"No data found for seed_size={seed_size} in {csv_path}")

    row = row.iloc[0]

    strategy_mapping = {
        'best_err': 'best_err_um',
        'best_eff': 'best_eff_um',
        'best_balance': 'best_balance_um'
    }

    if selection_strategy not in strategy_mapping:
        raise ValueError(f"Invalid selection_strategy: {selection_strategy}. Must be one of {list(strategy_mapping.keys())}")

    column = strategy_mapping[selection_strategy]
    return float(row[column])