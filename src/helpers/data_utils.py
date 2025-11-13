"""
Data loading and preprocessing utilities for HIL-CBM framework.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_stream_data(stream_dir='dataframe_stream'):
    """Load seed and stream dataframes from processed files"""
    seed_df = pd.read_csv(os.path.join(stream_dir, 'seed_df.csv'), dtype={'sample_id': str})
    stream_df = pd.read_csv(os.path.join(stream_dir, 'stream_df.csv'), dtype={'sample_id': str})

    # Filter out PAP4 trajectory (not used in experiments)
    stream_df = stream_df[stream_df['trajectory'] != 'PAP4'].reset_index(drop=True)

    return seed_df, stream_df


def load_raw_cycles(data_dir, n_timesteps=204, n_variables=21):
    """Load raw cycle data from processed CSV files with sample IDs"""
    raw_data = {}

    for fn in tqdm(os.listdir(data_dir), desc="Loading raw data"):
        if fn.endswith('_with_id.csv'):
            df = pd.read_csv(os.path.join(data_dir, fn), dtype={'sample_id': str})

            # Reshape data: remove sample_id column, reshape to (samples, timesteps, variables)
            # Drop first variable column (index) by taking [:, :, 1:]
            arr = df.drop('sample_id', axis=1).values.reshape(-1, n_timesteps, n_variables + 1)[:, :, 1:]

            # Store each sample cycle by sample_id
            for sid, cyc in zip(df['sample_id'], arr):
                raw_data[sid] = cyc

    return raw_data


def create_true_labels(stream_df):
    """Create true labels based on sample_id naming conventions"""
    stream_df['true_label'] = stream_df['sample_id'].apply(
        lambda s: 'unhealthy' if any(x in s.lower()
                                   for x in ['bottle', 'random', 'acc_band', 'overload', 'friction'])
                 else 'healthy'
    )
    return stream_df


def get_initial_training_data(seed_df, raw_data, seed_size):
    """Extract initial training data for each trajectory"""
    initial_training_data = {}

    for traj in ['PAP1', 'PAP2', 'PAP3']:
        # Get first 'seed_size' samples for this trajectory
        traj_samples = seed_df[seed_df['trajectory'] == traj]['sample_id'].iloc[:seed_size].tolist()
        initial_training_data[traj] = [raw_data[sid] for sid in traj_samples]

    return initial_training_data


def get_batch_data(stream_df, raw_data, batch_idx, batch_size=100):
    """Get data for a specific batch"""
    batch = stream_df.iloc[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    batch_cycles = [raw_data[sid] for sid in batch['sample_id']]
    batch_trajectories = batch['trajectory'].tolist()
    true_labels = batch['true_label'].tolist()

    return batch_cycles, batch_trajectories, true_labels


def validate_data_availability(seed_df, stream_df, raw_data):
    """Validate that all required cycles are available in raw_data"""
    needed = set(seed_df['sample_id']).union(stream_df['sample_id'])
    missing = needed - set(raw_data.keys())

    if missing:
        raise ValueError(f"Missing raw cycles: {missing}")

    print(f"Data validation passed: {len(raw_data)} cycles loaded")


def group_batch_by_trajectory(batch_cycles, batch_trajectories):
    """Group batch data by trajectory for parallel processing"""
    trajectory_groups = {}

    for i, (cycle, traj) in enumerate(zip(batch_cycles, batch_trajectories)):
        if traj not in trajectory_groups:
            trajectory_groups[traj] = {'indices': [], 'cycles': []}
        trajectory_groups[traj]['indices'].append(i)
        trajectory_groups[traj]['cycles'].append(cycle)

    return trajectory_groups


def prepare_experiment_data(config, experiment_name=None):
    """Prepare data for experiments with consistent loading and validation"""
    # Load stream data
    seed_df, stream_df = load_stream_data(config['paths']['data_stream'])

    # Load raw cycles
    raw_data = load_raw_cycles(
        config['paths']['data_processed'],
        config['global']['n_timesteps'],
        config['global']['n_variables']
    )

    # Create true labels
    stream_df = create_true_labels(stream_df)

    # Validate data availability
    validate_data_availability(seed_df, stream_df, raw_data)

    return seed_df, stream_df, raw_data
