import os
import argparse
import random
import numpy as np
import pandas as pd
import yaml


def load_config(config_path='config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def add_sample_id_to_file(filepath, output_dir, crop=False):
    """Add unique sample_id column to CSV file and optionally remove zero rows"""
    fname = os.path.basename(filepath)
    df = pd.read_csv(filepath, header=None)

    if crop:
        # Remove rows where all feature columns are zero
        feature_df = df.iloc[:, 1:]
        is_zero_row = (feature_df == 0).all(axis=1)
        df = df[~is_zero_row].reset_index(drop=True)
        print(f"Cropped fully-zero rows: {fname} -> {df.shape[0]} rows remain")

    # Generate unique sample identifiers
    df.insert(0, 'sample_id', [f'{fname}_{i}' for i in range(df.shape[0])])
    output_path = os.path.join(output_dir, fname.replace('.csv', '_with_id.csv'))
    df.to_csv(output_path, index=False)
    print(f"Saved with sample_id: {output_path}")


def generate_with_id_files(config):
    """Generate preprocessed files with sample IDs for all data files"""
    raw_dir = config['paths']['data_raw']
    processed_dir = config['paths']['data_processed']
    os.makedirs(processed_dir, exist_ok=True)

    all_files = config['data_files']['train'] + config['data_files']['test']

    for fname in all_files:
        src_path = os.path.join(raw_dir, fname)
        if os.path.exists(src_path):
            # Apply cropping only to large training files
            crop_this = config['preprocessing']['crop_zero_rows'] and fname.endswith('_1000.csv')
            add_sample_id_to_file(src_path, processed_dir, crop=crop_this)
        else:
            print(f"Missing (skipped): {src_path}")


def create_train_test_split(config):
    """Create stratified train/test split for streaming experiments"""
    SEED = config['global']['seed']
    random.seed(SEED)
    np.random.seed(SEED)

    processed_dir = config['paths']['data_processed']
    stream_dir = config['paths']['data_stream']
    os.makedirs(stream_dir, exist_ok=True)

    # Define file-to-trajectory-to-group mappings
    file_mappings = [
        ('data_1_PAP1_1000_with_id.csv', 'PAP1', 'train'),
        ('data_1_PAP2_1000_with_id.csv', 'PAP2', 'train'),
        ('data_1_PAP3_1000_with_id.csv', 'PAP3', 'train'),
        ('data_1_PAP1_with_id.csv',      'PAP1', 'normal_test'),
        ('data_1_PAP2_with_id.csv',      'PAP2', 'normal_test'),
        ('data_1_PAP3_with_id.csv',      'PAP3', 'normal_test'),
        ('data_1_PAP1_bottle_with_id.csv','PAP1','bottle'),
        ('data_1_PAP2_bottle_with_id.csv','PAP2','bottle'),
        ('data_1_PAP3_bottle_with_id.csv','PAP3','bottle'),
        ('data_1_PAP1_random_with_id.csv','PAP1','random'),
        ('data_1_PAP3_random_with_id.csv','PAP3','random'),
        ('data_1_PAP3_acc_band_with_id.csv','PAP3','acc_band'),
    ]

    HOLD_OUT = config['preprocessing']['hold_out']
    SAMPLE_RATE = config['preprocessing']['sample_rate']
    SEED_SIZE = config['preprocessing']['seed_size']

    # Load sample metadata from processed files
    dfs_train, dfs_test = [], []
    missing = []

    for fname, traj, grp in file_mappings:
        path = os.path.join(processed_dir, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue

        # Load only sample identifiers for memory efficiency
        df = pd.read_csv(path, usecols=['sample_id'], dtype={'sample_id': str})
        df['trajectory'] = traj

        if grp == 'train':
            dfs_train.append(df)
        else:
            dfs_test.append(df)

    if missing:
        print("[WARN] Missing files (skipped):", ", ".join(missing))

    if len(dfs_train) == 0:
        raise RuntimeError("No training files loaded. Check *_1000_with_id.csv presence.")

    df_train = pd.concat(dfs_train, ignore_index=True)
    df_test = pd.concat(dfs_test, ignore_index=True) if dfs_test else pd.DataFrame(columns=['sample_id','trajectory'])

    # Create stratified seed set for initial model training
    seed_chunks = []
    for t in sorted(df_train['trajectory'].unique()):
        sub = df_train[df_train['trajectory'] == t]
        if sub.shape[0] == 0:
            continue
        take = min(SEED_SIZE, sub.shape[0])
        seed_chunks.append(sub.sample(n=take, random_state=SEED))

    if len(seed_chunks) == 0:
        raise RuntimeError("Could not build any seed chunk. Training pool is empty.")

    seed_df = pd.concat(seed_chunks, ignore_index=True)

    # Build streaming dataset from holdout + test data
    if HOLD_OUT:
        # Remove seed samples from training set to create holdout
        holdout = df_train.merge(
            seed_df[['sample_id']], on='sample_id', how='left', indicator=True
        )
        holdout = holdout[holdout['_merge'] == 'left_only'].drop(columns=['_merge']).reset_index(drop=True)
        stream_df = pd.concat([holdout, df_test], ignore_index=True)
    else:
        stream_df = df_test.copy()

    # Apply sampling rate if specified
    stream_df = stream_df.sample(frac=SAMPLE_RATE, random_state=SEED).reset_index(drop=True)

    # Save train/test split files
    seed_out = os.path.join(stream_dir, 'seed_df.csv')
    stream_out = os.path.join(stream_dir, 'stream_df.csv')
    seed_df.to_csv(seed_out, index=False)
    stream_df.to_csv(stream_out, index=False)

    # Display summary statistics
    print(f"[OK] Wrote: {seed_out}  (n={len(seed_df)})")
    print(seed_df['trajectory'].value_counts().rename('seed_count_by_traj'))
    print(f"[OK] Wrote: {stream_out} (n={len(stream_df)})")
    print(stream_df['trajectory'].value_counts().rename('stream_count_by_traj'))


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Data preparation for CBM experiments')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 50)
    print("Data Preparation")
    print("=" * 50)

    # Generate preprocessed files with sample IDs
    print("\n1. Generating preprocessed files...")
    generate_with_id_files(config)

    # Create stratified train/test splits
    print("\n2. Creating train/test splits...")
    create_train_test_split(config)

    print("\n" + "=" * 50)
    print("Data preparation completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()