"""
Experiment 2: Uncertainty Optimization Analysis

Optimizes uncertainty thresholds by sweeping base and uncertainty multipliers
to find optimal 3-zone classification parameters for different training data sizes.
"""

import os
import time
import argparse
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product

from helpers.data_utils import prepare_experiment_data, get_initial_training_data, get_batch_data
from helpers.models import setup_tensorflow_environment, train_trajectory_models, save_trained_models
from helpers.uncertainty import batch_inference_with_uncertainty, calculate_uncertainty_metrics
from helpers.metrics import calculate_performance_metrics
from helpers.visualization import create_experiment_2_heatmaps


def load_config(config_path='config.yaml'):
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_uncertainty_multipliers(config):
    """Generate uncertainty multiplier values from config range"""
    start = config['experiment_2']['uncertainty_multipliers']['start']
    end = config['experiment_2']['uncertainty_multipliers']['end']
    step = config['experiment_2']['uncertainty_multipliers']['step']

    return np.arange(start, end + step/2, step).round(3)


def compute_mae_cache(models_dict, thresholds_dict, stream_df, raw_data, config):
    """Compute MAE scores once per batch and cache results"""
    batch_mae_cache = {}

    print("Computing MAE scores...")
    for batch_idx in tqdm(range(config['global']['n_batches']), desc="Computing MAE"):
        batch_cycles, batch_trajectories, true_labels = get_batch_data(
            stream_df, raw_data, batch_idx, config['global']['batch_size']
        )

        batch_mae_cache[batch_idx] = {}

        # Group by trajectory for efficient batch processing
        from helpers.data_utils import group_batch_by_trajectory
        trajectory_groups = group_batch_by_trajectory(batch_cycles, batch_trajectories)

        for traj, group_data in trajectory_groups.items():
            if traj not in models_dict:
                continue

            traj_cycles = group_data['cycles']
            traj_indices = group_data['indices']

            # Get MAE scores using our batch inference
            from helpers.models import batch_inference, get_base_predictions
            traj_mae_scores = batch_inference(traj_cycles, [traj] * len(traj_cycles), models_dict, config)
            traj_base_predictions = get_base_predictions(traj_mae_scores, [traj] * len(traj_cycles), thresholds_dict)

            # Store in cache with original indices
            batch_mae_cache[batch_idx][traj] = {
                'mae_scores': traj_mae_scores,
                'base_thresholds': thresholds_dict[traj],
                'indices': traj_indices,
                'true_labels': [true_labels[i] for i in traj_indices],
                'base_predictions': traj_base_predictions,
                'trajectory': traj
            }

    return batch_mae_cache


def apply_threshold_configuration(batch_mae_cache, base_multiplier, uncertainty_multiplier, config):
    """Apply threshold configuration to cached MAE results"""
    from helpers.uncertainty import apply_uncertainty_thresholds, apply_priority_logic

    results = []

    for batch_idx in range(config['global']['n_batches']):
        if batch_idx not in batch_mae_cache:
            continue

        for traj, cache_data in batch_mae_cache[batch_idx].items():
            mae_scores = cache_data['mae_scores']
            base_thresholds = cache_data['base_thresholds']

            # Apply 3-zone classification
            zones = apply_uncertainty_thresholds(
                mae_scores, base_thresholds, base_multiplier, uncertainty_multiplier
            )

            # Apply priority logic
            uncertainty_predictions, uncertainty_flags = apply_priority_logic(zones)

            # Store results for each sample
            for i, (pred, flag) in enumerate(zip(uncertainty_predictions, uncertainty_flags)):
                original_idx = cache_data['indices'][i]
                results.append({
                    'batch': batch_idx,
                    'sample_idx': original_idx,
                    'true_label': cache_data['true_labels'][i],
                    'uncertainty_prediction': pred,
                    'base_ae_prediction': cache_data['base_predictions'][i],
                    'uncertain_flag': flag,
                    'base_multiplier': base_multiplier,
                    'uncertainty_multiplier': uncertainty_multiplier
                })

    return results


def calculate_configuration_metrics(results_df):
    """Calculate performance metrics for a specific configuration"""
    # Filter out uncertain predictions for accuracy calculation
    definitive_mask = results_df['uncertainty_prediction'] != 'uncertain'

    if definitive_mask.sum() == 0:
        return {
            'accuracy': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'human_workload': 100.0,
            'automation_rate': 0.0,
            'error_catch_rate': 0.0,
            'workload_efficiency': 0.0,
            'total_samples': len(results_df),
            'uncertain_samples': results_df['uncertain_flag'].sum(),
            'definitive_samples': 0
        }

    # Calculate metrics on definitive predictions
    definitive_true = results_df[definitive_mask]['true_label'].tolist()
    definitive_pred = results_df[definitive_mask]['uncertainty_prediction'].tolist()

    basic_metrics = calculate_performance_metrics(definitive_pred, definitive_true)

    # Calculate uncertainty-specific metrics
    uncertainty_metrics = calculate_uncertainty_metrics(
        results_df['uncertainty_prediction'].tolist(),
        results_df['uncertain_flag'].values,
        results_df['true_label'].tolist(),
        results_df['base_ae_prediction'].tolist()
    )

    return {
        'accuracy': basic_metrics['accuracy'],
        'f1_score': basic_metrics['f1_score'],
        'precision': basic_metrics['precision'],
        'recall': basic_metrics['recall'],
        'human_workload': uncertainty_metrics['human_workload'],
        'automation_rate': uncertainty_metrics['automation_rate'],
        'error_catch_rate': uncertainty_metrics['error_catch_rate'],
        'workload_efficiency': uncertainty_metrics['workload_efficiency'],
        'total_samples': len(results_df),
        'uncertain_samples': int(uncertainty_metrics['uncertain_samples']),
        'definitive_samples': int(uncertainty_metrics['definitive_samples'])
    }


def run_uncertainty_optimization_experiment(config):
    """Execute uncertainty optimization experiment"""
    print("=" * 60)
    print("EXPERIMENT 2: Uncertainty Optimization Analysis")
    print("=" * 60)

    start_time = time.time()

    setup_tensorflow_environment(config)

    # Load data
    print("Loading data...")
    seed_df, stream_df, raw_data = prepare_experiment_data(config, 'experiment_2')

    # Configuration parameters
    training_sizes = config['experiment_2']['training_sizes']
    base_multipliers = config['experiment_2']['base_multipliers']
    uncertainty_multipliers = generate_uncertainty_multipliers(config)

    print(f"Training sizes: {training_sizes}")
    print(f"Base multipliers: {base_multipliers}")
    print(f"Uncertainty multipliers: {len(uncertainty_multipliers)} values from {uncertainty_multipliers[0]} to {uncertainty_multipliers[-1]}")

    # Calculate total configurations
    total_configs = len(training_sizes) * len(base_multipliers) * len(uncertainty_multipliers)
    print(f"Total configurations to evaluate: {total_configs}")

    # Collect all results
    all_results = []
    summary_results = []

    # Progress tracking
    config_count = 0

    # Process each training size
    for seed_size in training_sizes:
        print(f"\n--- Processing training size: {seed_size} samples per trajectory ---")

        # Extract training data
        initial_training_data = get_initial_training_data(seed_df, raw_data, seed_size)

        # Check if we have enough data
        min_samples = min(len(initial_training_data[traj]) for traj in ['PAP1', 'PAP2', 'PAP3'])
        if min_samples < seed_size:
            print(f"Warning: Not enough training data for seed_size={seed_size}. Min available: {min_samples}")
            continue

        # Train models (or load existing ones)
        print(f"Training models with {seed_size} samples per trajectory...")
        models_dict, thresholds_dict, was_loaded = train_trajectory_models(
            initial_training_data, config, seed_size, skip_existing=True
        )

        # Save models if newly trained
        if not was_loaded:
            save_trained_models(models_dict, thresholds_dict, config, seed_size)

        # Compute MAE scores once and cache for reuse
        batch_mae_cache = compute_mae_cache(models_dict, thresholds_dict, stream_df, raw_data, config)

        # Test all uncertainty configurations using cached MAE scores
        print("Testing threshold configurations...")
        for base_mult, uncertainty_mult in tqdm(
            product(base_multipliers, uncertainty_multipliers),
            desc=f"Testing uncertainty configs (seed_size={seed_size})",
            total=len(base_multipliers) * len(uncertainty_multipliers)
        ):
            config_count += 1

            # Apply threshold configuration to cached MAE results
            config_results = apply_threshold_configuration(
                batch_mae_cache, base_mult, uncertainty_mult, config
            )

            # Add training size to results
            for result in config_results:
                result['seed_size'] = seed_size
                all_results.append(result)

            # Calculate metrics for this configuration
            config_df = pd.DataFrame(config_results)
            metrics = calculate_configuration_metrics(config_df)

            # Store summary
            summary_results.append({
                'seed_size': seed_size,
                'base_multiplier': base_mult,
                'uncertainty_multiplier': uncertainty_mult,
                **metrics
            })

            # Print progress for key configurations
            if config_count % 50 == 0 or (base_mult == 1.0 and uncertainty_mult == 1.05):
                print(f"  Config {config_count}/{total_configs}: "
                      f"base={base_mult:.3f}, unc={uncertainty_mult:.3f} -> "
                      f"F1={metrics['f1_score']:.1f}%, workload={metrics['human_workload']:.1f}%")

    # Save detailed results
    results_df = pd.DataFrame(all_results)
    output_dir = config['paths']['results']
    os.makedirs(os.path.join(output_dir, 'experiment_2'), exist_ok=True)

    detailed_path = os.path.join(output_dir, 'experiment_2', 'uncertainty_optimization_detailed.csv')
    results_df.to_csv(detailed_path, index=False)

    # Save summary results
    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(output_dir, 'experiment_2', 'uncertainty_optimization_summary.csv')
    summary_df.to_csv(summary_path, index=False)


    # Generate heatmap visualizations
    print("Generating heatmap visualizations...")
    plot_paths = create_experiment_2_heatmaps(
        summary_path,
        os.path.join(output_dir, 'experiment_2'),
        base_multipliers
    )

    # Generate rank plots and selection CSV
    print("Generating rank plots and selection CSV...")
    from helpers.visualization import create_experiment_2_rankplots
    rankplot_paths = create_experiment_2_rankplots(
        summary_path,
        os.path.join(output_dir, 'experiment_2'),
        base_multiplier_focus=1.0,
        eff_min=60.0,
        ecr_min=85.0
    )

    # Print final summary
    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 COMPLETED")
    print("=" * 60)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Configurations tested: {len(summary_df)}")
    print(f"Total samples evaluated: {len(results_df)}")
    print(f"Results saved to: {detailed_path}")
    print(f"Summary saved to: {summary_path}")
    if plot_paths:
        print(f"Heatmaps saved to:")
        for plot_path in plot_paths:
            print(f"  {plot_path}")
    if rankplot_paths:
        print(f"Rank plots and selection CSV saved to:")
        for plot_path in rankplot_paths:
            print(f"  {plot_path}")

    return detailed_path, summary_path


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description='Run uncertainty optimization experiment')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip', action='store_true',
                       help='Skip inference if CSV exists, only generate plots')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Check if we should skip inference and only generate plots
    if args.skip:
        output_dir = config['paths']['results']
        csv_path = os.path.join(output_dir, 'experiment_2', 'uncertainty_optimization_summary.csv')

        if os.path.exists(csv_path):
            print("CSV file found, generating plots only...")
            base_multipliers = config['experiment_2']['base_multipliers']
            plot_paths = create_experiment_2_heatmaps(
                csv_path,
                os.path.join(output_dir, 'experiment_2'),
                base_multipliers
            )
            if plot_paths:
                print("Heatmaps saved to:")
                for plot_path in plot_paths:
                    print(f"  {plot_path}")

            from helpers.visualization import create_experiment_2_rankplots

            # After heatmaps:
            rankplot_paths = create_experiment_2_rankplots(
                csv_path,
                os.path.join(output_dir, 'experiment_2'),
                base_multiplier_focus=1.0,
                eff_min=60.0,
                ecr_min=85.0
            )
            if rankplot_paths:
                print("Rank plots saved to:")
                for p in rankplot_paths:
                    print(f"  {p}")

            return
        else:
            print("CSV file not found, exiting without running experiment.")
            return

    run_uncertainty_optimization_experiment(config)


if __name__ == '__main__':
    main()