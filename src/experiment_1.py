"""
Experiment 1: Training Data Scaling Analysis

Evaluates autoencoder diagnostic performance across varying training data sizes.
Tests training data per trajectory from 1 to 100 samples.
"""

import os
import time
import argparse
import yaml
import pandas as pd
from tqdm import tqdm

from helpers.data_utils import prepare_experiment_data, get_initial_training_data, get_batch_data
from helpers.models import setup_tensorflow_environment, train_trajectory_models, save_trained_models, batch_inference, get_base_predictions
from helpers.visualization import create_experiment_1_boxplot


def load_config(config_path='config.yaml'):
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def perform_streaming_evaluation(models_dict, thresholds_dict, stream_df, raw_data, config):
    """Evaluate trained models on streaming data batches"""
    results = []

    # Process batches
    for batch_idx in range(config['global']['n_batches']):
        batch_cycles, batch_trajectories, true_labels = get_batch_data(
            stream_df, raw_data, batch_idx, config['global']['batch_size']
        )

        # Perform batch inference using refactored inference functions
        mae_scores = batch_inference(batch_cycles, batch_trajectories, models_dict, config)
        predicted_labels = get_base_predictions(mae_scores, batch_trajectories, thresholds_dict)

        # Store results
        for true_label, predicted_label in zip(true_labels, predicted_labels):
            results.append({
                'batch': batch_idx,
                'true_label': true_label,
                'predicted_label': predicted_label
            })

    return results


def run_training_scaling_experiment(config):
    """Execute training data scaling experiment"""
    print("=" * 60)
    print("EXPERIMENT 1: Training Data Scaling Analysis")
    print("=" * 60)

    start_time = time.time()

    setup_tensorflow_environment(config)

    # Load data
    print("Loading data...")
    seed_df, stream_df, raw_data = prepare_experiment_data(config, 'experiment_1')

    # Training sizes to evaluate
    training_sizes = config['experiment_1']['training_sizes']
    print(f"Training sizes to test: {training_sizes}")

    # Collect results across all training sizes
    all_results = []

    # Process each training size
    for seed_size in tqdm(training_sizes, desc='Training Size Sweep'):
        print(f"\n--- Processing training size: {seed_size} samples per trajectory ---")

        # Extract training data
        initial_training_data = get_initial_training_data(seed_df, raw_data, seed_size)

        # Check if we have enough data
        min_samples = min(len(initial_training_data[traj]) for traj in ['PAP1', 'PAP2', 'PAP3'])
        if min_samples < seed_size:
            print(f"Warning: Not enough training data for seed_size={seed_size}. Min available: {min_samples}")
            continue

        # Train autoencoder models (or load existing ones)
        print(f"Training models with {seed_size} samples per trajectory...")
        models_dict, thresholds_dict, was_loaded = train_trajectory_models(initial_training_data, config, seed_size, skip_existing=True)

        # Save models to disk only if they were newly trained
        if not was_loaded:
            save_trained_models(models_dict, thresholds_dict, config, seed_size)

        # Evaluate on streaming data
        print("Performing streaming evaluation...")
        batch_results = perform_streaming_evaluation(models_dict, thresholds_dict, stream_df, raw_data, config)

        # Store results with training size metadata
        for result in batch_results:
            result['seed_size'] = seed_size
            all_results.append(result)

        # Display current results
        df_temp = pd.DataFrame(batch_results)
        accuracy = (df_temp['true_label'] == df_temp['predicted_label']).mean()
        print(f"Overall accuracy for seed_size={seed_size}: {accuracy:.3f}")

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['seed_size', 'batch', 'true_label', 'predicted_label']]

    # Save to results directory
    output_dir = config['paths']['results']
    os.makedirs(os.path.join(output_dir, 'experiment_1'), exist_ok=True)

    output_path = os.path.join(output_dir, 'experiment_1', 'ae_cycle_metrics.csv')
    results_df.to_csv(output_path, index=False)

    # Generate visualization
    print("Generating visualization...")
    plot_path, summary_path = create_experiment_1_boxplot(output_path, os.path.join(output_dir, 'experiment_1'))

    # Print final summary
    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETED")
    print("=" * 60)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Training sizes tested: {len(training_sizes)}")
    print(f"Total samples evaluated: {len(results_df)}")
    print(f"Raw results saved to: {output_path}")
    print(f"Summary CSV saved to: {summary_path}")
    print(f"Plot saved to: {plot_path}")

    return output_path


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description='Run training data scaling experiment')
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
        csv_path = os.path.join(output_dir, 'experiment_1', 'ae_cycle_metrics.csv')

        if os.path.exists(csv_path):
            print("CSV file found, generating plots only...")
            plot_path = create_experiment_1_boxplot(csv_path, os.path.join(output_dir, 'experiment_1'))
            print(f"Plot saved to: {plot_path}")
            return
        else:
            print("CSV file not found, exiting without running experiment.")
            return

    run_training_scaling_experiment(config)


if __name__ == '__main__':
    main()