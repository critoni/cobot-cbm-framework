"""
Experiment 3: Policy Comparison Analysis

Compares different retraining strategies and feedback timing approaches
to evaluate their impact on model performance over time.
"""

import os
import time
import argparse
import yaml
import json
from datetime import datetime
from tqdm import tqdm

from helpers.data_utils import prepare_experiment_data, get_initial_training_data, get_batch_data
from helpers.models import setup_tensorflow_environment, train_trajectory_models, save_trained_models, clone_models_for_retraining, retrain_models_with_data
from helpers.uncertainty import batch_inference_with_uncertainty, collect_certain_healthy_samples
from helpers.feedback import manage_batch_feedback_flow, process_feedback_for_performance, collect_feedback_samples_for_retraining
from helpers.metrics import calculate_quadruple_performance_metrics, create_batch_result, create_retraining_info, create_experiment_results as create_exp_results


def load_config(config_path='config.yaml'):
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)






def get_optimal_uncertainty_config(seed_size, selection_strategy, config):
    """Get optimal uncertainty configuration for given training size and strategy"""
    from helpers.selection import load_selected_multiplier

    csv_path = config['experiment_3']['selected_multipliers_csv']
    uncertainty_multiplier = load_selected_multiplier(csv_path, seed_size, selection_strategy)

    return {
        'base_multiplier': 1.00,
        'uncertainty_multiplier': uncertainty_multiplier
    }


def execute_policy_strategy(strategy_config, models_dict, thresholds_dict, stream_df, raw_data, config, seed_size, selection_strategy, initial_training_data):
    """Execute a specific policy strategy across all batches"""
    results = []

    # Get optimal uncertainty configuration
    uncertainty_config = get_optimal_uncertainty_config(seed_size, selection_strategy, config)
    base_multiplier = uncertainty_config['base_multiplier']
    uncertainty_multiplier = uncertainty_config['uncertainty_multiplier']

    # Initialize models based on configuration
    if strategy_config['certain_retraining'] or strategy_config['feedback_retraining']:
        # Clone models for retraining
        current_models, current_thresholds = clone_models_for_retraining(models_dict, thresholds_dict, config)
        # Track accumulated training data (starts with initial data)
        accumulated_training_data = {traj: initial_training_data[traj].copy() for traj in ['PAP1', 'PAP2', 'PAP3']}
        # Track new training data per batch (for retraining trigger)
        new_training_data = {traj: [] for traj in ['PAP1', 'PAP2', 'PAP3']}
    else:
        # Use static models
        current_models, current_thresholds = models_dict, thresholds_dict
        accumulated_training_data = None
        new_training_data = None

    # Initialize storage
    accumulated_uncertain_storage = []
    total_feedback_requests = 0
    total_feedback_samples = 0

    # Process each batch
    for batch_idx in tqdm(range(config['global']['n_batches']),
                         desc=f"Processing {strategy_config['name']}", leave=False):

        # Get batch data
        batch_cycles, batch_trajectories, true_labels = get_batch_data(
            stream_df, raw_data, batch_idx, config['global']['batch_size']
        )

        # Perform uncertainty inference using current models
        mae_scores, zones, uncertainty_flags, final_predictions, base_ae_predictions = batch_inference_with_uncertainty(
            batch_cycles, batch_trajectories, current_models, current_thresholds,
            base_multiplier, uncertainty_multiplier, config
        )

        # Separate certain and uncertain samples
        certain_mask = ~uncertainty_flags
        uncertain_mask = uncertainty_flags

        certain_true_labels = [true_labels[i] for i in range(len(true_labels)) if certain_mask[i]]
        certain_predicted_labels = [final_predictions[i] for i in range(len(final_predictions)) if certain_mask[i]]

        # Prepare uncertain samples for feedback
        uncertain_samples = []
        for i in range(len(final_predictions)):
            if uncertain_mask[i]:
                uncertain_samples.append({
                    'cycle': batch_cycles[i],
                    'trajectory': batch_trajectories[i],
                    'true_label': true_labels[i],
                    'mae_scores': mae_scores[i]
                })

        # Manage feedback flow
        is_last_batch = (batch_idx == config['global']['n_batches'] - 1)
        feedback_requested, feedback_labels, feedback_count, accumulated_uncertain_storage, accumulated_count_before, feedback_samples_used = manage_batch_feedback_flow(
            uncertain_samples, accumulated_uncertain_storage,
            strategy_config['feedback_mode'], strategy_config['feedback_threshold'], is_last_batch
        )

        # Process feedback for performance calculation
        feedback_true_labels, feedback_predicted_labels = process_feedback_for_performance(
            uncertain_samples, feedback_labels, strategy_config['feedback_mode'], feedback_samples_used
        )

        # Update feedback totals
        if feedback_requested:
            total_feedback_requests += 1
            total_feedback_samples += feedback_count

        # Calculate quadruple performance (raw + certain + uncertain + production)
        # Extract uncertain data (before feedback) - use raw autoencoder predictions
        uncertain_true_labels = [true_labels[i] for i in range(len(true_labels)) if uncertainty_flags[i]]
        uncertain_predicted_labels = [base_ae_predictions[i] for i in range(len(base_ae_predictions)) if uncertainty_flags[i]]

        raw_performance, certain_performance, uncertain_performance, production_performance = calculate_quadruple_performance_metrics(
            base_ae_predictions, true_labels, certain_true_labels, certain_predicted_labels,
            uncertain_true_labels, uncertain_predicted_labels, feedback_true_labels, feedback_predicted_labels, len(true_labels)
        )

        # Create feedback info structure
        feedback_info = {
            "feedback_requested": feedback_requested,
            "feedback_samples": feedback_count,
            "accumulated_uncertain_before_feedback": accumulated_count_before,
            "total_feedback_requests_so_far": total_feedback_requests,
            "total_feedback_samples_so_far": total_feedback_samples
        }

        # Create retraining info structure (will be updated after collecting samples)
        models_retrained = False
        retraining_info = create_retraining_info(
            strategy_config['certain_retraining'], strategy_config['feedback_retraining'],
            0, 0,  # Will be updated after collecting samples
            {traj: len(accumulated_training_data[traj]) for traj in ['PAP1', 'PAP2', 'PAP3']} if accumulated_training_data else {'PAP1': seed_size, 'PAP2': seed_size, 'PAP3': seed_size},
            models_retrained
        )

        # Store batch results
        batch_result = create_batch_result(
            batch_idx + 1, len(true_labels), len(uncertain_samples),
            feedback_info, raw_performance, certain_performance, uncertain_performance, production_performance, retraining_info
        )

        results.append(batch_result)

        # Collect certain samples if strategy uses certain retraining
        certain_healthy_count = 0
        if strategy_config['certain_retraining']:
            certain_healthy_cycles = collect_certain_healthy_samples(zones, final_predictions, batch_cycles, batch_trajectories)
            certain_healthy_count = sum(len(cycles) for cycles in certain_healthy_cycles.values())

            # Add to new training data for this batch
            for traj in ['PAP1', 'PAP2', 'PAP3']:
                new_training_data[traj].extend(certain_healthy_cycles[traj])

        # Collect feedback samples if strategy uses feedback retraining and feedback was given
        feedback_healthy_count = 0
        if strategy_config['feedback_retraining'] and feedback_requested:
            if strategy_config['feedback_mode'] == 'batch':
                # Use current batch feedback
                feedback_cycles = collect_feedback_samples_for_retraining(feedback_labels, uncertain_samples)
            else:
                # Use ALL accumulated feedback samples that were processed
                feedback_cycles = collect_feedback_samples_for_retraining(feedback_labels, feedback_samples_used)

            feedback_healthy_count = sum(len(cycles) for cycles in feedback_cycles.values())

            # Add to new training data for this batch
            for traj in ['PAP1', 'PAP2', 'PAP3']:
                new_training_data[traj].extend(feedback_cycles[traj])

        # Retrain Models (if new data available and before next batch)
        models_retrained = False
        retraining_duration = None
        if (strategy_config['certain_retraining'] or strategy_config['feedback_retraining']) and batch_idx < config['global']['n_batches'] - 1:
            # Check if we have NEW training data this batch
            if any(len(new_training_data[traj]) > 0 for traj in ['PAP1', 'PAP2', 'PAP3']):
                # Move new data to accumulated data
                for traj in ['PAP1', 'PAP2', 'PAP3']:
                    accumulated_training_data[traj].extend(new_training_data[traj])

                # Print retraining info
                total_samples = sum(len(accumulated_training_data[traj]) for traj in ['PAP1', 'PAP2', 'PAP3'])
                print(f"  Batch {batch_idx + 1}: Retraining models with {total_samples} samples (PAP1:{len(accumulated_training_data['PAP1'])}, PAP2:{len(accumulated_training_data['PAP2'])}, PAP3:{len(accumulated_training_data['PAP3'])}), use_validation=False")

                # Retrain models from scratch with timing
                retraining_start = time.time()
                retrain_models_with_data(current_models, current_thresholds, accumulated_training_data, config)
                retraining_duration = time.time() - retraining_start
                models_retrained = True

                print(f"  Retraining completed in {retraining_duration:.2f}s")

                # Reset new training data after retraining
                new_training_data = {traj: [] for traj in ['PAP1', 'PAP2', 'PAP3']}

        # Update retraining info with actual counts and status
        retraining_info = create_retraining_info(
            strategy_config['certain_retraining'], strategy_config['feedback_retraining'],
            certain_healthy_count, feedback_healthy_count,
            {traj: len(accumulated_training_data[traj]) for traj in ['PAP1', 'PAP2', 'PAP3']} if accumulated_training_data else {'PAP1': seed_size, 'PAP2': seed_size, 'PAP3': seed_size},
            models_retrained, retraining_duration
        )

        # Update batch result with correct retraining info
        batch_result["retraining_info"] = retraining_info

    return results


def run_policy_comparison_experiment(seed_size, selection_strategy, certain_retraining, feedback_retraining, feedback_mode, feedback_threshold, config):
    """Execute single policy comparison experiment"""
    print("=" * 60)
    print("EXPERIMENT 3: Policy Comparison Analysis (Single Strategy)")
    print("=" * 60)

    start_time = time.time()

    setup_tensorflow_environment(config)

    # Load data
    print("Loading data...")
    seed_df, stream_df, raw_data = prepare_experiment_data(config, 'experiment_3')

    print(f"Training size: {seed_size} samples per trajectory")
    print(f"Selection strategy: {selection_strategy}")
    print(f"Policy: certain_retraining={certain_retraining}, feedback_retraining={feedback_retraining}")
    print(f"Feedback: mode={feedback_mode}, threshold={feedback_threshold}")

    # Get initial training data
    initial_training_data = get_initial_training_data(seed_df, raw_data, seed_size)

    # Check if we have enough data
    min_samples = min(len(initial_training_data[traj]) for traj in ['PAP1', 'PAP2', 'PAP3'])
    if min_samples < seed_size:
        raise ValueError(f"Not enough training data for seed_size={seed_size}. Min available: {min_samples}")

    # Train initial models
    print(f"Training initial models with {seed_size} samples per trajectory...")
    models_dict, thresholds_dict, was_loaded = train_trajectory_models(
        initial_training_data, config, seed_size, skip_existing=True
    )

    # Save models if newly trained
    if not was_loaded:
        save_trained_models(models_dict, thresholds_dict, config, seed_size)

    # Create strategy configuration
    strategy_config = {
        'name': f"{selection_strategy}_{'certain' if certain_retraining else 'nocertain'}_{'feedback' if feedback_retraining else 'nofeedback'}_{feedback_mode}",
        'certain_retraining': certain_retraining,
        'feedback_retraining': feedback_retraining,
        'feedback_mode': feedback_mode,
        'feedback_threshold': feedback_threshold
    }

    # Execute strategy
    print(f"Executing strategy: {strategy_config['name']}")
    strategy_results = execute_policy_strategy(
        strategy_config, models_dict, thresholds_dict, stream_df, raw_data, config, seed_size, selection_strategy, initial_training_data
    )

    all_results = strategy_results

    # Get uncertainty config for summary
    uncertainty_config = get_optimal_uncertainty_config(seed_size, selection_strategy, config)

    # Create experiment results JSON structure using metrics module
    experiment_config = {
        'seed_size': seed_size,
        'selection_strategy': selection_strategy,
        'uncertainty_multiplier': uncertainty_config['uncertainty_multiplier'],
        'certain_retraining': certain_retraining,
        'feedback_retraining': feedback_retraining,
        'feedback_mode': feedback_mode,
        'feedback_threshold': feedback_threshold,
        'timestamp': datetime.now().isoformat(),
        'total_batches': config['global']['n_batches'],
        'batch_size': config['global']['batch_size']
    }
    experiment_results = create_exp_results(config, all_results, start_time, len(stream_df), "experiment_3")

    # Override experiment_config with only experiment-specific parameters
    experiment_results["experiment_config"] = experiment_config

    # Save results as JSON
    output_dir = config['paths']['results']
    os.makedirs(os.path.join(output_dir, 'experiment_3'), exist_ok=True)
    detailed_path = save_experiment_results(experiment_results, output_dir, seed_size, selection_strategy,
                                          certain_retraining, feedback_retraining, feedback_mode, feedback_threshold,
                                          uncertainty_config['uncertainty_multiplier'])

    # Calculate summary metrics from JSON structure
    summary = experiment_results["summary_results"]
    avg_accuracy = summary['overall_performance']['production_system']['final_accuracy']
    avg_f1_score = summary['overall_performance']['production_system']['average_f1_score']
    avg_workload = summary['feedback_analysis']['total_feedback_samples'] / config['global']['n_batches']  # Average feedback samples per batch
    final_feedback_requests = summary['feedback_analysis']['total_feedback_requests']
    final_feedback_samples = summary['feedback_analysis']['total_feedback_samples']

    # Print final summary
    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 COMPLETED")
    print("=" * 60)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Strategy: {strategy_config['name']}")
    print(f"Training size: {seed_size}")
    print(f"Selection strategy: {selection_strategy}")
    print(f"Total batch results: {len(all_results)}")
    print(f"Results saved to: {detailed_path}")
    print("\nPerformance Summary:")
    print(f"  Average Accuracy: {avg_accuracy:.1f}%")
    print(f"  Average F1 Score: {avg_f1_score:.1f}%")
    print(f"  Average Human Workload: {avg_workload:.1f}%")
    print(f"  Total Feedback Requests: {final_feedback_requests}")
    print(f"  Total Feedback Samples: {final_feedback_samples}")

    return detailed_path



def save_experiment_results(results, output_dir, seed_size, selection_strategy, certain_retraining,
                           feedback_retraining, feedback_mode, feedback_threshold, uncertainty_multiplier):
    """Save experiment results to JSON file with specific naming format"""

    filename = f"policy_experiment_seed_size{seed_size}_unc{uncertainty_multiplier:.3f}"
    filename += f"_certretrain{str(certain_retraining).lower()}"
    filename += f"_feedretrain{str(feedback_retraining).lower()}"
    filename += f"_feedtype{feedback_mode}"
    filename += f"_threshold{feedback_threshold}.json"

    output_path = os.path.join(output_dir, 'experiment_3', filename)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    return output_path


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description='Run single policy comparison experiment')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--seed_size', type=int, required=True, choices=[25, 50, 100], help='Training data size per trajectory')
    parser.add_argument('--selection_strategy', type=str, required=True, choices=['best_err', 'best_eff', 'best_balance'], help='Selection strategy from experiment 2')
    parser.add_argument('--certain_retraining', type=str, required=True, choices=['true', 'false'], help='Enable certain sample retraining')
    parser.add_argument('--feedback_retraining', type=str, required=True, choices=['true', 'false'], help='Enable feedback-based retraining')
    parser.add_argument('--feedback_mode', type=str, required=True, choices=['batch', 'accumulated'], help='Feedback timing mode')
    parser.add_argument('--feedback_threshold', type=int, default=30, help='Threshold for accumulated feedback')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Convert string booleans
    certain_retraining = args.certain_retraining.lower() == 'true'
    feedback_retraining = args.feedback_retraining.lower() == 'true'

    # Check if experiment already exists
    output_dir = config['paths']['results']
    # Get uncertainty multiplier to generate correct filename
    from helpers.selection import load_selected_multiplier
    csv_path = config['experiment_3']['selected_multipliers_csv']
    uncertainty_multiplier = load_selected_multiplier(csv_path, args.seed_size, args.selection_strategy)

    filename = f"policy_experiment_seed_size{args.seed_size}_unc{uncertainty_multiplier:.3f}"
    filename += f"_certretrain{str(certain_retraining).lower()}"
    filename += f"_feedretrain{str(feedback_retraining).lower()}"
    filename += f"_feedtype{args.feedback_mode}"
    filename += f"_threshold{args.feedback_threshold}.json"
    output_path = os.path.join(output_dir, 'experiment_3', filename)

    if os.path.exists(output_path):
        print(f"Experiment already exists: {filename}")
        print(f"Results available at: {output_path}")
        return

    # Run experiment
    run_policy_comparison_experiment(
        args.seed_size, args.selection_strategy, certain_retraining, feedback_retraining,
        args.feedback_mode, args.feedback_threshold, config
    )


if __name__ == '__main__':
    main()