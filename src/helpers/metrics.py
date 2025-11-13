"""
Performance metrics and evaluation utilities for HIL-CBM framework.
"""

import json
import os
import time
import numpy as np
from datetime import datetime


def calculate_performance_metrics(predicted_labels, true_labels):
    """Calculate binary classification metrics"""
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)

    # Convert to binary
    pred_binary = (predicted_labels == 'unhealthy').astype(int)
    true_binary = (true_labels == 'unhealthy').astype(int)

    # Calculate confusion matrix
    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))

    # Calculate metrics
    total = len(predicted_labels)
    correct = tp + tn
    accuracy = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        'accuracy': float(accuracy),
        'correct_count': int(correct),
        'total_count': int(total),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score)
    }


def calculate_quadruple_performance_metrics(base_ae_predictions, all_true_labels,
                                          certain_true_labels, certain_predicted_labels,
                                          uncertain_true_labels, uncertain_predicted_labels,
                                          feedback_true_labels, feedback_predicted_labels, batch_size):
    """Calculate raw autoencoder, certain-only, uncertain-only, and production system performance"""

    # Raw Autoencoder Performance (unaffected by feedback)
    # This is direct model prediction vs true labels for ALL samples
    raw_autoencoder_performance = calculate_performance_metrics(base_ae_predictions, all_true_labels)

    # Certain Autoencoder Performance (certain predictions only, no feedback)
    # This shows how well the model performs on samples it's certain about
    if len(certain_predicted_labels) > 0:
        certain_metrics = calculate_performance_metrics(certain_predicted_labels, certain_true_labels)

        # Add certain-specific metrics
        certain_samples = len(certain_predicted_labels)
        certain_automation_rate = (certain_samples / batch_size) * 100 if batch_size > 0 else 0.0

        certain_autoencoder_performance = {
            **certain_metrics,
            'certain_samples': certain_samples,
            'automation_rate': float(certain_automation_rate)
        }
    else:
        # No certain samples
        certain_autoencoder_performance = {
            'accuracy': 0.0,
            'correct_count': 0,
            'total_count': 0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'certain_samples': 0,
            'automation_rate': 0.0
        }

    # Uncertain Autoencoder Performance (uncertain predictions only, no feedback)
    # This shows how well the model performs on samples it's uncertain about
    if len(uncertain_predicted_labels) > 0:
        uncertain_metrics = calculate_performance_metrics(uncertain_predicted_labels, uncertain_true_labels)

        # Add uncertain-specific metrics
        uncertain_samples = len(uncertain_predicted_labels)
        uncertain_rate = (uncertain_samples / batch_size) * 100 if batch_size > 0 else 0.0

        uncertain_autoencoder_performance = {
            **uncertain_metrics,
            'uncertain_samples': uncertain_samples,
            'uncertain_rate': float(uncertain_rate)
        }
    else:
        # No uncertain samples
        uncertain_autoencoder_performance = {
            'accuracy': 0.0,
            'correct_count': 0,
            'total_count': 0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'uncertain_samples': 0,
            'uncertain_rate': 0.0
        }

    # Production System Performance (certain + feedback predictions)
    # This is performance after human feedback is applied to uncertain samples
    if len(certain_predicted_labels) > 0 or len(feedback_predicted_labels) > 0:
        # Combine certain + feedback predictions
        combined_predicted = certain_predicted_labels + feedback_predicted_labels
        combined_true = certain_true_labels + feedback_true_labels

        production_metrics = calculate_performance_metrics(combined_predicted, combined_true)

        # Add production-specific metrics
        total_processed = len(combined_predicted)
        uncertain_samples = len(feedback_predicted_labels)  # Total feedback samples processed (accumulated when triggered)
        automation_rate = (total_processed / batch_size) * 100 if batch_size > 0 else 0.0

        production_system_performance = {
            **production_metrics,
            'uncertain_samples': uncertain_samples,
            'automation_rate': float(automation_rate)
        }
    else:
        # All samples were uncertain, no processing done
        production_system_performance = {
            'accuracy': 0.0,
            'correct_count': 0,
            'total_count': 0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'uncertain_samples': batch_size,
            'automation_rate': 0.0
        }

    return raw_autoencoder_performance, certain_autoencoder_performance, uncertain_autoencoder_performance, production_system_performance


def create_batch_result(batch_number, batch_size, uncertain_count, feedback_info,
                       raw_performance, certain_performance, uncertain_performance, production_performance, retraining_info):
    """Create structured batch result"""
    batch_result = {
        "batch_info": {
            "batch_number": batch_number,
            "total_samples": batch_size,
            "uncertain_samples_this_batch": uncertain_count
        },
        "feedback_info": feedback_info,
        "performance": {
            "raw_autoencoder": raw_performance,
            "certain_autoencoder": certain_performance,
            "uncertain_autoencoder": uncertain_performance,
            "production_system": production_performance
        },
        "retraining_info": retraining_info
    }

    return batch_result


def create_retraining_info(certain_retraining, feedback_retraining, certain_healthy_count,
                          feedback_healthy_count, training_data_sizes, models_retrained=False, retraining_duration=None):
    """Create retraining information"""
    retraining_info = {
        "models_retrained": models_retrained,
        "certain_retraining_enabled": certain_retraining,
        "feedback_retraining_enabled": feedback_retraining,
        "new_training_samples": {
            "certain_healthy": certain_healthy_count,
            "feedback_healthy": feedback_healthy_count,
            "total_added": certain_healthy_count + feedback_healthy_count
        },
        "training_data_sizes": training_data_sizes,
        "retraining_duration_seconds": retraining_duration
    }

    return retraining_info


def calculate_summary_results(batch_results, config):
    """Calculate summary statistics across all batches"""
    n_batches = len(batch_results)

    # Aggregate raw autoencoder performance
    total_raw_correct = sum(batch["performance"]["raw_autoencoder"]["correct_count"]
                           for batch in batch_results)
    total_samples = sum(batch["batch_info"]["total_samples"] for batch in batch_results)

    avg_raw_f1 = np.mean([batch["performance"]["raw_autoencoder"]["f1_score"]
                         for batch in batch_results])

    # Aggregate certain autoencoder performance
    total_certain_correct = sum(batch["performance"]["certain_autoencoder"]["correct_count"]
                               for batch in batch_results)
    total_certain_samples = sum(batch["performance"]["certain_autoencoder"]["total_count"]
                               for batch in batch_results)

    avg_certain_f1 = np.mean([batch["performance"]["certain_autoencoder"]["f1_score"]
                             for batch in batch_results])
    avg_certain_automation = np.mean([batch["performance"]["certain_autoencoder"]["automation_rate"]
                                     for batch in batch_results])

    # Aggregate uncertain autoencoder performance
    total_uncertain_correct = sum(batch["performance"]["uncertain_autoencoder"]["correct_count"]
                                 for batch in batch_results)
    total_uncertain_samples = sum(batch["performance"]["uncertain_autoencoder"]["total_count"]
                                 for batch in batch_results)

    avg_uncertain_f1 = np.mean([batch["performance"]["uncertain_autoencoder"]["f1_score"]
                               for batch in batch_results])
    avg_uncertain_rate = np.mean([batch["performance"]["uncertain_autoencoder"]["uncertain_rate"]
                                 for batch in batch_results])

    # Aggregate production system performance
    total_prod_correct = sum(batch["performance"]["production_system"]["correct_count"]
                            for batch in batch_results)
    total_prod_processed = sum(batch["performance"]["production_system"]["total_count"]
                              for batch in batch_results)

    avg_prod_f1 = np.mean([batch["performance"]["production_system"]["f1_score"]
                          for batch in batch_results])
    avg_prod_automation = np.mean([batch["performance"]["production_system"]["automation_rate"]
                                  for batch in batch_results])

    # Feedback analysis
    total_feedback_requests = sum(1 for batch in batch_results
                                 if batch["feedback_info"]["feedback_requested"])
    total_feedback_samples = sum(batch["feedback_info"]["feedback_samples"]
                               for batch in batch_results)

    # Retraining analysis
    total_retraining_cycles = sum(1 for batch in batch_results
                                if batch["retraining_info"]["models_retrained"])

    summary_results = {
        "overall_performance": {
            "raw_autoencoder": {
                "final_accuracy": total_raw_correct / total_samples if total_samples > 0 else 0.0,
                "average_f1_score": float(avg_raw_f1)
            },
            "certain_autoencoder": {
                "final_accuracy": total_certain_correct / total_certain_samples if total_certain_samples > 0 else 0.0,
                "average_f1_score": float(avg_certain_f1),
                "average_automation_rate": float(avg_certain_automation),
                "total_certain_samples": total_certain_samples,
                "total_correct_among_certain": total_certain_correct
            },
            "uncertain_autoencoder": {
                "final_accuracy": total_uncertain_correct / total_uncertain_samples if total_uncertain_samples > 0 else 0.0,
                "average_f1_score": float(avg_uncertain_f1),
                "average_uncertain_rate": float(avg_uncertain_rate),
                "total_uncertain_samples": total_uncertain_samples,
                "total_correct_among_uncertain": total_uncertain_correct
            },
            "production_system": {
                "final_accuracy": total_prod_correct / total_prod_processed if total_prod_processed > 0 else 0.0,
                "average_f1_score": float(avg_prod_f1),
                "average_automation_rate": float(avg_prod_automation),
                "total_processed_samples": total_prod_processed,
                "total_correct_among_processed": total_prod_correct
            }
        },
        "feedback_analysis": {
            "total_feedback_requests": total_feedback_requests,
            "total_feedback_samples": total_feedback_samples,
            "feedback_frequency": f"{(total_feedback_requests / n_batches) * 100:.1f}% of batches",
            "average_feedback_per_request": total_feedback_samples / total_feedback_requests if total_feedback_requests > 0 else 0
        },
        "retraining_analysis": {
            "total_retraining_cycles": total_retraining_cycles,
            "new_samples_from_certain_healthy": sum(
                batch["retraining_info"]["new_training_samples"]["certain_healthy"]
                for batch in batch_results
            ),
            "new_samples_from_feedback": sum(
                batch["retraining_info"]["new_training_samples"]["feedback_healthy"]
                for batch in batch_results
            )
        }
    }

    return summary_results


def create_experiment_results(config, batch_results, start_time, total_samples, experiment_name):
    """Create complete experiment results structure"""

    # Calculate summary
    summary_results = calculate_summary_results(batch_results, config)

    # Create complete results
    experiment_results = {
        "experiment_config": {
            **config,
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "total_batches": len(batch_results),
            "batch_size": batch_results[0]["batch_info"]["total_samples"] if batch_results else config['global']['batch_size']
        },
        "detailed_results": {
            f"batch_{i+1}": batch_results[i] for i in range(len(batch_results))
        },
        "summary_results": summary_results,
        "metadata": {
            "experiment_duration_seconds": int(time.time() - start_time),
            "cpu_cores_used": config['computation']['cpu_threads'],
            "model_architecture": "Conv1D Autoencoder",
            "dataset_info": {
                "total_samples": total_samples,
                "trajectories": config['global']['trajectories'],
                "fault_types": ["normal", "bottle", "random", "acc_band"]
            }
        }
    }

    return experiment_results


def generate_filename(config, experiment_name, params=None):
    """Generate filename based on configuration and experiment parameters"""
    filename = f"{experiment_name}"

    if params:
        for key, value in params.items():
            if isinstance(value, float):
                filename += f"_{key}{value:.3f}"
            elif isinstance(value, bool):
                filename += f"_{key}{str(value).lower()}"
            else:
                filename += f"_{key}{value}"

    filename += ".json"
    return filename


def save_experiment_results(results, output_dir, experiment_name, params=None):
    """Save experiment results to JSON file"""
    filename = generate_filename(results['experiment_config'], experiment_name, params)
    output_path = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    return output_path


def calculate_temporal_stability(batch_results, metric_name='accuracy'):
    """Calculate temporal stability of performance metrics"""
    performance_values = []

    for batch in batch_results:
        if metric_name in batch['performance']['raw_autoencoder']:
            performance_values.append(batch['performance']['raw_autoencoder'][metric_name])

    if len(performance_values) < 2:
        return {'mean': 0, 'std': 0, 'coefficient_of_variation': 0}

    mean_performance = np.mean(performance_values)
    std_performance = np.std(performance_values)
    cv = std_performance / mean_performance if mean_performance != 0 else 0

    return {
        'mean': float(mean_performance),
        'std': float(std_performance),
        'coefficient_of_variation': float(cv),
        'values': performance_values
    }


def calculate_economic_metrics(batch_results, config):
    """Calculate comprehensive economic metrics"""
    costs = config['experiment_4']['costs']

    # Training data labeling cost
    initial_training_samples = config['preprocessing']['seed_size'] * len(config['global']['trajectories'])
    labeling_cost = initial_training_samples * costs['labeling_per_cycle']

    # Sum up all feedback costs
    total_feedback_cost = 0
    total_opportunity_cost = 0
    total_error_cost = 0

    for batch in batch_results:
        # Import feedback helper for cost calculation
        from .feedback import calculate_feedback_costs, calculate_opportunity_cost

        # Feedback costs
        feedback_costs = calculate_feedback_costs(batch['feedback_info'], config)
        total_feedback_cost += feedback_costs['total_feedback_cost']

        # Opportunity costs
        human_workload = batch['performance']['production_system'].get('uncertain_samples', 0) / batch['batch_info']['total_samples'] * 100
        opportunity_cost = calculate_opportunity_cost(human_workload, config)
        total_opportunity_cost += opportunity_cost

        # Error costs
        fp = batch['performance']['production_system']['false_positives']
        fn = batch['performance']['production_system']['false_negatives']
        error_cost = fp * costs['false_positive_cost'] + fn * costs['false_negative_cost']
        total_error_cost += error_cost

    total_cost = labeling_cost + total_feedback_cost + total_opportunity_cost + total_error_cost

    return {
        'labeling_cost': labeling_cost,
        'feedback_cost': total_feedback_cost,
        'opportunity_cost': total_opportunity_cost,
        'error_cost': total_error_cost,
        'total_cost': total_cost,
        'cost_breakdown_percent': {
            'labeling': (labeling_cost / total_cost * 100) if total_cost > 0 else 0,
            'feedback': (total_feedback_cost / total_cost * 100) if total_cost > 0 else 0,
            'opportunity': (total_opportunity_cost / total_cost * 100) if total_cost > 0 else 0,
            'error': (total_error_cost / total_cost * 100) if total_cost > 0 else 0
        }
    }