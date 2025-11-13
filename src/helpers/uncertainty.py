"""
Uncertainty quantification and 3-zone classification logic for HIL-CBM framework.
"""

import numpy as np


def apply_uncertainty_thresholds(mae_scores, base_thresholds, base_multiplier=1.0, uncertainty_multiplier=1.05):
    """Apply 3-zone classification to MAE scores"""
    n_samples, n_vars = mae_scores.shape
    zones = np.zeros((n_samples, n_vars), dtype=int)

    # Calculate thresholds
    modified_base = base_thresholds * base_multiplier
    uncertainty_th = base_thresholds * uncertainty_multiplier

    # Classify zones for each variable
    for j in range(n_vars):
        zones[:, j] = np.where(
            mae_scores[:, j] <= modified_base[j], 1,  # Zone 1: Certain Healthy
            np.where(mae_scores[:, j] <= uncertainty_th[j], 2, 3)  # Zone 2: Uncertain, Zone 3: Certain Unhealthy
        )

    return zones


def apply_priority_logic(zones):
    """Apply priority logic: Zone 3 > Zone 2 > Zone 1"""
    n_samples = zones.shape[0]
    uncertainty_flags = np.zeros(n_samples, bool)
    final_predictions = []

    for i in range(n_samples):
        sample_zones = zones[i, :]

        if np.any(sample_zones == 3):  # Any certain unhealthy
            final_predictions.append('unhealthy')
            uncertainty_flags[i] = False
        elif np.any(sample_zones == 2):  # Any uncertain
            final_predictions.append('uncertain')
            uncertainty_flags[i] = True
        else:  # All certain healthy
            final_predictions.append('healthy')
            uncertainty_flags[i] = False

    return final_predictions, uncertainty_flags


def get_base_ae_predictions(mae_scores, base_thresholds):
    """Get base autoencoder predictions (original logic)"""
    base_predictions = []
    for i in range(len(mae_scores)):
        base_pred = 'unhealthy' if np.any(mae_scores[i] > base_thresholds) else 'healthy'
        base_predictions.append(base_pred)
    return base_predictions


def get_uncertain_sample_info(uncertainty_flags, batch_cycles, batch_trajectories, true_labels, batch_idx):
    """Extract uncertain sample information for storage"""
    uncertain_samples = []
    uncertain_indices = np.where(uncertainty_flags)[0]

    for idx in uncertain_indices:
        uncertain_samples.append({
            'batch_idx': batch_idx,
            'local_idx': idx,
            'trajectory': batch_trajectories[idx],
            'true_label': true_labels[idx],
            'cycle': batch_cycles[idx]
        })

    return uncertain_samples, uncertain_indices


def collect_certain_healthy_samples(zones, final_predictions, batch_cycles, batch_trajectories):
    """Collect certain healthy samples for retraining"""
    certain_healthy_cycles = {'PAP1': [], 'PAP2': [], 'PAP3': []}

    for i in range(len(batch_cycles)):
        traj = batch_trajectories[i]
        # Only collect if ALL variables are in Zone 1 AND final prediction is 'healthy'
        if np.all(zones[i] == 1) and final_predictions[i] == 'healthy' and traj in certain_healthy_cycles:
            certain_healthy_cycles[traj].append(batch_cycles[i])

    return certain_healthy_cycles


def batch_inference_with_uncertainty(batch_cycles, batch_trajectories, models_dict, thresholds_dict,
                                   base_multiplier=1.0, uncertainty_multiplier=1.05, config=None):
    """Perform batch inference with uncertainty quantification"""
    from .models import batch_inference, get_base_predictions

    n_samples = len(batch_cycles)
    n_variables = config['global']['n_variables']

    # Get MAE scores using inference functions from models.py
    all_mae_scores = batch_inference(batch_cycles, batch_trajectories, models_dict, config)

    # Get base AE predictions
    base_ae_predictions = get_base_predictions(all_mae_scores, batch_trajectories, thresholds_dict)

    # Apply uncertainty quantification
    all_zones = np.zeros((n_samples, n_variables), dtype=int)
    uncertainty_predictions = []
    uncertainty_flags = np.zeros(n_samples, bool)

    for i in range(n_samples):
        traj = batch_trajectories[i]
        if traj in thresholds_dict:
            base_thresholds = thresholds_dict[traj]
            sample_mae = all_mae_scores[i, :]

            # 3-zone classification
            sample_zones = apply_uncertainty_thresholds(
                sample_mae.reshape(1, -1), base_thresholds,
                base_multiplier, uncertainty_multiplier
            )[0]
            all_zones[i, :] = sample_zones

            # Priority logic
            final_predictions, uncertainty_flag_array = apply_priority_logic(sample_zones.reshape(1, -1))
            uncertainty_predictions.append(final_predictions[0])
            uncertainty_flags[i] = uncertainty_flag_array[0]
        else:
            base_ae_predictions[i] = 'healthy'
            uncertainty_predictions.append('healthy')
            uncertainty_flags[i] = False
            all_zones[i, :] = 1

    return all_mae_scores, all_zones, uncertainty_flags, uncertainty_predictions, base_ae_predictions


def calculate_uncertainty_metrics(final_predictions, uncertainty_flags, true_labels, base_ae_predictions=None):
    """Calculate uncertainty model performance metrics"""
    final_predictions = np.array(final_predictions)
    uncertainty_flags = np.array(uncertainty_flags)
    true_labels = np.array(true_labels)

    # F1 Score on definitive samples (excluding uncertain)
    definitive_mask = final_predictions != 'uncertain'
    if np.any(definitive_mask):
        definitive_preds = final_predictions[definitive_mask]
        definitive_true = true_labels[definitive_mask]

        pred_binary = (definitive_preds == 'unhealthy').astype(int)
        true_binary = (definitive_true == 'unhealthy').astype(int)

        if len(np.unique(true_binary)) > 1 and len(pred_binary) > 0:
            tp = np.sum((pred_binary == 1) & (true_binary == 1))
            fp = np.sum((pred_binary == 1) & (true_binary == 0))
            fn = np.sum((pred_binary == 0) & (true_binary == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            f1_score = 0
    else:
        f1_score = 0

    # Human workload and automation rate
    human_workload = np.mean(uncertainty_flags) * 100
    automation_rate = (1 - np.mean(uncertainty_flags)) * 100

    # Base AE error analysis if provided
    if base_ae_predictions is not None:
        base_ae_predictions = np.array(base_ae_predictions)
        base_ae_wrong = (base_ae_predictions != true_labels)
        total_wrong = np.sum(base_ae_wrong)

        if total_wrong > 0:
            wrong_flagged_uncertain = np.sum(base_ae_wrong & uncertainty_flags)
            error_catch_rate = (wrong_flagged_uncertain / total_wrong) * 100
        else:
            error_catch_rate = 100

        # Workload efficiency
        total_uncertain = np.sum(uncertainty_flags)
        if total_uncertain > 0:
            workload_efficiency = (wrong_flagged_uncertain / total_uncertain) * 100
        else:
            workload_efficiency = 100
            wrong_flagged_uncertain = 0
    else:
        error_catch_rate = 0
        workload_efficiency = 0
        wrong_flagged_uncertain = 0
        total_wrong = 0

    return {
        'f1_score': f1_score * 100,
        'error_catch_rate': error_catch_rate,
        'human_workload': human_workload,
        'workload_efficiency': workload_efficiency,
        'automation_rate': automation_rate,
        'total_samples': len(final_predictions),
        'uncertain_samples': int(np.sum(uncertainty_flags)),
        'definitive_samples': int(np.sum(definitive_mask)) if 'definitive_mask' in locals() else len(final_predictions) - int(np.sum(uncertainty_flags)),
        'base_errors_caught': int(wrong_flagged_uncertain),
        'total_base_errors': int(total_wrong)
    }