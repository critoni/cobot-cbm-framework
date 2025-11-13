"""
Human feedback simulation and management for HIL-CBM framework.
"""

import numpy as np


def simulate_human_feedback(uncertain_samples, uncertain_true_labels):
    """Simulate 100% accurate human feedback"""
    return uncertain_true_labels.copy()


def check_accumulated_feedback_trigger(accumulated_uncertain_samples, feedback_threshold, is_last_batch=False):
    """Check if accumulated feedback should be triggered"""
    if is_last_batch and len(accumulated_uncertain_samples) > 0:
        return True  # Always trigger on last batch if we have samples

    return len(accumulated_uncertain_samples) >= feedback_threshold


def request_feedback(uncertain_storage, feedback_mode='batch'):
    """Request feedback and return results"""
    if len(uncertain_storage) == 0:
        return False, [], 0

    # Extract true labels from storage
    uncertain_true_labels = [sample['true_label'] for sample in uncertain_storage]

    # Simulate human feedback (100% accurate)
    feedback_labels = simulate_human_feedback(uncertain_storage, uncertain_true_labels)

    feedback_count = len(feedback_labels)

    return True, feedback_labels, feedback_count



def collect_feedback_samples_for_retraining(feedback_labels, uncertain_storage):
    """Collect healthy feedback samples for retraining"""
    feedback_cycles = {'PAP1': [], 'PAP2': [], 'PAP3': []}

    for i, sample in enumerate(uncertain_storage):
        if i < len(feedback_labels) and feedback_labels[i] == 'healthy':
            traj = sample['trajectory']
            if traj in feedback_cycles:
                feedback_cycles[traj].append(sample['cycle'])

    return feedback_cycles


def update_accumulated_uncertain_storage(accumulated_storage, new_uncertain_samples):
    """Update accumulated uncertain storage with new samples"""
    accumulated_storage.extend(new_uncertain_samples)
    return accumulated_storage


def clear_accumulated_uncertain_storage(accumulated_storage):
    """Clear accumulated uncertain storage after feedback"""
    return []




def manage_batch_feedback_flow(uncertain_samples, accumulated_storage, feedback_mode,
                              feedback_threshold, is_last_batch=False):
    """Manage complete feedback flow for a batch"""
    feedback_requested = False
    feedback_labels = []
    feedback_count = 0
    feedback_samples_used = []  # Track samples that received feedback for performance calculation

    # Update accumulated storage
    accumulated_count_before = len(accumulated_storage)
    accumulated_storage = update_accumulated_uncertain_storage(accumulated_storage, uncertain_samples)

    if feedback_mode == 'batch':
        # Request feedback immediately if we have uncertain samples
        if len(uncertain_samples) > 0:
            feedback_requested, feedback_labels, feedback_count = request_feedback(uncertain_samples)
            feedback_samples_used = uncertain_samples.copy()  # In batch mode, feedback applies to current batch
            # Don't clear accumulated storage in batch mode - only process current batch

    elif feedback_mode == 'accumulated':
        # Check if we should trigger accumulated feedback
        should_trigger = check_accumulated_feedback_trigger(accumulated_storage, feedback_threshold, is_last_batch)

        if should_trigger:
            feedback_samples_used = accumulated_storage.copy()  # Capture samples before clearing
            feedback_requested, feedback_labels, feedback_count = request_feedback(accumulated_storage)
            # Clear accumulated storage after feedback
            accumulated_storage = clear_accumulated_uncertain_storage(accumulated_storage)

    return (feedback_requested, feedback_labels, feedback_count,
            accumulated_storage, accumulated_count_before, feedback_samples_used)


def process_feedback_for_performance(uncertain_samples, feedback_labels, feedback_mode, feedback_samples_used=None):
    """Process feedback labels for performance calculation"""
    if not feedback_labels:
        return [], []

    if feedback_mode == 'batch':
        # Feedback applies to current batch uncertain samples
        feedback_true_labels = [sample['true_label'] for sample in uncertain_samples]
        feedback_predicted_labels = feedback_labels.copy()

    elif feedback_mode == 'accumulated':
        # For accumulated mode, when feedback was triggered, use ALL samples that received feedback
        if feedback_samples_used and len(feedback_samples_used) > 0:
            # Feedback was triggered - use ALL accumulated samples that received feedback
            feedback_true_labels = [sample['true_label'] for sample in feedback_samples_used]
            feedback_predicted_labels = feedback_labels.copy()
        else:
            # No feedback was triggered this batch
            feedback_true_labels = []
            feedback_predicted_labels = []

    return feedback_true_labels, feedback_predicted_labels


def calculate_cognitive_load_time(n_samples, base_time, gamma=0.2, delta=0.1):
    """Calculate cognitive load adjusted feedback time"""
    if n_samples <= 1:
        return base_time * n_samples

    cognitive_factor = 1 + gamma * (np.exp(delta * (n_samples - 1)) - 1)
    return base_time * n_samples * cognitive_factor


def calculate_feedback_costs(feedback_info, config):
    """Calculate comprehensive feedback costs including cognitive load"""
    costs = config['experiment_4']['costs']

    # Basic feedback time cost
    total_feedback_samples = feedback_info['total_feedback_samples_so_far']

    if total_feedback_samples == 0:
        return {
            'feedback_time_cost': 0.0,
            'walking_cost': 0.0,
            'total_feedback_cost': 0.0,
            'cognitive_load_factor': 1.0,
            'effective_time_per_sample': costs['baseline_feedback_time']
        }

    # Calculate cognitive load adjusted time
    total_requests = feedback_info['total_feedback_requests_so_far']
    avg_samples_per_request = total_feedback_samples / total_requests if total_requests > 0 else 1

    cognitive_time_per_sample = calculate_cognitive_load_time(
        avg_samples_per_request,
        costs['baseline_feedback_time'],
        costs['cognitive_gamma'],
        costs['cognitive_delta']
    ) / avg_samples_per_request

    # Total feedback time cost
    total_feedback_time = total_feedback_samples * cognitive_time_per_sample
    feedback_time_cost = total_feedback_time * costs['wage_rate'] / 3600  # Convert seconds to hours

    # Walking time cost
    walking_cost = total_requests * costs['walking_time'] * costs['wage_rate'] / 3600

    # Total cost
    total_feedback_cost = feedback_time_cost + walking_cost

    return {
        'feedback_time_cost': feedback_time_cost,
        'walking_cost': walking_cost,
        'total_feedback_cost': total_feedback_cost,
        'cognitive_load_factor': cognitive_time_per_sample / costs['baseline_feedback_time'],
        'effective_time_per_sample': cognitive_time_per_sample
    }


def calculate_opportunity_cost(human_workload_percent, config):
    """Calculate opportunity cost based on human workload"""
    costs = config['experiment_4']['costs']

    # Convert workload percentage to time burden
    time_burden_hours = human_workload_percent / 100.0  # Assuming 1 hour evaluation window

    # Calculate production value per hour
    production_value_per_hour = costs['product_value'] / (costs['cycle_time'] / 3600)  # cycles per hour * value per cycle

    # Opportunity cost
    opportunity_cost = time_burden_hours * production_value_per_hour

    return opportunity_cost


def simulate_feedback_strategies(config):
    """Generate all possible feedback strategy combinations"""
    strategies = []

    for strategy_name, strategy_config in config['experiment_3']['strategies'].items():
        for timing_name, timing_config in config['experiment_3']['timing'].items():
            strategies.append({
                'name': f"{strategy_name}_{timing_name}",
                'strategy': strategy_name,
                'timing': timing_name,
                'certain_retraining': strategy_config['certain_retraining'],
                'feedback_retraining': strategy_config['feedback_retraining'],
                'feedback_mode': timing_config['mode'],
                'feedback_threshold': timing_config.get('threshold', timing_config.get('frequency', 100))
            })

    return strategies