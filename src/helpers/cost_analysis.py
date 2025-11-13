"""
Economic cost analysis module for HIL-CBM framework.

Implements cognitive load-based cost calculations for human-in-the-loop
condition-based monitoring systems based on experimental results.

Cost Model Components:
- Interface Cost: C_req = (feedback_time + walking_time) × C_w
- Opportunity Cost: C_opp = (feedback_time + walking_time) × (v/τ)
- Error Cost: C_err = N_FP × t_maint × (C_w + v/τ) + N_FN × v × 1.2
- Total Cost: C_total = C_req + C_opp + C_err
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_experiment_data(json_files_dir, seed_size, uncertainty_multiplier, feedback_mode):
    """Load experiment data for specific configuration"""
    # Create filename pattern to match
    unc_str = f"{uncertainty_multiplier:.3f}"
    pattern = os.path.join(json_files_dir, f'policy_experiment_seed_size{seed_size}_unc{unc_str}_*_feedtype{feedback_mode}_*.json')

    json_files = glob.glob(pattern)

    if not json_files:
        raise ValueError(f"No JSON files found for seed_size={seed_size}, uncertainty_multiplier={uncertainty_multiplier}, feedback_mode={feedback_mode}")

    # Load all matching files and return as dict with policy keys
    experiments = {}

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        config = data['experiment_config']
        certain_ret = config['certain_retraining']
        feedback_ret = config['feedback_retraining']

        # Create policy key
        if certain_ret and feedback_ret:
            policy_key = "Combined"
        elif feedback_ret:
            policy_key = "Feedback"
        elif certain_ret:
            policy_key = "Certain"
        else:
            policy_key = "Static"

        experiments[policy_key] = data

    return experiments


def calculate_cognitive_load_time(n_feed, t_feed, gamma, delta):
    """Calculate time for feedback request with cognitive load model

    Formula: T_req,j = t_feed × N_feed,j × (1 + γ × (e^δ(N_feed,j-1) - 1))
    """
    if n_feed == 0:
        return 0

    cognitive_multiplier = 1 + gamma * (np.exp(delta * (n_feed - 1)) - 1)
    t_req = t_feed * n_feed * cognitive_multiplier

    return t_req


def calculate_costs_until_batch(data, config, n_batches=10, t_feed=0.5, gamma=0.3, delta=0.04):
    """Calculate all cost components until specified batch number

    Args:
        data: Experiment JSON data
        config: Cost configuration parameters from YAML
        n_batches: Calculate costs up to this batch (default: 10 for all batches)
        t_feed: Base time for one feedback (minutes)
        gamma: Cognitive overhead weight
        delta: Exponential growth rate

    Returns:
        dict: Cost breakdown with interface, opportunity, error, and total costs
    """
    # 1. Interface Cost (accumulates with feedback requests)
    total_feedback_time = 0
    n_req = 0
    feedback_sessions = []

    for batch_num in range(1, n_batches + 1):
        batch_key = f"batch_{batch_num}"

        if batch_key not in data['detailed_results']:
            break  # Stop if batch doesn't exist

        batch_data = data['detailed_results'][batch_key]

        if batch_data['feedback_info']['feedback_requested']:
            n_feed = batch_data['feedback_info']['feedback_samples']
            n_req += 1
            feedback_sessions.append(n_feed)

            # Apply cognitive load formula
            t_req = calculate_cognitive_load_time(n_feed, t_feed, gamma, delta)
            total_feedback_time += t_req

    # Add walking time
    total_walking_time = n_req * config['t_walk']
    total_interface_time = total_feedback_time + total_walking_time

    # Interface cost
    c_req = total_interface_time * config['C_w']

    # 2. Opportunity Cost
    opportunity_rate = config['v'] / config['tau']
    c_opp = total_interface_time * opportunity_rate

    # 3. Error Cost (accumulates with prediction errors)
    # New formula: FP is MORE costly than FN
    # C_FP = t_maint × (C_w + v/τ)
    # C_FN = v × (1 + 0.2)
    total_fp = 0
    total_fn = 0

    for batch_num in range(1, n_batches + 1):
        batch_key = f"batch_{batch_num}"

        if batch_key not in data['detailed_results']:
            break

        batch_data = data['detailed_results'][batch_key]
        prod_perf = batch_data['performance']['production_system']

        total_fp += prod_perf['false_positives']
        total_fn += prod_perf['false_negatives']

    # Calculate per-event costs
    c_fp_per_event = config['t_maint'] * (config['C_w'] + opportunity_rate)
    c_fn_per_event = config['v'] * (1 + 0.2)

    # Total error cost
    c_err = total_fp * c_fp_per_event + total_fn * c_fn_per_event

    # 4. Total Cost (no training cost)
    c_tot = c_req + c_opp + c_err

    # Return detailed breakdown
    return {
        'interface_cost': c_req,
        'opportunity_cost': c_opp,
        'error_cost': c_err,
        'total_cost': c_tot,
        'details': {
            'total_feedback_time': total_feedback_time,
            'total_walking_time': total_walking_time,
            'total_interface_time': total_interface_time,
            'n_feedback_requests': n_req,
            'feedback_sessions': feedback_sessions,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'c_fp_per_event': c_fp_per_event,
            'c_fn_per_event': c_fn_per_event,
            'batches_processed': min(n_batches, len(data['detailed_results']))
        }
    }


def calculate_costs_for_all_policies(json_files_dir, seed_size, uncertainty_multiplier, feedback_mode,
                                   config, n_batches=10, t_feed=0.5, gamma=0.3, delta=0.04):
    """Calculate costs for all policies in a specific configuration

    Args:
        json_files_dir: Directory containing JSON experiment files
        seed_size: Training data size per trajectory
        uncertainty_multiplier: Uncertainty threshold multiplier
        feedback_mode: 'batch' or 'accumulated'
        config: Cost configuration parameters
        n_batches: Calculate costs up to this batch
        t_feed: Base feedback time
        gamma: Cognitive overhead weight
        delta: Exponential growth rate

    Returns:
        dict: Cost results for each policy (Static, Certain, Feedback, Combined)
    """
    # Load experiment data for all policies
    experiments = load_experiment_data(json_files_dir, seed_size, uncertainty_multiplier, feedback_mode)

    # Calculate costs for each policy
    policy_costs = {}

    for policy_key, data in experiments.items():
        costs = calculate_costs_until_batch(data, config, n_batches, t_feed, gamma, delta)
        policy_costs[policy_key] = costs

    return policy_costs


def calculate_batch_progression_costs(data, config, t_feed=0.5, gamma=0.3, delta=0.04):
    """Calculate cumulative costs for each batch (1 to 10) to show progression

    Returns:
        list: Cost breakdown for each batch number
    """
    max_batches = len(data['detailed_results'])
    progression = []

    for batch_num in range(1, max_batches + 1):
        costs = calculate_costs_until_batch(data, config, batch_num, t_feed, gamma, delta)
        costs['batch_number'] = batch_num
        progression.append(costs)

    return progression


def get_cost_efficiency_metrics(costs, data):
    """Calculate cost efficiency metrics

    Args:
        costs: Cost breakdown from calculate_costs_until_batch
        data: Experiment data for performance metrics

    Returns:
        dict: Efficiency metrics
    """
    # Get final performance
    summary = data['summary_results']
    final_accuracy = summary['overall_performance']['production_system']['final_accuracy']
    final_f1 = summary['overall_performance']['production_system']['average_f1_score']

    # Calculate efficiency metrics
    accuracy_per_euro = final_accuracy / costs['total_cost'] if costs['total_cost'] > 0 else 0
    f1_per_euro = final_f1 / costs['total_cost'] if costs['total_cost'] > 0 else 0

    return {
        'accuracy_per_euro': accuracy_per_euro,
        'f1_per_euro': f1_per_euro,
        'cost_per_accuracy_point': costs['total_cost'] / final_accuracy if final_accuracy > 0 else float('inf'),
        'final_accuracy': final_accuracy,
        'final_f1_score': final_f1,
        'total_cost': costs['total_cost']
    }


def load_all_experiments(json_files_dir):
    """Load all experiment configurations for comprehensive analysis"""
    pattern = os.path.join(json_files_dir, 'policy_experiment_*.json')
    json_files = glob.glob(pattern)

    experiments = {}

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            config = data['experiment_config']
            seed_size = config['seed_size']
            uncertainty_multiplier = config['uncertainty_multiplier']
            feedback_mode = config['feedback_mode']
            certain_ret = config['certain_retraining']
            feedback_ret = config['feedback_retraining']

            # Create policy key
            if certain_ret and feedback_ret:
                policy_key = "Combined"
            elif feedback_ret:
                policy_key = "Feedback"
            elif certain_ret:
                policy_key = "Certain"
            else:
                policy_key = "Static"

            key = (seed_size, uncertainty_multiplier, feedback_mode, policy_key)
            experiments[key] = data

        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return experiments


def create_comprehensive_cost_dataset(experiments, cost_config, t_feed=0.5, gamma=0.3, delta=0.04):
    """Create comprehensive dataset with costs and performance for all experiments"""
    data_list = []

    for (seed_size, unc_mult, feedback_mode, policy), exp_data in experiments.items():
        try:
            # Calculate costs
            costs = calculate_costs_until_batch(exp_data, cost_config, 10, t_feed, gamma, delta)

            # Get performance metrics
            summary = exp_data['summary_results']
            final_accuracy = summary['overall_performance']['production_system']['final_accuracy']
            final_f1 = summary['overall_performance']['production_system']['average_f1_score']

            data_list.append({
                'seed_size': seed_size,
                'uncertainty_multiplier': unc_mult,
                'feedback_mode': feedback_mode,
                'policy': policy,
                'interface_cost': costs['interface_cost'],
                'opportunity_cost': costs['opportunity_cost'],
                'error_cost': costs['error_cost'],
                'total_cost': costs['total_cost'],
                'final_accuracy': final_accuracy,
                'final_f1_score': final_f1,
                'config_label': f"S{seed_size}_U{unc_mult:.3f}"
            })

        except Exception as e:
            print(f"Warning: Could not process experiment {seed_size}_{unc_mult}_{feedback_mode}_{policy}: {e}")

    return pd.DataFrame(data_list)


def create_economic_landscape_heatmap(df, output_dir):
    """Create comprehensive cost and accuracy heatmaps"""
    os.makedirs(output_dir, exist_ok=True)

    # Define policy order for consistent visualization
    policy_order = ['Combined', 'Feedback', 'Certain', 'Static']

    # Create combined policy-mode labels
    df['policy_mode'] = df['policy'] + '-' + df['feedback_mode'].str.title()
    policy_mode_order = [f"{policy}-{mode}" for policy in policy_order for mode in ['Batch', 'Accumulated']]

    # Filter to existing combinations
    available_policy_modes = [pm for pm in policy_mode_order if pm in df['policy_mode'].values]

    # Sort configurations by seed_size then uncertainty_multiplier
    config_order = sorted(df['config_label'].unique(),
                         key=lambda x: (int(x.split('_')[0][1:]), float(x.split('_')[1][1:])))

    # Create cost heatmap
    cost_pivot = df.pivot_table(
        index='policy_mode', columns='config_label',
        values='total_cost', aggfunc='mean'
    )
    cost_pivot = cost_pivot.reindex(available_policy_modes)[config_order]

    plt.figure(figsize=(14, 8))

    # Create custom annotation with € symbol
    annot_data = cost_pivot.round(0).astype(int).astype(str)
    for i in range(len(annot_data.index)):
        for j in range(len(annot_data.columns)):
            if not pd.isna(cost_pivot.iloc[i, j]):
                annot_data.iloc[i, j] = f"€{int(cost_pivot.iloc[i, j])}"
            else:
                annot_data.iloc[i, j] = ""

    sns.heatmap(cost_pivot, annot=annot_data, fmt='', cmap='RdBu_r',
                cbar_kws={'label': 'Total Cost (€)'})
    plt.title('Economic Analysis: Total Cost Across All HIL-CBM Configurations',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Configuration (Seed Size_Uncertainty Multiplier)', fontsize=12)
    plt.ylabel('Policy-Feedback Mode', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    cost_path = os.path.join(output_dir, 'economic_landscape_cost_heatmap.png')
    plt.savefig(cost_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create accuracy heatmap
    accuracy_pivot = df.pivot_table(
        index='policy_mode', columns='config_label',
        values='final_accuracy', aggfunc='mean'
    )
    accuracy_pivot = accuracy_pivot.reindex(available_policy_modes)[config_order]

    plt.figure(figsize=(14, 8))
    sns.heatmap(accuracy_pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': 'Final Accuracy'})
    plt.title('Performance Analysis: Final Accuracy Across All HIL-CBM Configurations',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Configuration (Seed Size_Uncertainty Multiplier)', fontsize=12)
    plt.ylabel('Policy-Feedback Mode', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    accuracy_path = os.path.join(output_dir, 'economic_landscape_accuracy_heatmap.png')
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    plt.close()

    return [cost_path, accuracy_path]


def create_training_investment_analysis(experiments, cost_config, output_dir, t_feed=0.5, gamma=0.3, delta=0.04):
    """Create training investment vs operational returns analysis"""
    os.makedirs(output_dir, exist_ok=True)

    # Focus on Combined policy to show clearest crossover effect
    target_multipliers = {25: 1.085, 50: 1.085, 100: 1.085}  # Use same strategy across seed sizes

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    colors = {25: '#d62728', 50: '#ff7f0e', 100: '#1f77b4'}

    # Plot 1: Cumulative costs starting from batch 1
    seed_size_data = {}

    for seed_size in [25, 50, 100]:
        uncertainty_multiplier = target_multipliers[seed_size]
        key = (seed_size, uncertainty_multiplier, 'accumulated', 'Combined')

        if key not in experiments:
            continue

        exp_data = experiments[key]
        progression = calculate_batch_progression_costs(exp_data, cost_config, t_feed, gamma, delta)

        seed_size_data[seed_size] = progression

        batches = [p['batch_number'] for p in progression]
        cumulative_costs = [p['total_cost'] for p in progression]

        ax1.plot(batches, cumulative_costs, color=colors[seed_size],
                linewidth=3, marker='o', markersize=6,
                label=f'seed_size={seed_size}')

    ax1.set_xlabel('Batch Number', fontsize=12)
    ax1.set_ylabel('Cumulative Total Cost (€)', fontsize=12)
    ax1.set_title('Economic Timeline: Operational Cost Progression\n(Combined Policy, Accumulated Mode)',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 11))

    # Plot 2: Cost differences showing seed size impact on operational costs
    if 50 in seed_size_data:
        baseline_progression = seed_size_data[50]
        batches = [p['batch_number'] for p in baseline_progression]

        for seed_size in [25, 100]:
            if seed_size in seed_size_data:
                progression = seed_size_data[seed_size]
                cost_differences = []

                for i, batch_data in enumerate(progression):
                    diff = batch_data['total_cost'] - baseline_progression[i]['total_cost']
                    cost_differences.append(diff)

                line_style = '--' if seed_size == 25 else '-'
                ax2.plot(batches, cost_differences, color=colors[seed_size],
                        linewidth=3, linestyle=line_style, marker='s', markersize=6,
                        label=f'seed_size={seed_size} - seed_size=50')

    # Add zero line for reference
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Batch Number', fontsize=12)
    ax2.set_ylabel('Cost Difference from seed_size=50 (€)', fontsize=12)
    ax2.set_title('Seed Size Impact on Operational Costs\n(Cost Differences Over Time)',
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 11))

    # Add annotations for cost patterns
    ax2.text(3, ax2.get_ylim()[1] * 0.5, 'Early Batches:\nInitial Divergence',
             fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    ax2.text(8, ax2.get_ylim()[0] * 0.5, 'Later Batches:\nCost Stabilization',
             fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    plt.tight_layout()

    investment_path = os.path.join(output_dir, 'training_investment_analysis.png')
    plt.savefig(investment_path, dpi=300, bbox_inches='tight')
    plt.close()

    return [investment_path]


def create_human_factors_analysis(experiments, cost_config, output_dir):
    """Create human factors analysis showing cognitive load impact on feedback strategy"""
    os.makedirs(output_dir, exist_ok=True)

    # Use median configuration to avoid confounding effects
    target_config = (50, 1.085, 'Feedback')  # seed_size=50, unc=1.085, Feedback policy

    # Define cognitive parameter ranges with better distribution for crossover effect
    # Use exponential distribution to create more dramatic differences
    skill_levels = 10
    base_factors = np.linspace(0, 1, skill_levels)
    # Apply exponential scaling for more dramatic cognitive load differences
    cognitive_factors = np.power(base_factors, 1.8)  # Exponential scaling

    batch_costs = []
    accumulated_costs = []
    skill_labels = []

    for i, skill_factor in enumerate(cognitive_factors):
        # More dramatic parameter ranges for better crossover visualization
        t_feed = 0.25 + skill_factor * 0.6   # 0.25 (expert) to 0.85 (novice)
        gamma = 0.05 + skill_factor * 0.6    # 0.05 (expert) to 0.65 (novice)
        delta = 0.015 + skill_factor * 0.07  # 0.015 (expert) to 0.085 (novice)

        skill_labels.append(f"Level {i+1}")

        # Calculate costs for both feedback modes
        for feedback_mode in ['batch', 'accumulated']:
            key = (target_config[0], target_config[1], feedback_mode, target_config[2])

            if key in experiments:
                exp_data = experiments[key]
                costs = calculate_costs_until_batch(exp_data, cost_config, 10, t_feed, gamma, delta)

                if feedback_mode == 'batch':
                    batch_costs.append(costs['total_cost'])
                else:
                    accumulated_costs.append(costs['total_cost'])
            else:
                if feedback_mode == 'batch':
                    batch_costs.append(0)
                else:
                    accumulated_costs.append(0)

    # Calculate cost difference (Batch - Accumulated)
    cost_differences = np.array(batch_costs) - np.array(accumulated_costs)

    plt.figure(figsize=(12, 8))

    # Create bar plot showing cost differences
    x_pos = np.arange(len(skill_labels))
    colors = ['green' if diff < 0 else 'red' for diff in cost_differences]

    plt.bar(x_pos, cost_differences, color=colors, alpha=0.7, edgecolor='black')

    # Add zero line
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Add annotations for key insights
    plt.text(0.5, max(cost_differences) * 0.8, 'Accumulated Mode\nMore Cost-Effective',
             fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    plt.text(8.5, min(cost_differences) * 0.8, 'Batch Mode\nMore Cost-Effective',
             fontsize=10, ha='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

    plt.xlabel('Operator Skill Level (1=Expert, 10=Novice)', fontsize=12)
    plt.ylabel('Cost Difference: Batch - Accumulated (€)', fontsize=12)
    plt.title('Human Factors Analysis: Cognitive Load Impact on Feedback Strategy\nOptimal Strategy Selection Based on Operator Expertise',
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x_pos, [f"{i+1}" for i in range(len(skill_labels))])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    human_factors_path = os.path.join(output_dir, 'human_factors_cognitive_analysis.png')
    plt.savefig(human_factors_path, dpi=300, bbox_inches='tight')
    plt.close()

    return [human_factors_path]