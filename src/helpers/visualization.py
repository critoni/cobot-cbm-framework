"""
Visualization utilities for HIL-CBM framework experiments.
"""

import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns

def create_experiment_1_boxplot(results_csv_path, output_dir):
    """Create boxplot visualization for experiment 1 training scaling results"""

    # Read results
    df = pd.read_csv(results_csv_path)
    df['is_correct'] = (df['true_label'] == df['predicted_label']).astype(int)

    # Calculate batch accuracy
    batch_accuracy = df.groupby(['seed_size', 'batch']).agg({
        'is_correct': 'mean'
    }).reset_index()
    batch_accuracy.columns = ['seed_size', 'batch', 'accuracy']

    # Calculate overall accuracy per training size
    overall_accuracy = df.groupby('seed_size').agg({
        'is_correct': ['mean', 'std', 'count']
    }).reset_index()
    overall_accuracy.columns = ['seed_size', 'accuracy_mean', 'accuracy_std', 'sample_count']

    # Save overall accuracy summary
    summary_path = os.path.join(output_dir, 'training_scaling_summary.csv')
    overall_accuracy.to_csv(summary_path, index=False)

    # Prepare data for plotting
    seed_size_list = sorted(batch_accuracy['seed_size'].unique())
    data_for_plot = [batch_accuracy[batch_accuracy['seed_size'] == seed_size]['accuracy'].values
                     for seed_size in seed_size_list]

    # Create figure
    plt.figure(figsize=(12, 8))

    # Generate colors
    n_boxes = len(seed_size_list)
    if n_boxes <= 10:
        cmap = cm.get_cmap('tab10')
    else:
        cmap = cm.get_cmap('tab20')

    colors = [cmap(i / max(n_boxes-1, 1)) for i in range(n_boxes)]

    # Create boxplot
    box_plot = plt.boxplot(data_for_plot,
                          labels=seed_size_list,
                          patch_artist=True,
                          showfliers=True,
                          flierprops=dict(marker='o', markersize=3,
                                        alpha=0.6, markeredgewidth=0.5))

    # Apply colors to boxes and outliers
    for patch, flier, color in zip(box_plot['boxes'], box_plot['fliers'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    # Formatting
    plt.title('Accuracy by Training Data Count per Trajectory', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Training Data Count per Trajectory', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1)

    # Format y-axis as percentage
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'training_scaling_accuracy_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path, summary_path


def create_experiment_2_heatmaps(summary_csv_path, output_dir, base_multipliers):
    """Create heatmaps for experiment 2 uncertainty optimization results using matplotlib"""

    # Read results
    summary_df = pd.read_csv(summary_csv_path)

    # Get training sizes
    training_sizes = sorted(summary_df['seed_size'].unique())

    # Filter to only specified base multipliers
    summary_filtered = summary_df[summary_df['base_multiplier'].isin(base_multipliers)]

    output_paths = []

    # Create heatmaps for each training size
    for seed_size in training_sizes:
        seed_size_data = summary_filtered[summary_filtered['seed_size'] == seed_size]

        if len(seed_size_data) == 0:
            continue

        # Prepare data for heatmaps
        config_df = seed_size_data[['base_multiplier', 'uncertainty_multiplier', 'error_catch_rate', 'workload_efficiency']].copy()

        if len(config_df) == 0:
            continue

        # Replace values where no uncertain samples exist with NaN
        summary_with_unc = seed_size_data[['base_multiplier', 'uncertainty_multiplier', 'uncertain_samples']]

        config_df = seed_size_data[['base_multiplier', 'uncertainty_multiplier',
                            'error_catch_rate', 'workload_efficiency']].copy()

        # Mask metrics where uncertain_samples == 0
        mask = summary_with_unc['uncertain_samples'] == 0
        config_df.loc[mask, ['error_catch_rate', 'workload_efficiency']] = np.nan

        # Heatmap 1: Error Catch Rate
        heatmap_errcatch = config_df.pivot(
            index='uncertainty_multiplier',
            columns='base_multiplier',
            values='error_catch_rate'
        ).sort_index(ascending=False)

        # Calculate figure size based on data dimensions
        n_base = len(heatmap_errcatch.columns)
        n_unc = len(heatmap_errcatch.index)
        fig_width = max(6, n_base * 1.5)
        fig_height = max(8, n_unc * 0.3)

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            heatmap_errcatch,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            cbar_kws={'label': 'Error Catch Rate (%)'},
            square=False,
            linewidths=0.5,
            linecolor='white'
        )
        plt.title(f'Error Catch Rate - Training Data per Trajectory = {seed_size}',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Base Multiplier', fontsize=12)
        plt.ylabel('Uncertainty Multiplier', fontsize=12)

        # Format axis ticks
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        # Save Error Catch Rate heatmap
        os.makedirs(output_dir, exist_ok=True)
        errcatch_path = os.path.join(output_dir, f'error_catch_rate_seed_size_{seed_size}.png')
        plt.savefig(errcatch_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths.append(errcatch_path)

        # Heatmap 2: Workload Efficiency
        heatmap_eff = config_df.pivot(
            index='uncertainty_multiplier',
            columns='base_multiplier',
            values='workload_efficiency'
        ).sort_index(ascending=False)

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            heatmap_eff,
            annot=True,
            fmt='.1f',
            cmap='viridis',
            cbar_kws={'label': 'Workload Efficiency (%)'},
            square=False,
            linewidths=0.5,
            linecolor='white'
        )
        plt.title(f'Workload Efficiency - Training Data per Trajectory = {seed_size}',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Base Multiplier', fontsize=12)
        plt.ylabel('Uncertainty Multiplier', fontsize=12)

        # Format axis ticks
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        # Save Workload Efficiency heatmap
        eff_path = os.path.join(output_dir, f'workload_efficiency_seed_size_{seed_size}.png')
        plt.savefig(eff_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths.append(eff_path)

    return output_paths

def create_experiment_2_rankplots(summary_csv_path, output_dir, base_multiplier_focus=1.0,
                                  eff_min=60.0, ecr_min=85.0):
    """Create ranking plots for uncertainty multiplier selection analysis"""
    from .selection import get_multiplier_selections

    df = pd.read_csv(summary_csv_path)
    seed_size_list = sorted(df['seed_size'].unique())
    os.makedirs(output_dir, exist_ok=True)

    selections = []
    output_paths = []

    for seed_size in seed_size_list:
        selection_result = get_multiplier_selections(df, seed_size, base_multiplier_focus, eff_min, ecr_min)
        if selection_result is None:
            continue

        plot_data = selection_result['all_data']
        sel = selection_result['selections']

        x = plot_data['uncertainty_multiplier'].values
        y_err = plot_data['ErrRank'].values
        y_eff = plot_data['EffRank'].values

        plt.figure(figsize=(11, 5.5))
        plt.plot(x, y_err, marker='o', label='ErrRank (Error Catch Rate)')
        plt.plot(x, y_eff, marker='o', label='EffRank (Workload Efficiency)')

        chosen_points = [
            sel['best_err']['uncertainty_multiplier'],
            sel['best_eff']['uncertainty_multiplier'],
            sel['best_balance']['uncertainty_multiplier']
        ]
        if len(x) >= 2:
            steps = np.diff(x)
            band = float(np.min(steps)) / 2.0 if np.all(steps > 0) else 0.01
        else:
            band = 0.01
        for xm in chosen_points:
            plt.axvspan(xm - band, xm + band, color='red', alpha=0.12)

        plt.xticks(ticks=x, labels=[f"{val}" for val in x], rotation=45, ha='right')
        plt.title(
            f'Loss vs Uncertainty Multiplier (seed_size={seed_size}, base={base_multiplier_focus})\n'
            f'Best-Err: {sel["best_err"]["uncertainty_multiplier"]}, '
            f'Best-Eff: {sel["best_eff"]["uncertainty_multiplier"]}, '
            f'Best-Balance: {sel["best_balance"]["uncertainty_multiplier"]}'
        )
        plt.xlabel('Uncertainty Multiplier')
        plt.ylabel('Loss (0 = best, 1 = worst)')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(output_dir,
            f'rankplot_seed_size_{seed_size}_base_{base_multiplier_focus:.2f}_2lines.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths.append(out_path)

        selections.append({
            'seed_size': selection_result['seed_size'],
            'best_err_um': sel['best_err']['uncertainty_multiplier'],
            'best_eff_um': sel['best_eff']['uncertainty_multiplier'],
            'best_balance_um': sel['best_balance']['uncertainty_multiplier'],
            'best_err_loss': sel['best_err']['err_rank'],
            'best_eff_loss': sel['best_eff']['eff_rank'],
            'best_balance_err_loss': sel['best_balance']['err_rank'],
            'best_balance_eff_loss': sel['best_balance']['eff_rank'],
        })

    if selections:
        sel_df = pd.DataFrame(selections).sort_values('seed_size')
        sel_path = os.path.join(output_dir,
            f'selected_multipliers_base_{base_multiplier_focus:.2f}.csv')
        sel_df.to_csv(sel_path, index=False)
        output_paths.append(sel_path)

    return output_paths


def create_experiment_3_policy_comparison(json_files_dir, seed_size, uncertainty_multiplier, feedback_mode, output_dir):
    """Create policy comparison visualization showing temporal performance evolution

    Displays three metrics over batches for different retraining policies:
    - Cumulative certain sample improvement vs static baseline
    - Cumulative feedback reduction vs static baseline
    - Cumulative correct prediction improvement vs static baseline
    """

    unc_str = f"{uncertainty_multiplier:.3f}"
    pattern = os.path.join(json_files_dir, f'policy_experiment_seed_size{seed_size}_unc{unc_str}_*_feedtype{feedback_mode}_*.json')
    json_files = glob.glob(pattern)

    if not json_files:
        print(f"No JSON files found for seed_size {seed_size} and uncertainty_multiplier {uncertainty_multiplier}")
        return []

    policy_data = []
    static_baseline = None

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        config = data['experiment_config']
        certain_ret = config['certain_retraining']
        feedback_ret = config['feedback_retraining']

        if certain_ret and feedback_ret:
            policy_name = "Combined"
        elif feedback_ret:
            policy_name = "Feedback"
        elif certain_ret:
            policy_name = "Certain"
        else:
            policy_name = "Static"

        is_static = not certain_ret and not feedback_ret

        batch_numbers = []
        certain_samples = []
        feedback_samples_per_batch = []
        correct_certain = []

        detailed_results = data['detailed_results']
        for batch_key in sorted(detailed_results.keys(), key=lambda x: int(x.split('_')[1])):
            batch_data = detailed_results[batch_key]
            batch_numbers.append(int(batch_key.split('_')[1]))

            certain_count = batch_data['performance']['certain_autoencoder']['certain_samples']
            production_total = batch_data['performance']['production_system']['total_count']
            feedback_this_batch = production_total - certain_count
            correct_count = batch_data['performance']['certain_autoencoder']['correct_count']

            certain_samples.append(certain_count)
            feedback_samples_per_batch.append(feedback_this_batch)
            correct_certain.append(correct_count)

        cumulative_certain = np.cumsum(certain_samples)
        cumulative_feedback = np.cumsum(feedback_samples_per_batch)
        cumulative_correct = np.cumsum(correct_certain)

        policy_info = {
            'policy_name': policy_name,
            'batch_numbers': batch_numbers,
            'cumulative_certain': cumulative_certain,
            'feedback_cumulative': cumulative_feedback,
            'cumulative_correct': cumulative_correct,
            'is_static': is_static
        }

        if is_static:
            static_baseline = policy_info
        else:
            policy_data.append(policy_info)

    if static_baseline is None:
        print("Warning: No static baseline found (C:False F:False)")
        return []

    policy_order = ["Certain", "Feedback", "Combined"]
    policy_data_sorted = sorted(policy_data, key=lambda x: policy_order.index(x['policy_name']) if x['policy_name'] in policy_order else len(policy_order))

    for policy in policy_data_sorted:
        policy['certain_difference'] = policy['cumulative_certain'] - static_baseline['cumulative_certain']
        policy['feedback_difference'] = [policy_fb - static_fb
                                       for policy_fb, static_fb
                                       in zip(policy['feedback_cumulative'], static_baseline['feedback_cumulative'])]
        policy['correct_difference'] = policy['cumulative_correct'] - static_baseline['cumulative_correct']

    _, axes = plt.subplots(1, 3, figsize=(12, 6))

    policy_colors = {
        "Certain": plt.cm.tab10(0),
        "Feedback": plt.cm.tab10(1),
        "Combined": plt.cm.tab10(2)
    }

    axes[0].axhline(y=0, color='red', linestyle='-', linewidth=2, label='Static Baseline', alpha=0.8)
    for policy in policy_data_sorted:
        axes[0].plot(policy['batch_numbers'], policy['certain_difference'],
                    marker='o', label=policy['policy_name'], color=policy_colors[policy['policy_name']], linewidth=2)
    axes[0].set_title('Cumulative Automated Sample Count vs Static Baseline', fontweight='bold', fontsize=8)
    axes[0].set_xlabel('Batch Number')
    axes[0].set_ylabel('Additional Automated Samples')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_xticks(range(1, max(static_baseline['batch_numbers']) + 1))

    axes[1].axhline(y=0, color='red', linestyle='-', linewidth=2, label='Static Baseline', alpha=0.8)
    for policy in policy_data_sorted:
        axes[1].plot(policy['batch_numbers'], policy['feedback_difference'],
                    marker='s', label=policy['policy_name'], color=policy_colors[policy['policy_name']], linewidth=2)
    axes[1].set_title('Cumulative Human Feedback Demand vs Static Baseline', fontweight='bold', fontsize=8)
    axes[1].set_xlabel('Batch Number')
    axes[1].set_ylabel('Human Feedback Sample Difference')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xticks(range(1, max(static_baseline['batch_numbers']) + 1))

    axes[2].axhline(y=0, color='red', linestyle='-', linewidth=2, label='Static Baseline', alpha=0.8)
    for policy in policy_data_sorted:
        axes[2].plot(policy['batch_numbers'], policy['correct_difference'],
                    marker='^', label=policy['policy_name'], color=policy_colors[policy['policy_name']], linewidth=2)
    axes[2].set_title('Cumulative Correct Automated Predictions vs Static Baseline', fontweight='bold', fontsize=8)
    axes[2].set_xlabel('Batch Number')
    axes[2].set_ylabel('Additional Correct Automated Predictions')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xticks(range(1, max(static_baseline['batch_numbers']) + 1))

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'policy_comparison_seed_size_{seed_size}_unc_{uncertainty_multiplier}_mode_{feedback_mode}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Policy comparison plot saved to: {output_path}")
    print(f"Policies compared: {len(policy_data_sorted)} + static baseline")

    return [output_path]


def create_experiment_3_overall_performance(json_files_dir, output_dir):
    """Create comprehensive performance analysis across all experiment configurations

    Generates two complementary visualizations:
    - Performance heatmap showing final improvements by policy and setting
    - Average performance comparison across different parameter combinations
    """

    # Load all experiment result files for comprehensive analysis
    pattern = os.path.join(json_files_dir, 'policy_experiment_*.json')
    json_files = glob.glob(pattern)

    if not json_files:
        print(f"No JSON files found in {json_files_dir}")
        return []

    # Collect performance data across all configurations
    data_matrix = []
    settings_info = []

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        config = data['experiment_config']
        seed_size = config['seed_size']
        uncertainty_multiplier = config['uncertainty_multiplier']
        feedback_mode = config['feedback_mode']
        certain_ret = config['certain_retraining']
        feedback_ret = config['feedback_retraining']

        if certain_ret and feedback_ret:
            policy_name = "Combined"
        elif feedback_ret:
            policy_name = "Feedback"
        elif certain_ret:
            policy_name = "Certain"
        else:
            policy_name = "Static"

        setting_key = f"S{seed_size}_U{uncertainty_multiplier:.3f}"
        policy_key = f"{policy_name}-{feedback_mode.title()}"

        final_batch_key = max(data['detailed_results'].keys(), key=lambda x: int(x.split('_')[1]))

        certain_samples = []
        feedback_samples = []
        correct_counts = []
        for i in range(1, int(final_batch_key.split('_')[1]) + 1):
            batch_data = data['detailed_results'][f'batch_{i}']
            certain_count = batch_data['performance']['certain_autoencoder']['certain_samples']
            production_total = batch_data['performance']['production_system']['total_count']
            feedback_this_batch = production_total - certain_count
            correct_count = batch_data['performance']['certain_autoencoder']['correct_count']

            certain_samples.append(certain_count)
            feedback_samples.append(feedback_this_batch)
            correct_counts.append(correct_count)

        certain_cumulative = sum(certain_samples)
        feedback_cumulative = sum(feedback_samples)
        correct_cumulative = sum(correct_counts)

        data_matrix.append({
            'setting_key': setting_key,
            'policy_key': policy_key,
            'seed_size': seed_size,
            'uncertainty_multiplier': uncertainty_multiplier,
            'feedback_mode': feedback_mode,
            'policy_name': policy_name,
            'certain_cumulative': certain_cumulative,
            'feedback_cumulative': feedback_cumulative,
            'correct_cumulative': correct_cumulative
        })

        if setting_key not in [s['setting_key'] for s in settings_info]:
            settings_info.append({
                'setting_key': setting_key,
                'seed_size': seed_size,
                'uncertainty_multiplier': uncertainty_multiplier
            })

    static_baselines = {}
    for entry in data_matrix:
        if entry['policy_name'] == 'Static':
            key = f"{entry['setting_key']}_{entry['feedback_mode']}"
            static_baselines[key] = entry

    performance_data = []
    for entry in data_matrix:
        if entry['policy_name'] != 'Static':
            baseline_key = f"{entry['setting_key']}_{entry['feedback_mode']}"
            if baseline_key in static_baselines:
                baseline = static_baselines[baseline_key]
                certain_diff = entry['certain_cumulative'] - baseline['certain_cumulative']
                feedback_diff = entry['feedback_cumulative'] - baseline['feedback_cumulative']
                correct_diff = entry['correct_cumulative'] - baseline['correct_cumulative']

                performance_data.append({
                    'setting_key': entry['setting_key'],
                    'policy_key': entry['policy_key'],
                    'certain_improvement': certain_diff,
                    'feedback_reduction': -feedback_diff,
                    'correct_improvement': correct_diff
                })

    settings_sorted = sorted(settings_info, key=lambda x: (x['seed_size'], x['uncertainty_multiplier']))
    # Define policy order for consistent visualization (top to bottom in heatmap)
    policy_order = ['Combined-Batch', 'Combined-Accumulated', 'Feedback-Batch', 'Feedback-Accumulated', 'Certain-Batch', 'Certain-Accumulated']

    # Construct performance matrix for heatmap visualization
    heatmap_data = np.zeros((len(policy_order), len(settings_sorted)))
    heatmap_data[:] = np.nan

    for i, policy in enumerate(policy_order):
        for j, setting in enumerate(settings_sorted):
            matches = [p for p in performance_data if p['setting_key'] == setting['setting_key'] and p['policy_key'] == policy]
            if matches:
                heatmap_data[i, j] = matches[0]['certain_improvement']

    # Generate dual visualization: heatmap and bar chart
    _, axes = plt.subplots(1, 2, figsize=(16, 8))

    im = axes[0].imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
    axes[0].set_xticks(range(len(settings_sorted)))
    axes[0].set_xticklabels([s['setting_key'] for s in settings_sorted], rotation=45, ha='right')
    axes[0].set_yticks(range(len(policy_order)))
    axes[0].set_yticklabels(policy_order)
    axes[0].set_title('Final Automated Sample Count Improvement vs Static Baseline', fontweight='bold')

    for i in range(len(policy_order)):
        for j in range(len(settings_sorted)):
            if not np.isnan(heatmap_data[i, j]):
                axes[0].text(j, i, f'{int(heatmap_data[i, j])}', ha='center', va='center', fontsize=8)

    plt.colorbar(im, ax=axes[0], label='Additional Automated Samples')

    bar_data = {}
    for setting in settings_sorted:
        setting_matches = [p for p in performance_data if p['setting_key'] == setting['setting_key']]
        if setting_matches:
            avg_certain = np.mean([p['certain_improvement'] for p in setting_matches])
            avg_feedback = np.mean([p['feedback_reduction'] for p in setting_matches])
            avg_correct = np.mean([p['correct_improvement'] for p in setting_matches])
            bar_data[setting['setting_key']] = {
                'automated': avg_certain,
                'feedback': avg_feedback,
                'correct': avg_correct
            }

    x_pos = np.arange(len(bar_data))
    automated_values = [bar_data[k]['automated'] for k in bar_data.keys()]
    feedback_values = [bar_data[k]['feedback'] for k in bar_data.keys()]
    correct_values = [bar_data[k]['correct'] for k in bar_data.keys()]

    width = 0.25
    axes[1].bar(x_pos - width, automated_values, width, label='Automated Sample Improvement', color='steelblue')
    axes[1].bar(x_pos, feedback_values, width, label='Human Feedback Reduction', color='orange')
    axes[1].bar(x_pos + width, correct_values, width, label='Correct Automated Improvement', color='green')

    axes[1].set_xlabel('Settings')
    axes[1].set_ylabel('Average Improvement')
    axes[1].set_title('Average Performance Improvement by Parameter Setting', fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(list(bar_data.keys()), rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'experiment_3_overall_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Overall performance plot saved to: {output_path}")
    print(f"Analyzed {len(performance_data)} policy combinations across {len(settings_sorted)} settings")

    return [output_path]
