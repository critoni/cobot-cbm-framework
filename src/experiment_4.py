"""
Experiment 4: Economic Analysis of HIL-CBM Systems

Analyzes the economic costs and benefits of human-in-the-loop condition-based
monitoring systems using cognitive load-based cost models.
"""

import os
import time
import argparse
import yaml
import pandas as pd

from helpers.cost_analysis import (
    load_all_experiments, create_comprehensive_cost_dataset,
    create_economic_landscape_heatmap, create_training_investment_analysis,
    create_human_factors_analysis
)


def load_config(config_path='config.yaml'):
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_economic_analysis_experiment(config):
    """Execute comprehensive economic analysis experiment"""
    print("=" * 60)
    print("EXPERIMENT 4: Economic Analysis of HIL-CBM Systems")
    print("=" * 60)

    start_time = time.time()

    # Load configuration
    cost_config = config['experiment_4']['costs']
    results_dir = config['paths']['results']
    json_files_dir = os.path.join(results_dir, 'experiment_3')
    output_dir = os.path.join(results_dir, 'experiment_4')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Default cognitive parameters
    default_cognitive_params = {
        't_feed': 0.5,      # minutes per feedback
        'gamma': 0.3,       # cognitive overhead weight
        'delta': 0.04       # exponential growth rate
    }

    print(f"Input directory: {json_files_dir}")
    print(f"Output directory: {output_dir}")
    print("Default cognitive parameters:")
    for param, value in default_cognitive_params.items():
        print(f"  {param}: {value}")

    # Load all experiment data
    print("\nLoading experiment data...")
    experiments = load_all_experiments(json_files_dir)

    if not experiments:
        raise ValueError(f"No experiment data found in {json_files_dir}")

    print(f"Loaded {len(experiments)} experiment configurations")

    # Create comprehensive cost dataset
    print("\nCreating comprehensive cost dataset...")
    df = create_comprehensive_cost_dataset(
        experiments, cost_config,
        default_cognitive_params['t_feed'],
        default_cognitive_params['gamma'],
        default_cognitive_params['delta']
    )

    if df.empty:
        raise ValueError("No valid cost data could be generated")

    print(f"Generated cost data for {len(df)} configurations")

    # Save comprehensive dataset
    dataset_path = os.path.join(output_dir, 'comprehensive_cost_dataset.csv')
    df.to_csv(dataset_path, index=False)
    print(f"Comprehensive dataset saved to: {dataset_path}")

    # Analysis 1: Economic Landscape Overview
    print("\n1. Generating economic landscape heatmaps...")
    heatmap_paths = create_economic_landscape_heatmap(df, output_dir)

    # Analysis 2: Training Investment Analysis
    print("\n2. Generating training investment analysis...")
    investment_paths = create_training_investment_analysis(
        experiments, cost_config, output_dir,
        default_cognitive_params['t_feed'],
        default_cognitive_params['gamma'],
        default_cognitive_params['delta']
    )

    # Analysis 3: Human Factors Analysis
    print("\n3. Generating human factors analysis...")
    human_factors_paths = create_human_factors_analysis(experiments, cost_config, output_dir)

    # Generate summary statistics
    print("\n4. Generating summary statistics...")
    summary_stats = generate_summary_statistics(df, output_dir)

    # Collect all output paths
    all_paths = heatmap_paths + investment_paths + human_factors_paths + [summary_stats]

    # Print final summary
    duration = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXPERIMENT 4 COMPLETED")
    print("=" * 60)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Configurations analyzed: {len(df)}")
    print(f"Total visualizations: {len(all_paths) - 1}")  # Subtract summary file
    print(f"Results saved to: {output_dir}")

    print("\nGenerated files:")
    for path in all_paths:
        print(f"  {path}")

    return all_paths


def generate_summary_statistics(df, output_dir):
    """Generate and save summary statistics"""

    # Overall statistics
    overall_stats = {
        'Total Configurations': len(df),
        'Unique Policies': df['policy'].nunique(),
        'Unique Seed Sizes': df['seed_size'].nunique(),
        'Unique Uncertainty Multipliers': df['uncertainty_multiplier'].nunique(),
        'Feedback Modes': df['feedback_mode'].nunique()
    }

    # Cost statistics
    cost_stats = {
        'Mean Total Cost': f"€{df['total_cost'].mean():.2f}",
        'Median Total Cost': f"€{df['total_cost'].median():.2f}",
        'Min Total Cost': f"€{df['total_cost'].min():.2f}",
        'Max Total Cost': f"€{df['total_cost'].max():.2f}",
        'Cost Standard Deviation': f"€{df['total_cost'].std():.2f}"
    }

    # Performance statistics
    performance_stats = {
        'Mean Final Accuracy': f"{df['final_accuracy'].mean():.3f}",
        'Median Final Accuracy': f"{df['final_accuracy'].median():.3f}",
        'Min Final Accuracy': f"{df['final_accuracy'].min():.3f}",
        'Max Final Accuracy': f"{df['final_accuracy'].max():.3f}",
        'Accuracy Standard Deviation': f"{df['final_accuracy'].std():.3f}"
    }

    # Best configurations
    best_cost_idx = df['total_cost'].idxmin()
    best_accuracy_idx = df['final_accuracy'].idxmax()

    # Calculate efficiency metric (accuracy per euro)
    df['efficiency'] = df['final_accuracy'] / df['total_cost']
    best_efficiency_idx = df['efficiency'].idxmax()

    best_configs = {
        'Lowest Cost Configuration': {
            'Cost': f"€{df.loc[best_cost_idx, 'total_cost']:.2f}",
            'Accuracy': f"{df.loc[best_cost_idx, 'final_accuracy']:.3f}",
            'Policy': f"{df.loc[best_cost_idx, 'policy']}-{df.loc[best_cost_idx, 'feedback_mode'].title()}",
            'Configuration': df.loc[best_cost_idx, 'config_label']
        },
        'Highest Accuracy Configuration': {
            'Cost': f"€{df.loc[best_accuracy_idx, 'total_cost']:.2f}",
            'Accuracy': f"{df.loc[best_accuracy_idx, 'final_accuracy']:.3f}",
            'Policy': f"{df.loc[best_accuracy_idx, 'policy']}-{df.loc[best_accuracy_idx, 'feedback_mode'].title()}",
            'Configuration': df.loc[best_accuracy_idx, 'config_label']
        },
        'Most Efficient Configuration': {
            'Cost': f"€{df.loc[best_efficiency_idx, 'total_cost']:.2f}",
            'Accuracy': f"{df.loc[best_efficiency_idx, 'final_accuracy']:.3f}",
            'Efficiency': f"{df.loc[best_efficiency_idx, 'efficiency']:.5f} accuracy/€",
            'Policy': f"{df.loc[best_efficiency_idx, 'policy']}-{df.loc[best_efficiency_idx, 'feedback_mode'].title()}",
            'Configuration': df.loc[best_efficiency_idx, 'config_label']
        }
    }

    # Policy comparison
    policy_comparison = df.groupby(['policy', 'feedback_mode']).agg({
        'total_cost': ['mean', 'std'],
        'final_accuracy': ['mean', 'std'],
        'efficiency': ['mean', 'std']
    }).round(3)

    # Create summary report
    summary_report = f"""
EXPERIMENT 4: ECONOMIC ANALYSIS SUMMARY REPORT
===============================================

OVERALL STATISTICS
{'-' * 20}
"""
    for key, value in overall_stats.items():
        summary_report += f"{key}: {value}\n"

    summary_report += f"""
COST ANALYSIS
{'-' * 20}
"""
    for key, value in cost_stats.items():
        summary_report += f"{key}: {value}\n"

    summary_report += f"""
PERFORMANCE ANALYSIS
{'-' * 20}
"""
    for key, value in performance_stats.items():
        summary_report += f"{key}: {value}\n"

    summary_report += f"""
BEST CONFIGURATIONS
{'-' * 20}
"""
    for config_type, config_data in best_configs.items():
        summary_report += f"\n{config_type}:\n"
        for key, value in config_data.items():
            summary_report += f"  {key}: {value}\n"

    summary_report += f"""
POLICY COMPARISON
{'-' * 20}
{policy_comparison.to_string()}

RESEARCH INSIGHTS
{'-' * 20}
1. Economic Landscape: Total cost varies from €{df['total_cost'].min():.2f} to €{df['total_cost'].max():.2f}
2. Performance Range: Final accuracy varies from {df['final_accuracy'].min():.3f} to {df['final_accuracy'].max():.3f}
3. Cost-Efficiency Trade-off: Best efficiency = {df['efficiency'].max():.5f} accuracy/€
4. Training Investment Impact: Visible in training investment analysis plot
5. Human Factors: Cognitive load parameters significantly affect feedback strategy costs
"""

    # Save summary report
    summary_path = os.path.join(output_dir, 'economic_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_report)

    print(f"Summary statistics saved to: {summary_path}")
    return summary_path


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description='Run economic analysis experiment')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Check if experiment 3 results exist
    results_dir = config['paths']['results']
    json_files_dir = os.path.join(results_dir, 'experiment_3')

    if not os.path.exists(json_files_dir):
        print(f"Error: Experiment 3 results not found in {json_files_dir}")
        print("Please run experiment 3 first to generate the required data.")
        return

    # Run experiment
    run_economic_analysis_experiment(config)


if __name__ == '__main__':
    main()