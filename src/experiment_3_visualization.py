"""
Experiment 3: Policy Comparison Visualization

Generates comprehensive visualizations for HIL-CBM policy comparison results
including temporal evolution plots and overall performance analysis.
"""

import os
import glob
import re
import argparse
import yaml

from helpers.visualization import create_experiment_3_policy_comparison, create_experiment_3_overall_performance


def load_config(config_path='config.yaml'):
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_parameter_combinations(json_files_dir):
    """Extract unique parameter combinations from JSON files"""
    pattern = os.path.join(json_files_dir, 'policy_experiment_*.json')
    json_files = glob.glob(pattern)

    combinations = set()
    for filepath in json_files:
        filename = os.path.basename(filepath)

        # Extract parameters using regex
        seed_match = re.search(r'seed_size(\d+)', filename)
        unc_match = re.search(r'unc([0-9.]+)', filename)
        mode_match = re.search(r'feedtype(\w+)_threshold', filename)

        if seed_match and unc_match and mode_match:
            seed_size = int(seed_match.group(1))
            uncertainty_multiplier = float(unc_match.group(1))
            feedback_mode = mode_match.group(1)  # This will be 'batch' or 'accumulated'
            combinations.add((seed_size, uncertainty_multiplier, feedback_mode))

    return sorted(combinations)


def generate_policy_comparison_plots(config):
    """Generate policy comparison plots for all parameter combinations"""
    results_dir = config['paths']['results']
    json_files_dir = os.path.join(results_dir, 'experiment_3')
    base_output_dir = os.path.join(results_dir, 'experiment_3', 'plots')

    if not os.path.exists(json_files_dir):
        print(f"Experiment 3 results directory not found: {json_files_dir}")
        return []

    # Extract parameter combinations
    combinations = extract_parameter_combinations(json_files_dir)

    if not combinations:
        print("No valid parameter combinations found in JSON files")
        return []

    print(f"Found {len(combinations)} parameter combinations")

    # Generate plots for each combination
    plot_paths = []
    for seed_size, uncertainty_multiplier, feedback_mode in combinations:
        print(f"Generating plot for seed_size={seed_size}, unc={uncertainty_multiplier:.3f}, mode={feedback_mode}")

        # Create seed_size specific output directory
        output_dir = os.path.join(base_output_dir, f'seed_size_{seed_size}')

        paths = create_experiment_3_policy_comparison(
            json_files_dir, seed_size, uncertainty_multiplier, feedback_mode, output_dir
        )
        plot_paths.extend(paths)

    return plot_paths


def generate_overall_performance_plot(config):
    """Generate overall performance analysis plot"""
    results_dir = config['paths']['results']
    json_files_dir = os.path.join(results_dir, 'experiment_3')
    output_dir = os.path.join(results_dir, 'experiment_3', 'plots')

    if not os.path.exists(json_files_dir):
        print(f"Experiment 3 results directory not found: {json_files_dir}")
        return []

    print("Generating overall performance analysis...")
    plot_paths = create_experiment_3_overall_performance(json_files_dir, output_dir)

    return plot_paths


def run_experiment_3_visualization(config):
    """Execute complete experiment 3 visualization analysis"""
    print("=" * 60)
    print("EXPERIMENT 3 VISUALIZATION: Policy Comparison Analysis")
    print("=" * 60)

    # Generate policy comparison plots
    print("\n1. Generating policy comparison plots...")
    policy_plots = generate_policy_comparison_plots(config)

    # Generate overall performance plot
    print("\n2. Generating overall performance analysis...")
    overall_plots = generate_overall_performance_plot(config)

    # Summary
    all_plots = policy_plots + overall_plots

    print("\n" + "=" * 60)
    print("EXPERIMENT 3 VISUALIZATION COMPLETED")
    print("=" * 60)
    print(f"Total plots generated: {len(all_plots)}")
    print(f"Policy comparison plots: {len(policy_plots)}")
    print(f"Overall performance plots: {len(overall_plots)}")

    if all_plots:
        print("\nGenerated files:")
        for plot_path in all_plots:
            print(f"  {plot_path}")

    return all_plots


def main():
    """Main visualization runner"""
    parser = argparse.ArgumentParser(description='Generate experiment 3 visualizations')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Run visualization
    run_experiment_3_visualization(config)


if __name__ == '__main__':
    main()