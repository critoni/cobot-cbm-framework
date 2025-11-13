"""
Experiment 3: Policy Comparison Table Generation

Generates Excel tables showing policy performance metrics from experiment 3 results.
Creates two sheets: raw performance and improvement vs static baseline.
"""

import os
import glob
import json
import argparse
import yaml
import pandas as pd
import numpy as np


def load_config(config_path='config.yaml'):
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_experiment_results(json_files_dir):
    """Load all experiment 3 JSON result files"""
    pattern = os.path.join(json_files_dir, 'policy_experiment_*.json')
    json_files = glob.glob(pattern)

    if not json_files:
        raise ValueError(f"No experiment result files found in {json_files_dir}")

    results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        results.append(data)

    return results


def extract_performance_data(experiment_result):
    """Extract performance metrics from a single experiment result"""
    config = experiment_result['experiment_config']
    summary = experiment_result['summary_results']

    # Extract configuration parameters
    seed_size = config['seed_size']
    uncertainty_multiplier = config['uncertainty_multiplier']
    certain_retraining = config['certain_retraining']
    feedback_retraining = config['feedback_retraining']
    feedback_mode = config['feedback_mode']

    # Determine retraining strategy
    if certain_retraining and feedback_retraining:
        strategy = "Combined"
    elif feedback_retraining:
        strategy = "Feedback"
    elif certain_retraining:
        strategy = "Certain"
    else:
        strategy = "Static"

    # Determine feedback policy
    policy = "Accum" if feedback_mode == "accumulated" else "Batch"

    # Extract performance metrics from certain_autoencoder
    certain_perf = summary['overall_performance']['certain_autoencoder']
    feedback_analysis = summary['feedback_analysis']

    # Convert accuracy to percentage (it's stored as 0-1 float)
    final_accuracy = certain_perf['final_accuracy'] * 100
    mean_f1 = certain_perf['average_f1_score']
    avg_automation_rate = certain_perf['average_automation_rate']

    total_feedback_samples = feedback_analysis['total_feedback_samples']
    feedback_requests = feedback_analysis['total_feedback_requests']

    return {
        'Seed Size': seed_size,
        'Uncertainty Multiplier (UM)': uncertainty_multiplier,
        'Retraining Strategy': strategy,
        'Feedback Policy': policy,
        'Final Accuracy (%)': final_accuracy,
        'Mean F1 (%)': mean_f1,
        'Avg Automation Rate (%)': avg_automation_rate,
        'Total Feedback Samples': total_feedback_samples,
        'Feedback Requests': feedback_requests,
        'is_static': strategy == "Static"
    }


def create_raw_performance_table(all_data):
    """Create raw performance table"""
    df = pd.DataFrame(all_data)

    # Remove is_static helper column
    df = df.drop(columns=['is_static'])

    # Sort by seed size, UM, strategy, policy
    strategy_order = {'Static': 0, 'Certain': 1, 'Feedback': 2, 'Combined': 3}
    df['_strategy_order'] = df['Retraining Strategy'].map(strategy_order)

    df = df.sort_values(by=['Seed Size', 'Uncertainty Multiplier (UM)', '_strategy_order', 'Feedback Policy'])
    df = df.drop(columns=['_strategy_order'])
    df = df.reset_index(drop=True)

    return df


def create_improvement_table(all_data):
    """Create improvement vs static baseline table"""
    df = pd.DataFrame(all_data)

    # Find static baselines for each (seed_size, UM, policy) combination
    static_baselines = {}
    for _, row in df[df['is_static']].iterrows():
        key = (row['Seed Size'], row['Uncertainty Multiplier (UM)'], row['Feedback Policy'])
        static_baselines[key] = {
            'final_accuracy': row['Final Accuracy (%)'],
            'mean_f1': row['Mean F1 (%)'],
            'avg_automation_rate': row['Avg Automation Rate (%)'],
            'total_feedback': row['Total Feedback Samples'],
            'feedback_requests': row['Feedback Requests']
        }

    # Filter out static rows and calculate improvements
    df_improvement = df[~df['is_static']].copy()

    # Calculate deltas
    improvements = []
    for _, row in df_improvement.iterrows():
        key = (row['Seed Size'], row['Uncertainty Multiplier (UM)'], row['Feedback Policy'])

        if key in static_baselines:
            baseline = static_baselines[key]

            improvements.append({
                'Seed Size': row['Seed Size'],
                'Uncertainty Multiplier (UM)': row['Uncertainty Multiplier (UM)'],
                'Retraining Strategy': row['Retraining Strategy'],
                'Feedback Policy': row['Feedback Policy'],
                'Δ Final Accuracy (%)': row['Final Accuracy (%)'] - baseline['final_accuracy'],
                'Δ Mean F1 (%)': row['Mean F1 (%)'] - baseline['mean_f1'],
                'Δ Avg Automation Rate (%)': row['Avg Automation Rate (%)'] - baseline['avg_automation_rate'],
                'Δ Total Feedback Samples': row['Total Feedback Samples'] - baseline['total_feedback'],
                'Δ Feedback Requests': row['Feedback Requests'] - baseline['feedback_requests']
            })

    df_improvement = pd.DataFrame(improvements)

    # Sort by seed size, UM, strategy, policy
    strategy_order = {'Certain': 1, 'Feedback': 2, 'Combined': 3}
    df_improvement['_strategy_order'] = df_improvement['Retraining Strategy'].map(strategy_order)

    df_improvement = df_improvement.sort_values(by=['Seed Size', 'Uncertainty Multiplier (UM)', '_strategy_order', 'Feedback Policy'])
    df_improvement = df_improvement.drop(columns=['_strategy_order'])
    df_improvement = df_improvement.reset_index(drop=True)

    return df_improvement


def generate_experiment_3_tables(config):
    """Generate Excel tables for experiment 3 results"""
    print("=" * 60)
    print("EXPERIMENT 3: Policy Comparison Table Generation")
    print("=" * 60)

    # Load experiment results
    results_dir = config['paths']['results']
    json_files_dir = os.path.join(results_dir, 'experiment_3')
    output_dir = os.path.join(results_dir, 'experiment_3')

    print(f"\nLoading experiment results from: {json_files_dir}")
    experiment_results = load_experiment_results(json_files_dir)
    print(f"Loaded {len(experiment_results)} experiment configurations")

    # Extract performance data from all experiments
    print("\nExtracting performance metrics...")
    all_data = []
    for result in experiment_results:
        data = extract_performance_data(result)
        all_data.append(data)

    print(f"Extracted data from {len(all_data)} experiments")

    # Create raw performance table
    print("\nCreating raw performance table...")
    raw_table = create_raw_performance_table(all_data)
    print(f"Raw performance table: {len(raw_table)} rows")

    # Create improvement table
    print("\nCreating improvement vs static table...")
    improvement_table = create_improvement_table(all_data)
    print(f"Improvement table: {len(improvement_table)} rows")

    # Save to Excel with two sheets
    output_path = os.path.join(output_dir, 'experiment_3_policy_performance_tables.xlsx')

    print(f"\nSaving to Excel file: {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Raw Performance
        raw_table.to_excel(writer, sheet_name='Raw Performance', index=False)

        # Sheet 2: Improvement vs Static
        improvement_table.to_excel(writer, sheet_name='Improvement vs Static', index=False)

        # Format columns
        for sheet_name in ['Raw Performance', 'Improvement vs Static']:
            worksheet = writer.sheets[sheet_name]

            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 30)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    # Print summary
    print("\n" + "=" * 60)
    print("TABLE GENERATION COMPLETED")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print(f"\nSheet 1 - Raw Performance: {len(raw_table)} rows")
    print(f"  Includes all policies: Static, Certain, Feedback, Combined")
    print(f"\nSheet 2 - Improvement vs Static: {len(improvement_table)} rows")
    print(f"  Shows improvements for: Certain, Feedback, Combined policies")

    # Display sample from raw table
    print("\n" + "-" * 60)
    print("Sample from Raw Performance table (first 5 rows):")
    print("-" * 60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')
    print(raw_table.head())

    return output_path


def main():
    """Main table generation runner"""
    parser = argparse.ArgumentParser(description='Generate experiment 3 performance tables')
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

    # Generate tables
    generate_experiment_3_tables(config)


if __name__ == '__main__':
    main()
