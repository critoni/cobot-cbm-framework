#!/bin/bash

# Experiment 3 Policy Comparison Grid Runner
# Runs all combinations of parameters systematically using experiment 2 selections

# Configuration grid
declare -a SEED_SIZE_VALUES=(25 50 100)
declare -a SELECTION_STRATEGIES=(best_err best_eff best_balance)
declare -a CERTAIN_RETRAINING_VALUES=(true false)
declare -a FEEDBACK_RETRAINING_VALUES=(true false)
declare -a FEEDBACK_MODE_VALUES=(batch accumulated)
declare -a FEEDBACK_THRESHOLD_VALUES=(30)

# Function to run a single experiment
run_experiment() {
    local seed_size=$1
    local selection_strategy=$2
    local certain_retraining=$3
    local feedback_retraining=$4
    local feedback_mode=$5
    local feedback_threshold=$6

    echo "Running: seed_size=$seed_size strategy=$selection_strategy certain=$certain_retraining feedback=$feedback_retraining mode=$feedback_mode threshold=$feedback_threshold"

    python src/experiment_3.py \
        --seed_size $seed_size \
        --selection_strategy $selection_strategy \
        --certain_retraining $certain_retraining \
        --feedback_retraining $feedback_retraining \
        --feedback_mode $feedback_mode \
        --feedback_threshold $feedback_threshold

    if [ $? -ne 0 ]; then
        echo "Experiment failed, stopping execution"
        exit 1
    fi
}

# Calculate total experiments
total_experiments=$((${#SEED_SIZE_VALUES[@]} * ${#SELECTION_STRATEGIES[@]} * ${#CERTAIN_RETRAINING_VALUES[@]} * ${#FEEDBACK_RETRAINING_VALUES[@]} * ${#FEEDBACK_MODE_VALUES[@]} * ${#FEEDBACK_THRESHOLD_VALUES[@]}))

echo "Experiment 3 Policy Comparison Grid Runner"
echo "Total experiments: $total_experiments"
echo "Continue? (y/N):"
read -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Execution cancelled"
    exit 0
fi

# Generate and shuffle combinations
temp_combinations=$(mktemp)

for seed_size in "${SEED_SIZE_VALUES[@]}"; do
    for selection_strategy in "${SELECTION_STRATEGIES[@]}"; do
        for certain_retraining in "${CERTAIN_RETRAINING_VALUES[@]}"; do
            for feedback_retraining in "${FEEDBACK_RETRAINING_VALUES[@]}"; do
                for feedback_mode in "${FEEDBACK_MODE_VALUES[@]}"; do
                    for feedback_threshold in "${FEEDBACK_THRESHOLD_VALUES[@]}"; do
                        echo "$seed_size|$selection_strategy|$certain_retraining|$feedback_retraining|$feedback_mode|$feedback_threshold"
                    done
                done
            done
        done
    done
done | shuf > "$temp_combinations"

echo "Generated $(wc -l < "$temp_combinations") shuffled combinations"

# Run experiments
experiment_count=0
while IFS='|' read -r seed_size selection_strategy certain_retraining feedback_retraining feedback_mode feedback_threshold; do
    experiment_count=$((experiment_count + 1))
    echo "[$experiment_count/$total_experiments] $(date '+%H:%M:%S')"
    run_experiment $seed_size $selection_strategy $certain_retraining $feedback_retraining $feedback_mode $feedback_threshold
done < "$temp_combinations"

rm "$temp_combinations"
echo "All experiments completed"
