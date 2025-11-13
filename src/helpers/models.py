"""
Autoencoder model architecture and training utilities for HIL-CBM framework.
"""

import os
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def setup_tensorflow_environment(config):
    """Configure TensorFlow for deterministic CPU-only execution"""
    # Set environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['OMP_NUM_THREADS'] = str(config['computation']['cpu_threads'])
    os.environ['TF_NUM_INTEROP_THREADS'] = str(config['computation']['cpu_threads'])
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(config['computation']['cpu_threads'])
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU only

    # Set seeds for reproducibility
    seed = config['global']['seed']
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)

    # Force TensorFlow to be deterministic
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        # For older TF versions
        pass


def build_autoencoder(config):
    """Build convolutional autoencoder with deterministic weight initialization"""
    # Ensure deterministic weight initialization
    seed = config['global']['seed']
    n_timesteps = config['global']['n_timesteps']

    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)

    # Model architecture matching original implementation
    inp = tf.keras.layers.Input(shape=(n_timesteps, 1))
    x = tf.keras.layers.Conv1D(
        config['model']['filters'][0],
        config['model']['kernel_size'],
        strides=config['model']['strides'],
        padding='same',
        activation='relu'
    )(inp)
    x = tf.keras.layers.Dropout(config['model']['dropout_rate'], seed=seed)(x)
    x = tf.keras.layers.Conv1D(
        config['model']['filters'][1],
        config['model']['kernel_size'],
        strides=config['model']['strides'],
        padding='same',
        activation='relu'
    )(x)
    x = tf.keras.layers.Conv1DTranspose(
        config['model']['filters'][1],
        config['model']['kernel_size'],
        strides=config['model']['strides'],
        padding='same',
        activation='relu'
    )(x)
    x = tf.keras.layers.Dropout(config['model']['dropout_rate'], seed=seed)(x)
    x = tf.keras.layers.Conv1DTranspose(
        config['model']['filters'][0],
        config['model']['kernel_size'],
        strides=config['model']['strides'],
        padding='same',
        activation='relu'
    )(x)
    out = tf.keras.layers.Conv1DTranspose(
        1,
        config['model']['kernel_size'],
        strides=1,
        padding='same',
        activation='linear'
    )(x)

    model = tf.keras.models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['model']['learning_rate']),
        loss='mse'
    )

    return model


def train_autoencoder(X_train, config, use_validation=True):
    """Train autoencoder from scratch with deterministic seeding"""
    # Set seed before each training to ensure reproducibility
    seed = config['global']['seed']
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)

    model = build_autoencoder(config)

    # Dynamic batch size and validation split
    batch_size = min(config['model']['batch_size'], len(X_train))
    n = len(X_train)

    # Skip validation for retraining to improve computational efficiency
    if use_validation and n >= config['model']['min_samples_for_validation']:
        val_split = config['model']['validation_split']
        monitor = 'val_loss'
    else:
        val_split = 0.0
        monitor = 'loss'

    # Early stopping callback
    es = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=config['model']['patience'],
        mode='min',
        restore_best_weights=True,
        verbose=0
    )

    # Train model with validation split
    if val_split > 0:
        model.fit(
            X_train, X_train,
            batch_size=batch_size,
            epochs=config['model']['epochs'],
            validation_split=val_split,
            callbacks=[es],
            verbose=0,
            shuffle=config['model']['shuffle']
        )
    else:
        model.fit(
            X_train, X_train,
            batch_size=batch_size,
            epochs=config['model']['epochs'],
            callbacks=[es],
            verbose=0,
            shuffle=config['model']['shuffle']
        )

    # Calculate threshold as maximum training MAE
    preds = model.predict(X_train, verbose=0)
    mae = np.mean(np.abs(preds - X_train), axis=(1, 2))
    threshold = np.max(mae)

    return model, threshold


def train_single_model_task(traj, var_idx, training_data, config, use_validation=True):
    """Train a single autoencoder model from scratch for given trajectory and variable"""
    # Set seed before each individual model training for full determinism
    seed = config['global']['seed']
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)

    # Prepare training data for this variable
    Xtr = np.stack(training_data)
    Xv = Xtr[:, :, var_idx][:, :, np.newaxis]

    # Train model
    model, threshold = train_autoencoder(Xv, config, use_validation)

    return traj, var_idx, model, threshold


def train_trajectory_models(trajectory_training_data, config, seed_size=None, skip_existing=True, use_validation=True):
    """Train models for trajectories, optionally loading existing ones. Returns (models, thresholds, was_loaded)"""
    models_dict = {}
    thresholds_dict = {}
    n_variables = config['global']['n_variables']

    # Try to load existing models first if seed_size is provided and skip_existing is True
    if skip_existing and seed_size is not None:
        print(f"Checking for existing models at seed_size_{seed_size}...")
        existing_models, existing_thresholds = load_pretrained_models(config, seed_size)
        if existing_models is not None and existing_thresholds is not None:
            print(f"Found existing models for seed_size_{seed_size}, skipping training")
            return existing_models, existing_thresholds, True
        else:
            print(f"No existing models found for seed_size_{seed_size}, training from scratch...")

    # Create all training tasks (trajectory, variable pairs)
    training_tasks = []
    for traj in ['PAP1', 'PAP2', 'PAP3']:
        if traj in trajectory_training_data and len(trajectory_training_data[traj]) > 0:
            models_dict[traj] = [None] * n_variables
            thresholds_dict[traj] = np.zeros(n_variables)

            for var_idx in range(n_variables):
                training_tasks.append((traj, var_idx, trajectory_training_data[traj]))

    if not training_tasks:
        return models_dict, thresholds_dict, False

    # Parallel training from scratch of individual models
    total_models = len(training_tasks)
    max_workers = config['computation']['model_workers']

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=total_models, desc="Training autoencoder models", leave=False) as pbar:
            # Submit all tasks
            future_to_task = {}
            for traj, var_idx, training_data in training_tasks:
                future = executor.submit(train_single_model_task, traj, var_idx, training_data, config, use_validation)
                future_to_task[future] = (traj, var_idx)

            # Process completed tasks
            from concurrent.futures import as_completed
            for future in as_completed(future_to_task):
                traj, var_idx, model, threshold = future.result()
                models_dict[traj][var_idx] = model
                thresholds_dict[traj][var_idx] = threshold
                pbar.update(1)

    return models_dict, thresholds_dict, False


def load_pretrained_models(config, seed_size):
    """Load pre-trained models from disk (if available)"""
    models_dict = {}
    thresholds_dict = {}
    n_variables = config['global']['n_variables']

    model_base_dir = os.path.join(config['paths']['models'], f'seed_size_{seed_size}')

    if not os.path.exists(model_base_dir):
        print(f"No pre-trained models found at {model_base_dir}")
        return None, None

    for traj in ['PAP1', 'PAP2', 'PAP3']:
        traj_dir = os.path.join(model_base_dir, traj)

        if not os.path.exists(traj_dir):
            print(f"Missing trajectory directory: {traj_dir}")
            return None, None

        models = []

        # Load all variable models
        for j in range(n_variables):
            model_path = os.path.join(traj_dir, f'ae_var{j}.h5')
            if not os.path.exists(model_path):
                print(f"Missing model: {model_path}")
                return None, None
            model = tf.keras.models.load_model(model_path, compile=False)
            models.append(model)

        # Load thresholds
        thresholds_path = os.path.join(traj_dir, 'thresholds.npy')
        if not os.path.exists(thresholds_path):
            print(f"Missing thresholds: {thresholds_path}")
            return None, None
        base_thresholds = np.load(thresholds_path)

        models_dict[traj] = models
        thresholds_dict[traj] = base_thresholds

    return models_dict, thresholds_dict


def clone_models_for_retraining(static_models, static_thresholds, config):
    """Clone models for retraining experiments"""
    retrain_models = {}
    retrain_thresholds = {}
    n_variables = config['global']['n_variables']

    for traj in ['PAP1', 'PAP2', 'PAP3']:
        if traj in static_models:
            retrain_models[traj] = []
            retrain_thresholds[traj] = static_thresholds[traj].copy()

            for j in range(n_variables):
                model_copy = tf.keras.models.clone_model(static_models[traj][j])
                model_copy.set_weights(static_models[traj][j].get_weights())
                model_copy.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=config['model']['learning_rate']),
                    loss='mse'
                )
                retrain_models[traj].append(model_copy)

    return retrain_models, retrain_thresholds


def retrain_models_with_data(models_dict, thresholds_dict, accumulated_training_data, config, trajectories_to_retrain=None):
    """Retrain models from scratch with accumulated training data"""
    if trajectories_to_retrain is None:
        trajectories_to_retrain = ['PAP1', 'PAP2', 'PAP3']

    # Filter trajectories that have training data
    active_trajectories = [traj for traj in trajectories_to_retrain
                          if traj in accumulated_training_data and len(accumulated_training_data[traj]) > 0]

    if not active_trajectories:
        return

    # Create filtered training data
    filtered_training_data = {traj: accumulated_training_data[traj] for traj in active_trajectories}

    # Train new models without validation for computational efficiency
    new_models, new_thresholds, _ = train_trajectory_models(filtered_training_data, config, use_validation=False)

    # Update existing models
    for traj in active_trajectories:
        if traj in new_models:
            models_dict[traj] = new_models[traj]
            thresholds_dict[traj] = new_thresholds[traj]


def save_trained_models(models_dict, thresholds_dict, config, seed_size, experiment_name=None):
    """Save trained models to disk"""
    if experiment_name:
        model_dir = os.path.join(config['paths']['models'], experiment_name, f'seed_size_{seed_size}')
    else:
        model_dir = os.path.join(config['paths']['models'], f'seed_size_{seed_size}')

    os.makedirs(model_dir, exist_ok=True)

    for traj in models_dict:
        traj_dir = os.path.join(model_dir, traj)
        os.makedirs(traj_dir, exist_ok=True)

        # Save models
        for j, model in enumerate(models_dict[traj]):
            model_path = os.path.join(traj_dir, f'ae_var{j}.h5')
            model.save(model_path)

        # Save thresholds
        thresholds_path = os.path.join(traj_dir, 'thresholds.npy')
        np.save(thresholds_path, thresholds_dict[traj])

    print(f"Models saved to: {model_dir}")


    


def batch_inference(batch_cycles, batch_trajectories, models_dict, config):
    """Perform batch inference with trajectory-grouped, variable-parallel processing"""
    n_samples = len(batch_cycles)
    n_variables = config['global']['n_variables']
    all_mae_scores = np.zeros((n_samples, n_variables))

    # Group samples by trajectory
    trajectory_groups = {}
    for i, traj in enumerate(batch_trajectories):
        if traj not in trajectory_groups:
            trajectory_groups[traj] = {'indices': [], 'cycles': []}
        trajectory_groups[traj]['indices'].append(i)
        trajectory_groups[traj]['cycles'].append(batch_cycles[i])

    def process_trajectory_group(traj, group_data):
        if traj not in models_dict:
            return None

        indices = group_data['indices']
        cycles = group_data['cycles']
        group_size = len(cycles)

        # Compute MAE for all variables in parallel for this trajectory group
        def process_variable(j):
            Xb_j = np.stack([cycle[:, j] for cycle in cycles])[:, :, np.newaxis]
            preds = models_dict[traj][j].predict(Xb_j, verbose=0, batch_size=group_size)
            return j, np.mean(np.abs(preds - Xb_j), axis=(1, 2))

        group_mae_scores = np.zeros((group_size, n_variables))
        var_workers = config['computation']['max_workers']
        with ThreadPoolExecutor(max_workers=var_workers) as var_exec:
            from concurrent.futures import as_completed
            futures = [var_exec.submit(process_variable, j) for j in range(n_variables)]
            for fut in as_completed(futures):
                j, mae_vec = fut.result()
                group_mae_scores[:, j] = mae_vec

        return traj, indices, group_mae_scores

    # Optionally parallelize across trajectory groups
    traj_workers = min(3, len(trajectory_groups)) if trajectory_groups else 0
    if traj_workers > 1:
        with ThreadPoolExecutor(max_workers=traj_workers) as exec_traj:
            futures = [exec_traj.submit(process_trajectory_group, traj, data)
                       for traj, data in trajectory_groups.items()]
            from concurrent.futures import as_completed
            for fut in as_completed(futures):
                result = fut.result()
                if result is None:
                    continue
                _, indices, group_mae = result
                for local_idx, original_idx in enumerate(indices):
                    all_mae_scores[original_idx] = group_mae[local_idx]
    else:
        for traj, data in trajectory_groups.items():
            result = process_trajectory_group(traj, data)
            if result is None:
                continue
            _, indices, group_mae = result
            for local_idx, original_idx in enumerate(indices):
                all_mae_scores[original_idx] = group_mae[local_idx]

    return all_mae_scores


def get_base_predictions(mae_scores, batch_trajectories, thresholds_dict):
    """Get base autoencoder predictions using original thresholds"""
    base_predictions = []

    for i, traj in enumerate(batch_trajectories):
        if traj in thresholds_dict:
            base_pred = 'unhealthy' if np.any(mae_scores[i] > thresholds_dict[traj]) else 'healthy'
        else:
            base_pred = 'healthy'
        base_predictions.append(base_pred)

    return base_predictions
