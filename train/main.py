from datetime import datetime
import os
import json

from .trainer import Trainer
from .settings import settings


if __name__ == '__main__':
    # Check if logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Create log directory for this specific experiment using timestamp
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join('logs', current_time)
    os.makedirs(log_dir)

    # Save settings to log directory
    settings_dict = {
        "model_type": settings.model_type.value,
        "test_size": settings.test_size,
        "categorical_features": settings.categorical_features,
        "model_params": settings.model_params,
        "data_params": settings.data_params,
        "experiment_params": settings.experiment_params
    }

    with open(os.path.join(log_dir, 'settings.json'), 'w') as f:
        json.dump(settings_dict, f)

    train = Trainer(settings.model_type, log_dir)
    train.train()
