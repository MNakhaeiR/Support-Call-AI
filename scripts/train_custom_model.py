import os
import yaml
from src.models.model_manager import ModelManager

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(model_name, training_data_path, config):
    model_manager = ModelManager()
    model = model_manager.load_model(model_name)
    
    # Assuming the model has a train method
    model.train(training_data_path, config)
    
    model_manager.save_model(model_name, model)

if __name__ == "__main__":
    config_path = os.path.join('config', 'model_config.yaml')
    training_data_path = os.path.join('data', 'training_data')
    
    config = load_config(config_path)
    
    model_name = config.get('model_name', 'default_model')
    
    train_model(model_name, training_data_path, config)