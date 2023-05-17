from loguru import logger
import gin

@gin.configurable
def execute_pipeline(tasks):
    for t in tasks:
        logger.info(f'Executing task {t.__name__}')
        t()

@gin.configurable
def set_seed(seed):
    logger.info(f'Running experiment with seed {seed}')

@gin.configurable
def load_dataset(dataset, dataset_path):
    logger.info(f'Loading {dataset} located in {dataset_path}')

@gin.configurable
def split_dataset(type, proportion, mode):
    logger.info(f'Splitting dataset by {type} with proportion {proportion} and {mode} mode')

@gin.configurable
def extract_features(type):
    logger.info(f'Extracting {type}')

@gin.configurable
def fit_model(model):
    logger.info(f'Fitting model {model}')

@gin.configurable
class Resnet():
    def __init__(self, n_layers=5, n_classes=20):
        logger.info(f'Creating Resnet model with {n_layers} layers to classify {n_classes} classes')

def main():
    gin.parse_config_file('minimal_example.gin')
    logger.info('\nExperiment configuration: \n'+gin.config_str())
    
    execute_pipeline()

main()
