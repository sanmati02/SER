import importlib
from loguru import logger

# Import available model classes
from .bi_lstm import BiLSTM, StackedLSTM, LSTMAdditiveAttention, StackedLSTMAdditiveAttention

# Define what should be accessible when using `from module import *`
__all__ = ['build_model']

def build_model(input_size, configs):
    """
    Dynamically builds a model instance based on configuration.

    :param input_size: Dimensionality of the input features
    :param configs: Configuration object containing model settings
    :return: Instantiated model object
    """
    # Get the model name from config (default is 'BiLSTM')
    use_model = configs.model_conf.get('model', 'BiLSTM')
    
    # Get additional arguments specific to the model (e.g., hidden size, num layers)
    model_args = configs.model_conf.get('model_args', {})
    
    # Dynamically get the model class from the current module
    mod = importlib.import_module(__name__)
    model = getattr(mod, use_model)(input_size=input_size, **model_args)

    # Log the model creation
    logger.info(f'Model created successfully: {use_model} with args: {model_args}')
    
    return model
