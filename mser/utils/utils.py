import distutils.util
import os
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np


def print_arguments(args=None, configs=None, title=None):
    """
    Log command-line arguments and/or config file parameters.

    :param args: Parsed argparse arguments.
    :param configs: Configuration dictionary (e.g., loaded from YAML or JSON).
    :param title: Optional custom title for logging config parameters.
    """
    if args:
        logger.info("----------- Extra Command-line Arguments -----------")
        for arg, value in sorted(vars(args).items()):
            logger.info("%s: %s" % (arg, value))
        logger.info("----------------------------------------------------")
    if configs:
        title = title if title else "Configuration File Parameters"
        logger.info(f"----------- {title} -----------")
        for arg, value in sorted(configs.items()):
            if isinstance(value, dict):
                logger.info(f"{arg}:")
                for a, v in sorted(value.items()):
                    if isinstance(v, dict):
                        logger.info(f"\t{a}:")
                        for a1, v1 in sorted(v.items()):
                            logger.info("\t\t%s: %s" % (a1, v1))
                    else:
                        logger.info("\t%s: %s" % (a, v))
            else:
                logger.info("%s: %s" % (arg, value))
        logger.info("----------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """
    Add an argument to an ArgumentParser with bool support.

    :param argname: Argument name (string, no leading dashes)
    :param type: Expected type (e.g., str, int, bool)
    :param default: Default value
    :param help: Help description for argparse
    :param argparser: ArgumentParser object
    :param kwargs: Additional argparse options
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' Default: %(default)s.',
                           **kwargs)


# Enables dot-access to dictionary values
class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dict_obj):
    """
    Recursively convert nested dictionaries into Dict objects
    for attribute-style access.

    :param dict_obj: Dictionary
    :return: Dict object (with attribute access)
    """
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = Dict()
    for k, v in dict_obj.items():
        inst[k] = dict_to_object(v)
    return inst


def plot_confusion_matrix(cm, save_path, class_labels, show=False):
    """
    Plot and save a normalized confusion matrix.

    :param cm: Confusion matrix (2D numpy array)
    :param save_path: Path to save image
    :param class_labels: List of class names
    :param show: If True, also display the matrix
    """
    # Check for non-ASCII characters (e.g., Chinese)
    s = ''.join(class_labels)
    is_ascii = all(ord(c) < 128 for c in s)
    if not is_ascii:
        # Use a compatible font for non-ASCII labels
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # Plot each cell's normalized value
    ind_array = np.arange(len(class_labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val] / (np.sum(cm[:, x_val]) + 1e-6)
        if c < 1e-4:
            continue
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    # Normalize by column totals
    m = np.sum(cm, axis=0) + 1e-6
    plt.imshow(cm / m, interpolation='nearest', cmap=plt.cm.binary)
    plt.title('Confusion Matrix' if is_ascii else 'Confusion Matrix (Non-ASCII Labels)')
    plt.colorbar()

    # Label axes
    xlocations = np.array(range(len(class_labels)))
    plt.xticks(xlocations, class_labels, rotation=90)
    plt.yticks(xlocations, class_labels)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # Add minor ticks for grid lines
    tick_marks = np.array(range(len(class_labels))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png')
    if show:
        plt.show()


def convert_string_based_on_type(a, b):
    """
    Convert string b to the same type as variable a.

    :param a: Example variable with target type
    :param b: String to convert
    :return: Converted value
    """
    if isinstance(a, int):
        try:
            b = int(b)
        except ValueError:
            logger.error("Failed to convert string to int.")
    elif isinstance(a, float):
        try:
            b = float(b)
        except ValueError:
            logger.error("Failed to convert string to float.")
    elif isinstance(a, str):
        return b
    elif isinstance(a, bool):
        b = b.lower() == 'true'
    else:
        try:
            b = eval(b)
        except Exception as e:
            logger.exception("Failed to convert string to other types; ignoring type conversion.")
    return b
