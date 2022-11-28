import argparse


def get_args():
    parser = argparse.ArgumentParser(description='cv-product-inference')
    parser.add_argument('-epochs', default=1, type=int)
    parser.add_argument('-learning_rate', default=1e-2, type=float)
    parser.add_argument('-batch_size', default=8, type=int)
    parser.add_argument('-image_size', default=224, type=int)
    parser.add_argument('-image_channel', default=224, type=int)
    parser.add_argument('-output_shape', required=True, type=int)
    parser.add_argument('-random_state', default=1234, type=int)

    return parser.parse_args()


def modify_args(args):
    return args


def check_args(args):
    return args