import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--epoch_num', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=int)
    parser.add_argument('--iter_display', default=2, type=int)
    parser.add_argument('--lr_milestone', default=[30, 80], type=list)

    args = parser.parse_args()
    return args