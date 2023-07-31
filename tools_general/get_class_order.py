import argparse

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--total', type=int, default=-1)
    parser.add_argument('--selected', type=int, default=-1)
    args = parser.parse_args()

    np.random.seed(2023)
    class_num = args.total
    selected_num = args.selected
    if class_num == selected_num:
        order = np.arange(class_num)
        np.random.shuffle(order)
        print_str = ''
        for idx, i in enumerate(order):
            print_str += '{},'.format(i)
            if (idx + 1) % 20 == 0:
                print_str += '\n'
        print(print_str)
    else:
        assert class_num > selected_num
        order = np.random.choice(np.arange(class_num), selected_num, replace=False)
        print_str = ''
        for idx, i in enumerate(order):
            print_str += '{},'.format(i)
            if (idx + 1) % 20 == 0:
                print_str += '\n'
        print(print_str)
