import os, sys
import argparse
import numpy as np
from pathlib import Path
from load_data import *


TRAIN_PATHS = {
    'atis': './atis/atis_train_actual.csv',
    'snips': './snips/snips_train_actual.csv',
}

TEST_PATHS = {
    'atis': './atis/atis_test_actual.csv',
    'snips': './snips/snips_test_actual.csv',
}


def load_data(train_path, test_path):
    data_loader = DataLoader(train_path, None, test_path)
    data_loader.split_train_valid(valid_size=0.05, keep_class_ratios=True)
    X_train, y_train = data_loader.get_train_data()
    X_valid, y_valid = data_loader.get_valid_data()
    X_test, y_test = data_loader.get_valid_data()
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def save_encodings(X, y, dataset, config, split):
    save_dir = Path(f'{dataset}_bert_encodings')
    save_dir = save_dir / config
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_dir_X = save_dir / f'X_{split}.npy'
    save_dir_y = save_dir / f'y_{split}.npy'
    np.save(save_dir_X, X)
    np.save(save_dir_y, y)

def create_encodings(dataset, config, data):
    X_train, y_train, X_valid, y_valid, X_test, y_test = data
    feature_extractor = Word2VecFeatureExtractor(config, X_train, X_valid, X_test)
    feature_extractor.extract_features()
    X_train = feature_extractor.get_train_encodings()
    X_valid = feature_extractor.get_valid_encodings()
    X_test = feature_extractor.get_test_encodings()

    save_encodings(X_train, y_train, dataset, config, 'train')
    save_encodings(X_valid, y_valid, dataset, config, 'valid')
    save_encodings(X_test, y_test, dataset, config, 'test')

def generate_encodings(dataset):
    train_path = TRAIN_PATHS[dataset]
    test_path = TEST_PATHS[dataset]
    data = load_data(train_path, test_path)

    for config in GENSIM_PRETRAINED_MODELS:
        create_encodings(dataset, config, data)

def main():
    parser = argparse.ArgumentParser(description = 'PyTorch ResNet Training.',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type = str, default = 'atis',
                        help = 'Dataset.', choices = ['atis', 'snips'])
    args = parser.parse_known_args()[0]
    generate_encodings(args.dataset)

if __name__ == "__main__":
    main()