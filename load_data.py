import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import collections



class SnipsDataLoader():
    def __init__(self, train_path, valid_path=None, test_path=None):
        self.train_data = self.load_dataset(train_path)
        self.valid_data = self.load_dataset(valid_path) if valid_path != None else None
        self.test_data = self.load_dataset(test_path) if test_path != None else None
        
        self.create_label_mapping()
        
    def load_dataset(self, data_path):
        data = pd.read_csv(data_path, header=None, delimiter='\t')
        data.columns = ['labels', 'texts']
        output = {'X': data['texts'], 'y': data['labels']}
        return output
        
    def create_label_mapping(self):
        self.text_to_index_label_mapping = {}
        self.index_to_text_label_mapping = {}
        for i, label in enumerate(self.train_data['y'].unique()):
            self.text_to_index_label_mapping[label] = i
            self.index_to_text_label_mapping[i] = label
        
        self.train_data['y'] = \
            self.train_data['y'].map(lambda x: self.text_to_index_label_mapping[x])
        if self.valid_data:
            self.valid_data['y'] = \
                self.valid_data['y'].map(lambda x: self.text_to_index_label_mapping[x])
        if self.test_data:
            self.test_data['y'] = \
                self.test_data['y'].map(lambda x: self.text_to_index_label_mapping[x])
    
    def split_train_valid(self, valid_size, keep_class_ratios=True, random_state=0):
        X, y = self.train_data['X'], self.train_data['y']
        if keep_class_ratios:
            X_train, X_valid, y_train, y_valid = \
                train_test_split(X, y, test_size=valid_size, random_state=random_state, stratify=y)
        else:
            X_train, X_valid, y_train, y_valid = \
                train_test_split(X, y, test_size=valid_size, random_state=random_state)
            
        self.train_data = {'X': X_train, 'y': y_train}
        self.valid_data = {'X': X_valid, 'y': y_valid}
    
    def get_train_data(self):
        return list(self.train_data['X']), self.train_data['y'].to_numpy()
    
    def get_valid_data(self):
        return list(self.valid_data['X']), self.valid_data['y'].to_numpy()
    
    def get_test_data(self):
        return list(self.test_data['X']), self.test_data['y'].to_numpy()
    
class FeatureExtractor2():
    def __init__(self, X_train, X_valid=None, X_test=None):
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test

    def extract_features(self, keep_words_threshold=5):
        self.keep_words_threshold = keep_words_threshold
        self.X_train = self.preprocess_data(self.X_train)
        if self.X_valid:
            self.X_valid = self.preprocess_data(self.X_valid)
        if self.X_test:
            self.X_test = self.preprocess_data(self.X_test)

        self.create_vocab(self.X_train)

        self.X_train = self.create_encodings(self.X_train)
        self.max_dim = self.X_train.shape[1]
        if self.X_valid:
            self.X_valid = self.create_encodings(self.X_valid)
            self.max_dim = max(self.max_dim, self.X_valid.shape[1])
        if self.X_test:
            self.X_test = self.create_encodings(self.X_test)
            max_dim = max(self.max_dim, self.X_test.shape[1])

        self.X_train = np.concatenate((self.X_train, np.zeros((self.X_train.shape[0], self.max_dim - self.X_train.shape[1]))), axis=1)
        if self.X_valid is not None:
            self.X_valid = np.concatenate((self.X_valid, np.zeros((self.X_valid.shape[0], self.max_dim - self.X_valid.shape[1]))), axis=1)
        if self.X_test is not None:
            self.X_test = np.concatenate((self.X_test, np.zeros((self.X_test.shape[0], self.max_dim - self.X_test.shape[1]))), axis=1)


    def preprocess_data(self, text_data):
        output = []
        for example in text_data:
            words = [word.lower() for word in example.split()]
            output.append(words)
        return output

    def create_vocab(self, text_data):
        word_occurences = collections.defaultdict(int)
        for example in text_data:
            word_counts = self.get_word_counts(example)
            for word in word_counts.keys():
                word_occurences[word] += 1

        vocab_words = [word for word in sorted(word_occurences.keys())
                       if word_occurences[word] >= self.keep_words_threshold]
        self.vocab = {word: index for index, word in enumerate(vocab_words)}
        self.vocab_size = len(self.vocab)

    def create_encodings(self, text_data):
        num_examples = len(text_data)

        max_length = 0
        for text in text_data:
            if(len(text) > max_length):
                max_length = len(text)

        encodings = np.zeros((num_examples, max_length))

        for row, example in enumerate(text_data):
            for i in range(len(example)):
                if example[i] in self.vocab:
                    index = self.vocab[example[i]]
                    #print(example[i])
                    #print(index)
                    encodings[row, i] = index
                else:
                    # unknown
                    index = self.vocab_size
                    encodings[row, i] = index

        return encodings

    def get_word_counts(self, word_list):
        counts = collections.defaultdict(int)
        for word in word_list:
            counts[word] += 1
        return counts

    def get_train_encodings(self):
        return self.X_train

    def get_valid_encodings(self):
        return self.X_valid

    def get_test_encodings(self):
        return self.X_test
    
class FeatureExtractor():
    def __init__(self, X_train, X_valid=None, X_test=None):
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
    
    def extract_features(self, keep_words_threshold=5):
        self.keep_words_threshold = keep_words_threshold
        
        self.X_train = self.preprocess_data(self.X_train)
        if self.X_valid:
            self.X_valid = self.preprocess_data(self.X_valid)
        if self.X_test:
            self.X_test = self.preprocess_data(self.X_test)
        
        self.create_vocab(self.X_train)
        
        self.X_train = self.create_encodings(self.X_train)
        if self.X_valid:
            self.X_valid = self.create_encodings(self.X_valid)
        if self.X_test:
            self.X_test = self.create_encodings(self.X_test)
    
    def preprocess_data(self, text_data):
        output = []
        for example in text_data:
            words = [word.lower() for word in example.split()]
            output.append(words)
        return output
    
    def create_vocab(self, text_data):
        word_occurences = collections.defaultdict(int)
        for example in text_data:
            word_counts = self.get_word_counts(example)
            for word in word_counts.keys():
                word_occurences[word] += 1
        
        vocab_words = [word for word in sorted(word_occurences.keys()) 
                       if word_occurences[word] >= self.keep_words_threshold]
        self.vocab = {word: index for index, word in enumerate(vocab_words)}
        self.vocab_size = len(self.vocab)
        
    def create_encodings(self, text_data):
        num_examples = len(text_data)
        encodings = np.zeros((num_examples, self.vocab_size))
        
        for row, example in enumerate(text_data):
            word_counts = self.get_word_counts(example)
            for word, count in word_counts.items():
                if word in self.vocab:
                    col = self.vocab[word]
                    encodings[row, col] = count
                    
        return encodings
                    
    def get_word_counts(self, word_list):
        counts = collections.defaultdict(int)
        for word in word_list:
            counts[word] += 1
        return counts
    
    def get_train_encodings(self):
        return self.X_train
    
    def get_valid_encodings(self):
        return self.X_valid
    
    def get_test_encodings(self):
        return self.X_test