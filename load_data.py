import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import transformers
import sentence_transformers
import gensim
import gensim.downloader

import collections



HUGGING_FACE_PRETRAINED_MODELS = {
    'bert-base-uncased': {'tokenizer': transformers.BertTokenizer, 'model': transformers.BertModel},
    'distilbert-base-uncased': {'tokenizer': transformers.DistilBertTokenizer, 'model': transformers.DistilBertModel},
    'roberta-base': {'tokenizer': transformers.RobertaTokenizer, 'model': transformers.RobertaModel},
    'google/electra-small-discriminator': {'tokenizer': transformers.ElectraTokenizer, 'model': transformers.ElectraModel},
}

SBERT_PRETRAINED_MODELS = {
    'bert-base-nli-mean-tokens': {'tokenizer': None, 'model': sentence_transformers.SentenceTransformer},
    'bert-large-nli-mean-tokens': {'tokenizer': None, 'model': sentence_transformers.SentenceTransformer},
    'bert-base-nli-stsb-mean-tokens': {'tokenizer': None, 'model': sentence_transformers.SentenceTransformer},
    'bert-large-nli-stsb-mean-tokens': {'tokenizer': None, 'model': sentence_transformers.SentenceTransformer},
    'distilbert-base-nli-mean-tokens': {'tokenizer': None, 'model': sentence_transformers.SentenceTransformer},
    'distilbert-base-nli-stsb-mean-tokens': {'tokenizer': None, 'model': sentence_transformers.SentenceTransformer},
    'roberta-base-nli-stsb-mean-tokens': {'tokenizer': None, 'model': sentence_transformers.SentenceTransformer},
    'roberta-large-nli-stsb-mean-tokens': {'tokenizer': None, 'model': sentence_transformers.SentenceTransformer},
}

GENSIM_PRETRAINED_MODELS = set([
    'word2vec-google-news-300',
    'glove-wiki-gigaword-50',
    'glove-wiki-gigaword-100',
    'glove-wiki-gigaword-200',
    'glove-wiki-gigaword-300',
    'glove-twitter-25',
    'glove-twitter-50',
    'glove-twitter-100',
    'glove-twitter-200',
    'fasttext-wiki-news-subwords-300',
    'conceptnet-numberbatch-17-06-300',
])



class DataLoader():
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
    
    
    
class BertFeatureExtractor():
    def __init__(self, config, X_train, X_valid=None, X_test=None):
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.config = config
        self.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer, self.model = self.get_pretrain_model(self.config, self.device)

    def extract_features(self):
        self.X_train = self.get_embeddings(self.X_train)
        if self.X_valid:
            self.X_valid = self.get_embeddings(self.X_valid)
        if self.X_test:
            self.X_test = self.get_embeddings(self.X_test)

    def get_pretrain_model(self, config, device):
        if config in HUGGING_FACE_PRETRAINED_MODELS:
            tokenizer = HUGGING_FACE_PRETRAINED_MODELS[config]['tokenizer'].from_pretrained(config)
            model = HUGGING_FACE_PRETRAINED_MODELS[config]['model'].from_pretrained(config, return_dict=True)
            model = model.to(device)
            model.eval()
            return tokenizer, model

        elif config in SBERT_PRETRAINED_MODELS:
            tokenizer = SBERT_PRETRAINED_MODELS[config]['tokenizer']
            if tokenizer != None:
                tokenizer = tokenizer(config)
            model = SBERT_PRETRAINED_MODELS[config]['model'](config)
            model = model.to(device)
            model.eval()
            return tokenizer, model

        else:
            raise RuntimeError("Unsupported models.")

    def get_embeddings(self, texts):
        if self.config in HUGGING_FACE_PRETRAINED_MODELS:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            return embeddings

        elif self.config in SBERT_PRETRAINED_MODELS:
            embeddings = self.model.encode(texts)
            return embeddings

        else:
            raise RuntimeError("Unsupported models.")

    def get_train_encodings(self):
        return self.X_train
    
    def get_valid_encodings(self):
        return self.X_valid
    
    def get_test_encodings(self):
        return self.X_test
    
    
    
class Word2VecFeatureExtractor():
    def __init__(self, config, X_train, X_valid=None, X_test=None, output_encodings='mean'):
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.config = config
        self.output_encodings = output_encodings
        self.word_embeddings = self.get_pretrain_model(self.config)

    def extract_features(self):
        self.X_train = self.get_embeddings(self.X_train)
        if self.X_valid:
            self.X_valid = self.get_embeddings(self.X_valid)
        if self.X_test:
            self.X_test = self.get_embeddings(self.X_test)

    def get_pretrain_model(self, config):
        if config in GENSIM_PRETRAINED_MODELS:
            word_embeddings = gensim.downloader.load(config)
            return word_embeddings

        else:
            raise RuntimeError("Unsupported models.")

    def get_embeddings(self, texts):
        if self.config in GENSIM_PRETRAINED_MODELS:
            data_embeddings = []
            for sentence in texts:
                words = [word.lower() for word in sentence.split()]
                sentence_embeddings = []
                for word in words:
                    if word in self.word_embeddings:
                        sentence_embeddings.append(self.word_embeddings[word])
                sentence_embeddings = np.asarray(sentence_embeddings)
                if self.output_encodings == 'mean':
                    sentence_embeddings = np.mean(sentence_embeddings, axis=0)
                data_embeddings.append(sentence_embeddings)
            data_embeddings = np.asarray(data_embeddings)
            return data_embeddings

        else:
            raise RuntimeError("Unsupported models.")

    def get_train_encodings(self):
        return self.X_train
    
    def get_valid_encodings(self):
        return self.X_valid
    
    def get_test_encodings(self):
        return self.X_test