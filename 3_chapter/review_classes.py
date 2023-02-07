from collections import Counter
import json
import string

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Vocabulary(object):
    """
    Класс, предназначенный для обработки текста и извлечения значений токенов
    """
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        Args:
            token_to_idx (dict): pre-existing map of tokens to indices
            add_unk (bool): flag that indicates whether add the UNK token
            unk_token (str): UNK token to add in Vocab
        """
        if token_to_idx is None:
            token_to_idx = dict()
        self._token_to_idx = token_to_idx
        self._idx_to_token = {
            idx: token 
            for token, idx in self._token_to_idx.items()
        }
        self._add_unk = add_unk
        self._unk_token = unk_token
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)
            
    def to_serializable(self):
        """ возвращает словарь с возможностью сериализации """
        return {
            'token_to_idx': self._token_to_idx,
            'add_unk': self._add_unk,
            'unk_token': self._unk_token
        }
    @classmethod
    def from_serializable(cls, contents):
        """ создает экземпляр класса Vocabulary из сериализованного словаря """
        return cls(**contents)
    
    def add_token(self, token):
        """ Добавляет токен в словари, возвращая его индекс
        
        Args:
            token (str): токен, добавляемый в Vocabulary
        Returns:
            index (int): индекс токена в словарях
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index
    
    def add_many(self, tokens):
        """Добавляет список токенов в словарь
        
        Args:
            tokens (list): список токенов типа string
        Returns:
            indices (list): список индексов, соответствующих списку токенов
        """
        return [self.add_token for token in tokens]
    
    def lookup_token(self, token):
        """Возвращает число, соответствующее токену или индекс элемента UNK.
        
        Args:
            token (str): токен
        Returns:
            index (int): индекс, соответствующий токену
        Notes:
            `unk_index` должен быть >=0 (добавлен в словарь) 
              для функционирования UNK
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]
        
    def lookup_index(self, index):
        """
        """
        if index not in self._idx_to_token:
            raise KeyError(f"The index {index} is not in Vocabulary")
        return self._idx_to_token[index]
    
    def __str__(self):
        return f"Vocabulary(size={len(self)})"
    
    def __len__(self):
        return len(self._token_to_idx)
    
class Vectorizer(object):
    """ Класс, координирующий Vocabularies и использует их
    """
    def __init__(self, review_vocab, rating_vocab):
        """
        Args:
            review_vocab (Vocabulary): токены - цифры
            rating_vocab (Vocabulary): метки классов - цифры
        """
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab
        
    def vectorize(self, review):
        """Создает вектор для обзора
        Args:
            review (str): обзор
        Returns:
            one_hot (np.ndarray): one-hot вектор
        """
        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)
        
        for token in review.split(" "):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1
        return one_hot
    
    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        """Инициализирует Vectorizer из pandas.DataFrame
        
        Args:
            review_df (pandas.DataFrame): датасет обзоров
            cutoff (int): параметр для отсеивания по частоте
        Returns:
            Экземпляр Vectorizer
        """        
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)
        
        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)
            
        word_counts = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1
        for word, count in word_counts:
            if count > cutoff:
                review_vocab.add_token(word)
        return cls(review_vocab, rating_vocab)
    
    @classmethod
    def from_serializable(cls, contents):
        """Инициализирует Vectorizer из сериализованного словаря
        Args:
            contents (dict): сериализованный словарь
        Returns:
            Экземпляр Vectorizer
        """
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['rating_vocab'])
        
        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)
    
    def to_serializable(self):
        """Создает сериализованный словарь на основе класса Vectorizer
        """
        return {
            'review_vocab': self.review_vocab.to_serializable(),
            'rating_vocab': self.rating_vocab.to_serializable()
        }

class ReviewDataset(Dataset):
    def __init__(self, review_df, vectorizer):
        """
        Args:
            review_df (pandas.DataFrame): Датасет
            vectorizer (Vectorizer): Vectorizer, созданный из датасета
        """
        self.review_df = review_df
        self._vectorizer = vectorizer
        
        self.train_df = self.review_df[self.review_df.split=='train']
        self.train_size = len(self.train_df)
        
        self.test_df = self.review_df[self.review_df.split=='test']
        self.test_size = len(self.test_df)
        
        self.val_df = self.review_df[self.review_df.split=='val']
        self.val_size = len(self.val_df)
        
        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.val_size),
            'test': (self.test_df, self.test_size)
        }
        self.set_split('train')
        
    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
        """Загружает датасет из пути review_csv и создает для него Vectorizer
        Args:
            review_csv (str): путь к данным
        Returns:
            Экземпляр ReviewDataset
        """
        review_df = pd.read_csv(review_csv)
        train_review_df = review_df[review_df.split=='train']
        return cls(review_df, Vectorizer.from_dataframe(train_review_df))
    
    @classmethod
    def load_dataset_and_load_vectorizer(cls, review_csv, vectorizer_filepath):
        """Загружает датасет из пути review_csv и Vectorizer из пути vectorizer_filepath
        Используется в случае если Vectorizer был закеширован
        Args:
            review_csv (str): путь к данным
            vectorizer_filepath (str): путь к Vectorizer
        Returns:
            Экземпляр ReviewDataset
        """
        review_df = pd.read_csv(review_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(review_df, vectorizer)
    
    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """Загружает Vectorizer из vectorizer_filepath
        Args:
            vectorizer_filepath (str): путь к Vectorizer
        Returns:
            Экземпляр Vectorizer
        """
        with open(vectorizer_filepath) as fp:
            return Vectorizer.from_serializable(json.load(fp))
    
    def save_vectorizer(self, vectorizer_filepath):
        """Сохраняет Vectorizer в json
        
        Args:
            vectorizer_filepath (str): путь, куда сохранять
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)
            
    def get_vectorizer(self):
        """ Возвращает Vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """Выбирает разбиение, используя колонку split в dataframe
        
        Args:
            split (str): одно из "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
        
    def __len__(self):
        """ Для PyTorch, возвращает длину датасета """
        return self._target_size
    
    def __getitem__(self, index):
        """ Для PyTorch, возвращает строку по индексу
        Args:
            index (int): индекс строки данных 
        Returns:
            словарь, содержащий фичи (x_data) и метку (y_target)
        """
        row = self._target_df.iloc[index]
        review_vector = self._vectorizer.vectorize(row.review)
        rating_index = self._vectorizer.rating_vocab.lookup_token(row.rating)
        return {
            'x_data': review_vector,
            'y_target': rating_index
        }
    def get_num_batches(self, batch_size):
        """ Возврращает количество батчей при заданном batch_size
        Args:
            batch_size (int): размер батча
        Returns:
            количество батчей в датасете
        """
        return len(self) // batch_size
    