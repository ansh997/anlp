import os
import torch
from torch.utils.data import Dataset
from collections import Counter
import numpy as np

scratch_location = f'/scratch/hmnshpl/anlp_data'
filename = 'Auguste_Maquet.txt'
emb_filename = 'glove.6B.300d.txt'

data_filepath = os.path.join(scratch_location, filename)
emb_filepath = os.path.join(scratch_location, emb_filename)

class TextDataset(Dataset):
    def __init__(self, file_path, embedding_model, context_size=5,
                unk_token="<UNK>", embedding_dim=300):
        """
        Custom Dataset for Language Modeling.
        
        Args:
        - file_path (str): Path to the text dataset file.
        - embedding_model (dict): Pre-trained word embeddings (e.g., GloVe or word2vec in dictionary format).
        - context_size (int): Size of the context window (5 for 5-gram).
        - unk_token (str): Token for unknown words.
        - embedding_dim (int): Dimensionality of word embeddings.
        """
        self.context_size = context_size
        self.unk_token = unk_token
        self.embedding_dim = embedding_dim
        
        self.tokenized_text = self._tokenize_text(file_path)
        
        self.vocab, self.word_to_idx, self.idx_to_word = self._build_vocabulary(self.tokenized_text)
        
        self.data = self._prepare_ngrams(self.tokenized_text)
        
        self.embedding_matrix = self._create_embedding_matrix(embedding_model)
    
    def _tokenize_text(self, file_path=data_filepath):
        """
        Tokenizes the text data into words.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().lower().split()  # Lowercasing and splitting into words
        return text
    
    def _build_vocabulary(self, tokens):
        """
        Constructs a vocabulary from tokens and assigns indices to each word.
        """
        # Count frequencies of each word
        counter = Counter(tokens)
        
        # Create vocab list sorted by frequency
        vocab = sorted(counter, key=counter.get, reverse=True)
        
        # Add <UNK> token for unknown words
        vocab.insert(0, self.unk_token)
        
        # Create word-to-index and index-to-word mappings
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        return vocab, word_to_idx, idx_to_word
    
    def _prepare_ngrams(self, tokens):
        """
        Prepares sequences of 5-gram contexts and their next-word targets.
        """
        data = []
        for i in range(len(tokens) - self.context_size):
            context = tokens[i:i+self.context_size]
            target = tokens[i+self.context_size]
            data.append((context, target))
        return data
    
    def _create_embedding_matrix(self, embedding_model):
        """
        Creates an embedding matrix for the vocabulary using pre-trained embeddings.
        """
        # Initialize the embedding matrix with zeros
        embedding_matrix = np.zeros((len(self.vocab), self.embedding_dim))
        
        for word, idx in self.word_to_idx.items():
            # Assign the embedding if it exists in the pre-trained model
            if word in embedding_model:
                embedding_matrix[idx] = embedding_model[word]
            else:
                # Assign a random vector if the word is unknown
                embedding_matrix[idx] = np.random.normal(size=(self.embedding_dim,))
        
        return torch.tensor(embedding_matrix, dtype=torch.float32)
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves the input-output pair for the given index.
        """
        context, target = self.data[idx]
        
        # Convert context words to their indices
        context_indices = [self.word_to_idx.get(word, self.word_to_idx[self.unk_token]) for word in context]
        
        # Convert target word to its index
        target_index = self.word_to_idx.get(target, self.word_to_idx[self.unk_token])
        
        # Retrieve the embeddings for the context
        context_embeddings = torch.cat([self.embedding_matrix[idx].unsqueeze(0) for idx in context_indices], dim=0)
        
        return context_embeddings, target_index
