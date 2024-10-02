import torch
import os
from utils import simple_tokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import re


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
        self.word2idx[self.pad_token] = self.pad_idx
        self.word2idx[self.unk_token] = self.unk_idx
        self.word2idx[self.sos_token] = self.sos_idx
        self.word2idx[self.eos_token] = self.eos_idx
        self.idx2word[self.pad_idx] = self.pad_token
        self.idx2word[self.unk_idx] = self.unk_token
        self.idx2word[self.sos_idx] = self.sos_token
        self.idx2word[self.eos_idx] = self.eos_token

    def build_vocab(self, sentences, min_freq=1):
        # print(f'sentences are {", ".join(sentences)}')
        # exit()
        word_freq = {}
        idx = 4  # Start indexing from 3, since 0, 1 and 2 are for <PAD>, <UNK> <SOS>
        
        # Count word frequencies
        for sentence in sentences:
            for word in sentence.lower().split():
                # print(f'{word=}', end=' ')
                word_freq[word] = word_freq.get(word, 0) + 1
                # print(f'{word_freq=}')
        
        # Add words to the vocabulary based on min_freq
        for word, freq in word_freq.items():
            # print(f'{word=} {freq=}', end=' ')
            if freq >= min_freq:
                # print(f'{idx=}', end=' ')
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                # print(f'{self.word2idx[word]=} {self.idx2word[idx]=}')
                idx += 1
                

    def tokenize(self, sentence, type='simple'):
        # Convert sentence into a list of token indices
        if type == 'simple':
            return [self.word2idx.get(word.lower(), self.unk_idx) for word in simple_tokenizer(sentence)]
        elif type == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            sentence = re.sub(r'<[^>]+>', '', sentence)
            tokens = tokenizer.tokenize(sentence)
            return [self.word2idx.get(token, self.unk_idx) for token in tokens]
        else:
            raise ValueError("Invalid tokenizer type. Use 'simple' or 'bert'")

    def __len__(self):
        return len(self.word2idx)
    
    def decode(self, tokens):
        return ' '.join([self.idx2word[token] for token in tokens if token != self.pad_idx])
    
    def batch_decode(self, batch_tokens):
        """
        Decode a batch of token sequences into a list of sentences.
        """
        return [' '.join([self.idx2word[token] for token in tokens if token != self.pad_idx])
                for tokens in batch_tokens]


class Dataset(Dataset):
    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab, max_len, tokenizer='simple', add_sos_eos=True) -> None:
        super(Dataset, self).__init__()
        self.src_sentences = self.load_sentences(src_file)
        self.tgt_sentences = self.load_sentences(tgt_file)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.add_sos_eos = add_sos_eos
        
        assert len(self.src_sentences) == len(self.tgt_sentences), "Source and target files should have same number of lines"
    
    def load_sentences(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        return [sentence.strip() for sentence in sentences]
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # Tokenize and convert to IDs
        src_tokens = self.src_vocab.tokenize(src_sentence, type=self.tokenizer)
        tgt_tokens = self.tgt_vocab.tokenize(tgt_sentence, type=self.tokenizer)
        
        if self.add_sos_eos:
            tgt_tokens = [self.tgt_vocab.sos_idx] + tgt_tokens + [self.tgt_vocab.eos_idx]
        
        # Pad and create input tensors
        src_tensor = self.pad_sequence(src_tokens, self.max_len)
        tgt_tensor = self.pad_sequence(tgt_tokens, self.max_len)
        
        src_tensor = torch.tensor(src_tensor, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_tensor, dtype=torch.long)
        
        
        return src_tensor, tgt_tensor
    
    def pad_sequence(self, tokens, max_len):
        padded_tokens = tokens[:max_len] + [self.src_vocab.pad_idx] * (max_len - len(tokens))
        return torch.tensor(padded_tokens, dtype=torch.long)

def get_dataloaders(scratch_location, src_lang, tgt_lang, batch_size, max_len, min_freq=2, tokenizer='simple'):
    # File paths
    train_src = os.path.join(scratch_location, f'train.{src_lang}')
    train_tgt = os.path.join(scratch_location, f'train.{tgt_lang}')
    
    dev_src = os.path.join(scratch_location, f'dev.{src_lang}')
    dev_tgt = os.path.join(scratch_location, f'dev.{tgt_lang}')
    
    test_src = os.path.join(scratch_location, f'test.{src_lang}')
    test_tgt = os.path.join(scratch_location, f'test.{tgt_lang}')

    # Read training sentences to build vocabularies
    train_src_sentences = open(train_src, 'r').readlines()
    train_tgt_sentences = open(train_tgt, 'r').readlines()

    # Initialize and build vocabularies from the training data
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    src_vocab.build_vocab(train_src_sentences, min_freq=min_freq) 
    tgt_vocab.build_vocab(train_tgt_sentences, min_freq=min_freq)
    
    

    # Create datasets using vocabularies for token-to-ID conversion
    train_dataset = Dataset(train_src, train_tgt, src_vocab, tgt_vocab, max_len, tokenizer)
    dev_dataset = Dataset(dev_src, dev_tgt, src_vocab, tgt_vocab, max_len, tokenizer)
    test_dataset = Dataset(test_src, test_tgt, src_vocab, tgt_vocab, max_len, tokenizer)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader

# Collate function to handle batches of variable-length sequences
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    return src_batch, tgt_batch




if __name__ == '__main__':
    # Set paths and parameters
    scratch_location = "/scratch/hmnshpl/anlp_data/ted-talks-corpus"
    src_lang = "en"
    tgt_lang = "fr"
    batch_size = 64
    max_len = 100  # Maximum sequence length

    # Get DataLoader objects
    train_loader, dev_loader, test_loader = get_dataloaders(scratch_location, src_lang, tgt_lang, batch_size, max_len)

    # Now you can use `train_loader` in your training loop
    for src_batch, tgt_batch in train_loader:
        # Train the model with src_batch and tgt_batch
        print(src_batch.shape, tgt_batch.shape)
        print('*'*80)
        print('src_batch: \n', src_batch)
        print('*'*80)
        exit()
