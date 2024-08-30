## To use pretrained embedding

import numpy as np
import time

def load_glove_embeddings(glove_file_path='/scratch/hmnshpl/anlp_data/glove.6B.300d.txt', embedding_dim=300):
    """
    Loads GloVe word embeddings from a file into a dictionary.
    """
    embedding_model = {}
    
    with open(glove_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            embedding_vector = np.asarray(values[1:], dtype='float32')
            if len(embedding_vector) == embedding_dim:
                embedding_model[word] = embedding_vector
    
    return embedding_model

if __name__ == '__main__':

    # Example usage
    t0 = time.time()
    glove_file_path = '/scratch/hmnshpl/anlp_data/glove.6B.300d.txt' 
    embedding_dim = 300

    embedding_model = load_glove_embeddings(glove_file_path, embedding_dim)
    print(f"Loaded {len(embedding_model)} word vectors.")
    print(f'took {time.time() - t0}secs.')
