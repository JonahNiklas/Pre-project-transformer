import numpy as np

import logging
logger = logging.getLogger(__name__)

# Load GloVe embeddings
def load_glove(file_path):
    logger.info(f"Loading GloVe embeddings from {file_path}")
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    logger.info(f"GloVe embeddings loaded successfully")
    return embeddings

glove_path = './glove/glove.6B.300d.txt'
glove_embeddings = load_glove(glove_path)

def get_sentence_embedding(sentence):
    words = sentence.split()
    word_embeddings = [glove_embeddings[word] for word in words if word in glove_embeddings]
    if not word_embeddings:
        return np.zeros(300)  # Return zero vector if no words are found in GloVe
    return np.array(word_embeddings)

if __name__ == "__main__":
    # Example usage
    sentence = "The quick brown fox jumps over the lazy dog."
    embedding = get_sentence_embedding(sentence)
    print(f"Sentence embedding shape: {embedding.shape}")
    print(f"Sentence embedding: {embedding[:5]}...")  # Print first 5 values

