import numpy as np
import logging
from constants import transformer_config, embedding_dimension


logger = logging.getLogger(__name__)


# Load GloVe embeddings
def load_glove(file_path):
    logger.info(f"Loading GloVe embeddings from {file_path}")
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    logger.info(f"GloVe embeddings loaded successfully")
    return embeddings


glove_path = f"./glove/glove.6B.{embedding_dimension}d.txt"
glove_embeddings = load_glove(glove_path)


def embed_descriptions(descriptions):
    embeddings = np.array([get_sentence_embedding(desc) for desc in descriptions])
    assert embeddings.shape == (
        len(descriptions),
        transformer_config.max_seq_length,
        embedding_dimension,
    )
    return embeddings


def get_sentence_embedding(description):
    description = description[: transformer_config.max_seq_length]
    words = description.split()
    word_embeddings = [
        (
            glove_embeddings[word]
            if word in glove_embeddings
            else np.zeros(embedding_dimension)
        )
        for word in words
    ]
    if not word_embeddings:
        raise Exception(f"No embeddings found for description: {description}")
    word_embeddings = np.pad(
        word_embeddings,
        ((0, transformer_config.max_seq_length - len(word_embeddings)), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return np.array(word_embeddings)


if __name__ == "__main__":
    # Example usage
    sentence = "The quick brown fox jumps over the lazy dog."
    embedding = get_sentence_embedding(sentence)
    print(f"Sentence embedding shape: {embedding.shape}")
    print(f"Sentence embedding: {embedding[:5]}...")  # Print first 5 values
