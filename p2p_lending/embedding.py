import numpy as np
import logging
from p2p_lending.constants import transformer_config, embedding_dimension
import nltk

logger = logging.getLogger(__name__)


# Load GloVe embeddings
def _load_glove(file_path: str) -> dict[str, np.ndarray]:
    logger.info(f"Loading GloVe embeddings from {file_path}")
    embeddings: dict[str, np.ndarray] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    logger.info(f"GloVe embeddings loaded successfully")
    return embeddings


glove_path = f"./p2p_lending/glove/glove.6B.{embedding_dimension}d.txt"
glove_embeddings = _load_glove(glove_path)


def embed_descriptions(descriptions: list[str]) -> np.ndarray:
    embeddings: np.ndarray = np.array([_get_sentence_embedding(desc) for desc in descriptions])

    # Calculate the percentage of words not found in embeddings
    total_words = sum(len(tokenize_text(desc)) for desc in descriptions)
    missing_words = sum(
        1
        for desc in descriptions
        for word in tokenize_text(desc)
        if word not in glove_embeddings
    )
    missing_percentage = (missing_words / total_words) * 100
    logger.debug(
        f"Percentage of words not found in embeddings: {missing_percentage:.2f}%"
    )

    assert embeddings.shape == (
        len(descriptions),
        transformer_config.max_seq_length,
        embedding_dimension,
    )
    return embeddings


def tokenize_text(text: str) -> list[str]:
    tokens: list[str] = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    return tokens


def _get_sentence_embedding(description: str) -> np.ndarray:
    tokens: list[str] = tokenize_text(description)
    words: list[str] = tokens[: transformer_config.max_seq_length]
    word_embeddings_list: list[np.ndarray] = [
        (
            glove_embeddings[word]
            if word in glove_embeddings
            else np.zeros(embedding_dimension)
        )
        for word in words
    ]
    word_embeddings = np.pad(
        word_embeddings_list,
        ((0, transformer_config.max_seq_length - len(word_embeddings_list)), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    return np.array(word_embeddings)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)
    sentence = "The quick brown fox jumps over the lazy dog."
    embedding = embed_descriptions([sentence])
    print(f"Sentence embedding shape: {embedding.shape}")
    print(f"Sentence embedding: {embedding[:5]}...")  # Print first 5 values
