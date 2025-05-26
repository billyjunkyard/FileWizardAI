import logging
from sentence_transformers import SentenceTransformer
import numpy as np

# Setup logger
logger = logging.getLogger(__name__)
# Configure logger if not already configured by the main application
if not logger.handlers:
    logging.basicConfig(level=logging.INFO) # Default to INFO if no handlers

class EmbeddingService:
    """
    Service for generating text embeddings using a sentence-transformer model.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the EmbeddingService and loads the sentence transformer model.

        Args:
            model_name (str): The name of the sentence-transformer model to use.
        """
        self.model_name = model_name
        self.model = None
        try:
            logger.info(f"Attempting to load embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model '{self.model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}'. Error: {e}", exc_info=True)
            # Depending on application requirements, you might want to raise the error
            # or handle it in a way that the application can still run without embeddings.
            # For now, we log the error and self.model will remain None.

    def get_embedding(self, text: str) -> list[float] | None:
        """
        Generates an embedding for the given text.

        Args:
            text (str): The text to embed.

        Returns:
            list[float] | None: The embedding vector as a list of floats, 
                                or None if the model is not loaded or text is empty.
        """
        if not self.model:
            logger.error("Embedding model is not loaded. Cannot generate embedding.")
            return None
        
        if not text or not text.strip():
            logger.warning("Input text for embedding is empty or whitespace.")
            # Depending on requirements, could return None, empty list, or a zero vector.
            # Returning None for now to indicate an issue.
            return None

        try:
            # The encode method can take a single sentence or a list of sentences.
            # normalize_embeddings=True is often recommended for sentence similarity tasks.
            embedding_np = self.model.encode(text, normalize_embeddings=True)
            
            if isinstance(embedding_np, np.ndarray):
                return embedding_np.tolist()
            else:
                # Should not happen with standard SentenceTransformer usage for single string
                logger.error(f"Model encode() did not return a NumPy array. Type: {type(embedding_np)}")
                return None
        except Exception as e:
            logger.error(f"Error generating embedding for text: '{text[:50]}...'. Error: {e}", exc_info=True)
            return None

# Singleton instance of the EmbeddingService
# This ensures the model is loaded only once when the module is imported.
embedding_service = EmbeddingService()

# Example of how to get an embedding (for testing purposes if this file is run directly)
if __name__ == '__main__':
    logger.info("Running embedding_service.py directly for testing.")
    
    test_sentence = "This is a test sentence for the embedding service."
    
    # Check if model loaded
    if embedding_service.model:
        vector = embedding_service.get_embedding(test_sentence)
        if vector:
            logger.info(f"Test sentence: {test_sentence}")
            logger.info(f"Embedding (first 10 dims): {vector[:10]}")
            logger.info(f"Embedding dimension: {len(vector)}")
        else:
            logger.error("Failed to get embedding for the test sentence.")
            
        empty_vector = embedding_service.get_embedding("")
        logger.info(f"Embedding for empty string: {empty_vector}")

        none_vector = embedding_service.get_embedding(None) # type: ignore
        logger.info(f"Embedding for None: {none_vector}")

    else:
        logger.error("Embedding model failed to load. Cannot perform tests.")
