import chromadb
import logging
import os

# Setup logger
logger = logging.getLogger(__name__)
# Configure logger if not already configured by the main application
if not logger.handlers:
    logging.basicConfig(level=logging.INFO) # Default to INFO if no handlers

# Define the path for ChromaDB persistence.
# This will create the 'chroma_db_store' directory in the same directory as this script (backend/app/).
# For production, you might want a more configurable path or one outside the app directory.
CHROMA_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db_store")
THESIS_DOC_CHUNKS_COLLECTION = "thesis_document_chunks"

class VectorStoreService:
    """
    Service for managing and querying text chunk embeddings using ChromaDB.
    """
    def __init__(self, path: str = CHROMA_DATA_PATH, collection_name: str = THESIS_DOC_CHUNKS_COLLECTION):
        """
        Initializes the VectorStoreService, ChromaDB client, and collection.

        Args:
            path (str): The path to the directory where ChromaDB should persist data.
            collection_name (str): The name of the collection to use.
        """
        self.collection_name = collection_name
        try:
            logger.info(f"Initializing ChromaDB client with path: {path}")
            if not os.path.exists(path):
                os.makedirs(path)
                logger.info(f"Created ChromaDB data directory: {path}")
            self.client = chromadb.PersistentClient(path=path)
            
            logger.info(f"Getting or creating ChromaDB collection: {self.collection_name}")
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logger.info(f"Successfully initialized VectorStoreService with collection '{self.collection_name}'.")
            logger.info(f"Collection contains {self.collection.count()} items.")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client or collection '{self.collection_name}'. Error: {e}", exc_info=True)
            self.client = None
            self.collection = None
            # Depending on application requirements, you might want to raise the error
            # to prevent the application from starting if the vector store is critical.

    def add_chunk_embeddings(self, file_path: str, chunks_data: list[dict]) -> bool:
        """
        Adds chunk embeddings to the ChromaDB collection.
        Deletes existing chunks for the given file_path before adding new ones.

        Args:
            file_path (str): The path of the file these chunks belong to.
            chunks_data (list[dict]): A list of dictionaries, each containing:
                - 'chunk_id_db': ID from the SQLite document_chunks table.
                - 'chunk_text': Text of the chunk.
                - 'embedding': Vector embedding (list[float]).

        Returns:
            bool: True if successful, False otherwise.
        """
        if not self.collection:
            logger.error("ChromaDB collection is not available. Cannot add embeddings.")
            return False

        if not chunks_data:
            logger.info(f"No chunks data provided for file_path: {file_path}. Nothing to add.")
            # Delete existing chunks if no new chunks are provided, to maintain consistency
            return self.delete_chunks_for_file(file_path)


        # 1. Delete existing chunks for this file_path
        try:
            logger.info(f"Deleting existing chunks for file_path: {file_path} before adding new ones.")
            # Check if there's anything to delete first to avoid benign errors if file_path not found
            existing_chunks = self.collection.get(where={"file_path": file_path}, include=[]) # only need ids
            if existing_chunks and existing_chunks['ids']:
                 self.collection.delete(where={"file_path": file_path})
                 logger.info(f"Successfully deleted {len(existing_chunks['ids'])} existing chunk(s) for {file_path}.")
            else:
                logger.info(f"No existing chunks found for {file_path} to delete.")
        except Exception as e_del:
            logger.error(f"Error deleting existing chunks for file_path {file_path}: {e_del}", exc_info=True)
            # Depending on policy, might want to fail here or proceed with caution
            # For now, proceed to add, which might lead to duplicates if delete failed partially for some reason

        # 2. Prepare data for new batch insertion
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in chunks_data:
            if not all(k in chunk for k in ['chunk_id_db', 'chunk_text', 'embedding']):
                logger.warning(f"Skipping chunk due to missing keys: {chunk}. Required: 'chunk_id_db', 'chunk_text', 'embedding'.")
                continue
            if not isinstance(chunk['embedding'], list) or not all(isinstance(x, float) for x in chunk['embedding']):
                logger.warning(f"Skipping chunk due to invalid embedding format for chunk_id_db {chunk['chunk_id_db']}. Embedding must be list[float].")
                continue


            chroma_id = f"{file_path}_{chunk['chunk_id_db']}"
            ids.append(chroma_id)
            documents.append(chunk['chunk_text'])
            embeddings.append(chunk['embedding'])
            metadatas.append({
                "file_path": file_path,
                "chunk_id_db": str(chunk['chunk_id_db']) # Ensure it's a string if ChromaDB expects basic types
            })
        
        if not ids:
            logger.info(f"No valid chunks to add for file_path: {file_path} after validation.")
            return True # Considered success as there's nothing valid to add

        # 3. Add to collection
        try:
            logger.info(f"Adding {len(ids)} new chunks for file_path: {file_path}.")
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Successfully added {len(ids)} chunks for {file_path}. Collection now has {self.collection.count()} items.")
            return True
        except Exception as e:
            logger.error(f"Failed to add chunk embeddings for file_path {file_path}. Error: {e}", exc_info=True)
            return False

    def search_similar_chunks(self, query_embedding: list[float], top_n: int = 5, 
                              file_paths: list[str] | None = None) -> list[dict]:
        """
        Searches for chunks with embeddings similar to the query_embedding.

        Args:
            query_embedding (list[float]): The embedding to search against.
            top_n (int): The number of similar chunks to return.
            file_paths (list[str] | None): Optional list of file_paths to filter the search.

        Returns:
            list[dict]: A list of found chunks, each containing 'chunk_id_db', 
                        'file_path', 'chunk_text', and 'distance'.
        """
        if not self.collection:
            logger.error("ChromaDB collection is not available. Cannot search.")
            return []
        if not query_embedding:
            logger.warning("Query embedding is empty. Cannot perform search.")
            return []

        where_filter = None
        if file_paths:
            if len(file_paths) == 1:
                where_filter = {"file_path": file_paths[0]}
            else:
                where_filter = {"file_path": {"$in": file_paths}}
            logger.info(f"Searching with filter: {where_filter}")
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_n,
                where=where_filter,
                include=["metadatas", "documents", "distances"] 
            )
            
            processed_results = []
            if results and results['ids'] and results['ids'][0]: # Results are lists of lists for batch queries
                num_results_for_query = len(results['ids'][0])
                for i in range(num_results_for_query):
                    meta = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    doc = results['documents'][0][i] if results['documents'] and results['documents'][0] else ""
                    dist = results['distances'][0][i] if results['distances'] and results['distances'][0] else float('inf')
                    
                    processed_results.append({
                        "chunk_id_db": int(meta.get("chunk_id_db")) if meta.get("chunk_id_db") else None,
                        "file_path": meta.get("file_path"),
                        "chunk_text": doc,
                        "distance": dist
                    })
            
            logger.info(f"Search returned {len(processed_results)} chunks for query (top_n={top_n}).")
            return processed_results
        except Exception as e:
            logger.error(f"Error searching for similar chunks. Error: {e}", exc_info=True)
            return []

    def delete_chunks_for_file(self, file_path: str) -> bool:
        """
        Deletes all chunks associated with a given file_path from the ChromaDB collection.

        Args:
            file_path (str): The file_path whose chunks are to be deleted.

        Returns:
            bool: True on successful deletion or if no chunks found, False on error.
        """
        if not self.collection:
            logger.error("ChromaDB collection is not available. Cannot delete chunks.")
            return False
        try:
            # Check if there's anything to delete first
            existing_chunks = self.collection.get(where={"file_path": file_path}, include=[]) # only need ids
            if not existing_chunks or not existing_chunks['ids']:
                logger.info(f"No chunks found for file_path '{file_path}' to delete. Considered success.")
                return True

            self.collection.delete(where={"file_path": file_path})
            logger.info(f"Successfully deleted {len(existing_chunks['ids'])} chunk(s) for file_path '{file_path}'. Collection now has {self.collection.count()} items.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunks for file_path '{file_path}'. Error: {e}", exc_info=True)
            return False

# Singleton instance of the VectorStoreService
# This ensures the ChromaDB client and collection are initialized once.
vector_store_service = VectorStoreService()

# Example usage (for testing if run directly)
if __name__ == '__main__': # pragma: no cover
    logger.info("Running vector_store_service.py directly for testing.")

    if not vector_store_service.collection:
        logger.error("VectorStoreService failed to initialize. Cannot run tests.")
    else:
        # Test adding chunks
        test_file_path = "/test/document1.txt"
        test_chunks_data = [
            {"chunk_id_db": 1, "chunk_text": "This is the first test chunk.", "embedding": [0.1] * 384}, # all-MiniLM-L6-v2 has 384 dims
            {"chunk_id_db": 2, "chunk_text": "Another chunk for testing purposes.", "embedding": [0.2] * 384},
            {"chunk_id_db": 3, "chunk_text": "The final test chunk from document1.", "embedding": [0.3] * 384},
        ]
        add_success = vector_store_service.add_chunk_embeddings(test_file_path, test_chunks_data)
        logger.info(f"Add chunks for {test_file_path} success: {add_success}")
        logger.info(f"Collection count after add: {vector_store_service.collection.count()}")

        test_file_path_2 = "/test/document2.txt"
        test_chunks_data_2 = [
            {"chunk_id_db": 10, "chunk_text": "Chunk from a different document.", "embedding": [0.4] * 384},
        ]
        add_success_2 = vector_store_service.add_chunk_embeddings(test_file_path_2, test_chunks_data_2)
        logger.info(f"Add chunks for {test_file_path_2} success: {add_success_2}")
        logger.info(f"Collection count after add 2: {vector_store_service.collection.count()}")


        # Test searching
        if add_success:
            query_emb = [0.11] * 384 # Similar to the first chunk
            search_results = vector_store_service.search_similar_chunks(query_emb, top_n=2)
            logger.info(f"Search results for query similar to first chunk: {search_results}")

            # Test search with file filter
            search_results_filtered = vector_store_service.search_similar_chunks(query_emb, top_n=2, file_paths=[test_file_path])
            logger.info(f"Search results filtered for {test_file_path}: {search_results_filtered}")
            
            search_results_filtered_2 = vector_store_service.search_similar_chunks(query_emb, top_n=2, file_paths=[test_file_path_2])
            logger.info(f"Search results filtered for {test_file_path_2}: {search_results_filtered_2}")


        # Test deleting chunks
        delete_success = vector_store_service.delete_chunks_for_file(test_file_path)
        logger.info(f"Delete chunks for {test_file_path} success: {delete_success}")
        logger.info(f"Collection count after delete: {vector_store_service.collection.count()}")

        # Test deleting non-existent file path
        delete_non_existent_success = vector_store_service.delete_chunks_for_file("/test/non_existent.txt")
        logger.info(f"Delete chunks for non-existent file success: {delete_non_existent_success}")
        logger.info(f"Collection count: {vector_store_service.collection.count()}")
        
        # Clean up test file 2
        vector_store_service.delete_chunks_for_file(test_file_path_2)
        logger.info(f"Cleaned up {test_file_path_2}. Final collection count: {vector_store_service.collection.count()}")
        
        # Test adding empty list of chunks (should delete existing if any)
        vector_store_service.add_chunk_embeddings(test_file_path_2, []) # Add again then delete by providing empty list
        vector_store_service.add_chunk_embeddings(test_file_path_2, test_chunks_data_2)
        logger.info(f"Collection count after re-adding doc2: {vector_store_service.collection.count()}")
        vector_store_service.add_chunk_embeddings(test_file_path_2, [])
        logger.info(f"Collection count after adding empty chunks for doc2: {vector_store_service.collection.count()}")
