from llama_index.core.schema import ImageDocument
import asyncio
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
import os
import logging
from pathlib import Path
import hashlib

from .database import SQLiteDB
from .settings import CustomFormatter
from .settings import Model
from .embedding_service import embedding_service # Added
from .vector_store_service import vector_store_service # Added
import shutil
import sqlite3 # Added
import json # Added

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
from asyncio import Queue

db = SQLiteDB()


async def summarize_document(
    doc: Document, 
    file_info_item: dict, # Contains full_text, text_chunks, etc.
    model: Model, 
    custom_prompt: str | None = None, 
    progress_queue: Queue | None = None,
    research_topic_prompt: str | None = None,
    quick_topic_analysis_enabled: bool = False,
    full_doc_topic_analysis_enabled: bool = False
):
    file_path = doc.metadata.get('file_path', 'Unknown file')
    logger.info(f"Processing file {file_path} for summarization and analysis.")
    if progress_queue:
        progress_queue.put_nowait({"type": "file_processing_start", "file": file_path, "stage": "summarization_db_check"})

    doc_hash = get_file_hash(file_path) # Assuming get_file_hash is synchronous
    summary_text = None
    analysis_data_from_db = None
    
    # Result dictionary to hold all data for this file
    result_data = {
        "file_path": file_path,
        "summary": None,
        "research_topic": None,
        "is_topic_relevant": None,
        "sub_topics": None,
        "topic_connections": None,
        "analysis_type": None,
        "file_analysis_hash": None
    }

    # 1. Basic Summarization (from initial_summary_text, which is doc.text)
    if db.is_file_exist(file_path, doc_hash): # Checks if file_hash matches
        summary_text = db.get_file_summary(file_path)
        logger.info(f"Summary for {file_path} (hash: {doc_hash}) found in DB.")
        if progress_queue:
            progress_queue.put_nowait({"type": "file_processing_update", "file": file_path, "status": "summary_from_db"})
    else:
        logger.info(f"No existing summary or hash mismatch for {file_path}. Generating new summary.")
        if progress_queue:
            progress_queue.put_nowait({"type": "file_processing_update", "file": file_path, "status": "llm_summarization_start"})
        
        summary_text = await model.summarize_document_api(
            doc.text, # doc.text is initial_summary_text
            custom_prompt=custom_prompt, 
            progress_queue=progress_queue, 
            file_path_for_logging=file_path
        )
        # Store initial summary
        db.insert_file_summary(file_path, doc_hash, summary_text) 
        if progress_queue:
            progress_queue.put_nowait({"type": "file_processing_update", "file": file_path, "status": "llm_summarization_complete"})
    
    result_data["summary"] = summary_text

    # 2. Topic Analysis
    if research_topic_prompt and (quick_topic_analysis_enabled or full_doc_topic_analysis_enabled):
        logger.info(f"Proceeding with topic analysis for {file_path}. Topic: '{research_topic_prompt[:50]}...'")
        analysis_type_str = ""
        text_for_topic_analysis = ""
        
        if quick_topic_analysis_enabled:
            analysis_type_str = "summary"
            text_for_topic_analysis = summary_text if summary_text else "" # Use the generated summary
            logger.info(f"Using summary text for 'quick' topic analysis of {file_path}.")
        elif full_doc_topic_analysis_enabled:
            analysis_type_str = "full_doc"
            text_for_topic_analysis = file_info_item.get('full_text', '')
            logger.info(f"Using full document text for 'in-depth' topic analysis of {file_path}.")

        if analysis_type_str and text_for_topic_analysis.strip():
            if progress_queue:
                progress_queue.put_nowait({"type": "file_processing_update", "file": file_path, "status": f"topic_analysis_start_{analysis_type_str}"})

            current_file_analysis_hash = hashlib.sha256(
                (doc_hash + research_topic_prompt + analysis_type_str).encode()
            ).hexdigest()

            # Check DB for existing, up-to-date analysis
            analysis_data_from_db = db.get_file_analysis_data(file_path, current_file_analysis_hash)

            if analysis_data_from_db:
                logger.info(f"Up-to-date '{analysis_type_str}' analysis for {file_path} found in DB.")
                result_data.update(analysis_data_from_db)
                if progress_queue:
                    progress_queue.put_nowait({"type": "file_processing_update", "file": file_path, "status": f"topic_analysis_from_db_{analysis_type_str}"})
            else:
                logger.info(f"No up-to-date '{analysis_type_str}' analysis for {file_path}. Calling LLM.")
                if progress_queue:
                    progress_queue.put_nowait({"type": "file_processing_update", "file": file_path, "status": f"llm_topic_analysis_start_{analysis_type_str}"})

                analysis_results_llm = await model.analyze_text_for_topic_api(
                    text_content=text_for_topic_analysis,
                    research_topic=research_topic_prompt,
                    analysis_type=analysis_type_str,
                    progress_queue=progress_queue,
                    file_path_for_logging=file_path
                )

                if analysis_results_llm:
                    logger.info(f"LLM analysis successful for {file_path} ({analysis_type_str}).")
                    analysis_data_to_store = {
                        "research_topic": research_topic_prompt,
                        "is_topic_relevant": analysis_results_llm.get("is_relevant"),
                        "sub_topics": analysis_results_llm.get("sub_topics"),
                        "topic_connections": analysis_results_llm.get("connections"),
                        "analysis_llm_prompt": research_topic_prompt, # Storing the user's raw topic
                        "analysis_type": analysis_type_str,
                        "file_analysis_hash": current_file_analysis_hash
                    }
                    # Update DB with new analysis data AND existing summary
                    db.insert_file_summary(file_path, doc_hash, summary_text, **analysis_data_to_store)
                    result_data.update(analysis_data_to_store)
                    if progress_queue:
                        progress_queue.put_nowait({"type": "file_processing_update", "file": file_path, "status": f"llm_topic_analysis_complete_{analysis_type_str}"})
                else:
                    logger.warning(f"LLM analysis returned no results for {file_path} ({analysis_type_str}).")
                    if progress_queue:
                        progress_queue.put_nowait({"type": "file_processing_update", "file": file_path, "status": f"llm_topic_analysis_failed_{analysis_type_str}"})
        else:
            logger.info(f"Skipping topic analysis for {file_path} due to missing analysis type or text.")
    else:
        logger.info(f"Topic analysis not enabled or no research topic provided for {file_path}.")
        if summary_text and db.is_file_exist(file_path, doc_hash): 
             pass 

    if progress_queue:
        progress_queue.put_nowait({"type": "file_processing_end", "file": file_path, "stage": "summarization_and_analysis"})
    
    return result_data


async def summarize_image_document(
    doc: ImageDocument, 
    model: Model, 
    progress_queue: Queue | None = None,
    research_topic_prompt: str | None = None, 
    quick_topic_analysis_enabled: bool = False, 
    full_doc_topic_analysis_enabled: bool = False 
):
    image_path = doc.image_path
    logger.info(f"Processing image {image_path} for summarization.")
    if research_topic_prompt:
        logger.debug(f"Image summarization for {image_path} received research_topic_prompt: '{research_topic_prompt[:30]}...' "
                     f"Q:{quick_topic_analysis_enabled} F:{full_doc_topic_analysis_enabled} (analysis not applied to images).")

    if progress_queue:
        progress_queue.put_nowait({"type": "file_processing_start", "file": image_path, "stage": "image_summarization_db_check"})

    image_hash = get_file_hash(image_path)
    summary_text = None
    
    result_data = {"file_path": image_path, "summary": None} 

    if db.is_file_exist(image_path, image_hash):
        summary_text = db.get_file_summary(image_path)
        if progress_queue:
            progress_queue.put_nowait({"type": "file_processing_update", "file": image_path, "status": "image_summary_from_db"})
    else:
        if progress_queue:
            progress_queue.put_nowait({"type": "file_processing_update", "file": image_path, "status": "llm_image_summarization_start"})
        summary_text = await model.summarize_image_api(image_path=image_path, progress_queue=progress_queue)
        db.insert_file_summary(image_path, image_hash, summary_text) 
        if progress_queue:
            progress_queue.put_nowait({"type": "file_processing_update", "file": image_path, "status": "llm_image_summarization_complete"})

    result_data["summary"] = summary_text
    if progress_queue:
        progress_queue.put_nowait({"type": "file_processing_end", "file": image_path, "stage": "image_summarization"})
    return result_data


async def dispatch_summarize_document(
    doc_for_summary: Document, 
    file_info_item: dict,      
    model: Model, 
    custom_prompt: str | None = None, 
    progress_queue: Queue | None = None,
    research_topic_prompt: str | None = None,
    quick_topic_analysis_enabled: bool = False,
    full_doc_topic_analysis_enabled: bool = False
):
    file_path_for_log_dispatch = file_info_item.get('file_path', 'N/A')
    if progress_queue and hasattr(progress_queue, '_cancelled_event') and progress_queue._cancelled_event.is_set():
         logger.info(f"Summarization dispatch cancelled for doc: {file_path_for_log_dispatch}")
         raise asyncio.CancelledError("Summarization dispatch cancelled by client request.")

    if isinstance(doc_for_summary, ImageDocument):
        return await summarize_image_document(
            doc_for_summary, model, progress_queue=progress_queue,
            research_topic_prompt=research_topic_prompt, 
            quick_topic_analysis_enabled=quick_topic_analysis_enabled, 
            full_doc_topic_analysis_enabled=full_doc_topic_analysis_enabled 
        )
    elif isinstance(doc_for_summary, Document): 
        return await summarize_document(
            doc_for_summary, file_info_item, model=model, 
            custom_prompt=custom_prompt, progress_queue=progress_queue,
            research_topic_prompt=research_topic_prompt,
            quick_topic_analysis_enabled=quick_topic_analysis_enabled,
            full_doc_topic_analysis_enabled=full_doc_topic_analysis_enabled
        )
    else:
        logger.warning(f"Unsupported document type encountered in dispatch: {type(doc_for_summary)} for {file_path_for_log_dispatch}")
        return None


async def get_summaries(
    docs_and_file_info_list: list[tuple[Document, dict]], 
    model: Model, 
    custom_prompt: str | None = None, 
    progress_queue: Queue | None = None,
    research_topic_prompt: str | None = None,
    quick_topic_analysis_enabled: bool = False,
    full_doc_topic_analysis_enabled: bool = False
):
    docs_summaries_and_analysis = [] 
    total_items = len(docs_and_file_info_list)
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "message": f"Starting summarization and analysis of {total_items} documents."})

    for i, (doc_for_summary, file_info_item) in enumerate(docs_and_file_info_list):
        file_path_for_log = file_info_item.get('file_path', doc_for_summary.metadata.get('file_path', 'Unknown file'))
        
        if progress_queue:
            progress_queue.put_nowait({
                "type": "progress_update", 
                "current_doc_index": i + 1, 
                "total_docs": total_items, 
                "file": file_path_for_log, 
                "status": "processing_dispatch_started" 
            })
        
        processed_data = await dispatch_summarize_document(
            doc_for_summary, file_info_item, model=model, 
            custom_prompt=custom_prompt, progress_queue=progress_queue,
            research_topic_prompt=research_topic_prompt,
            quick_topic_analysis_enabled=quick_topic_analysis_enabled,
            full_doc_topic_analysis_enabled=full_doc_topic_analysis_enabled
        )
        if processed_data: 
            docs_summaries_and_analysis.append(processed_data)

        if progress_queue:
            progress_queue.put_nowait({
                "type": "progress_update", 
                "current_doc_index": i + 1,
                "total_docs": total_items, 
                "file": file_path_for_log, 
                "status": "processing_dispatch_completed"
            })
    return docs_summaries_and_analysis


async def remove_deleted_files():
    file_paths = db.get_all_files()
    deleted_file_paths = [file_path for file_path in file_paths if not os.path.exists(file_path)]
    db.delete_records(deleted_file_paths)
    if deleted_file_paths:
        logger.info(f"Attempting to delete chunks from vector store for {len(deleted_file_paths)} deleted files.")
        deleted_from_chroma_count = 0
        for file_path in deleted_file_paths:
            try:
                if vector_store_service.delete_chunks_for_file(file_path):
                    logger.debug(f"Successfully deleted chunks for {file_path} from vector store.")
                    deleted_from_chroma_count +=1
                else:
                    logger.warning(f"Failed to delete chunks for {file_path} from vector store or file not found there.")
            except Exception as e_chroma_del:
                logger.error(f"Error deleting chunks from vector store for {file_path}: {e_chroma_del}", exc_info=True)
        logger.info(f"Completed deletion from vector store. Successfully deleted chunks for {deleted_from_chroma_count}/{len(deleted_file_paths)} files.")


def load_documents(path: str, recursive: bool, required_exts: list):
    logger.info(f"Attempting to load documents from {path} with extensions: {required_exts}")
    splitter = TokenTextSplitter(chunk_size=2048, chunk_overlap=200) 
    processed_files_data = [] 
    INITIAL_SUMMARY_CHAR_LIMIT = 8000 

    if required_exts: # Modified: Check if there are any extensions to process
        logger.info(f"Processing file types {required_exts} with SimpleDirectoryReader.")
        aggregated_file_contents = {}
        try:
            # Modified: SimpleDirectoryReader now handles all required_exts
            reader = SimpleDirectoryReader(
                input_dir=path, recursive=recursive, required_exts=required_exts, errors='warn' 
            )
            for doc_chunk_list in reader.iter_data(): 
                for doc_obj in doc_chunk_list: 
                    file_path_for_log = doc_obj.metadata.get('file_path', 'Unknown_file')
                    if file_path_for_log == 'Unknown_file':
                        logger.warning(f"Document object missing 'file_path' in metadata. Skipping.")
                        continue
                    if file_path_for_log not in aggregated_file_contents:
                        aggregated_file_contents[file_path_for_log] = {
                            "texts": [], "metadata": doc_obj.metadata
                        }
                    if doc_obj.text and doc_obj.text.strip():
                        aggregated_file_contents[file_path_for_log]["texts"].append(doc_obj.text)

            for file_path_str, content_data in aggregated_file_contents.items():
                logger.info(f"Processing aggregated content for file: {file_path_str}")
                full_text = "\n".join(content_data["texts"])
                current_metadata = content_data["metadata"] 
                text_chunks = splitter.split_text(full_text) if full_text.strip() else []
                if not full_text.strip():
                    logger.warning(f"File {file_path_str} has no text content after aggregation.")
                elif not text_chunks and full_text.strip(): # If there's text but no chunks, use full text as a chunk
                    logger.warning(f"Splitting text for file {file_path_str} resulted in no chunks, using full text as one chunk.")
                    text_chunks = [full_text]

                initial_summary_text = full_text[:INITIAL_SUMMARY_CHAR_LIMIT]
                processed_files_data.append({
                    "file_path": file_path_str, "full_text": full_text, "text_chunks": text_chunks,
                    "initial_summary_text": initial_summary_text, "metadata": current_metadata
                })
                logger.debug(f"Successfully processed file: {file_path_str}, {len(text_chunks)} chunks created.")
        except Exception as e:
            # Modified: Generalized error message
            logger.error(f"Error during SimpleDirectoryReader processing for path {path} with extensions {required_exts}: {e}", exc_info=True)
    
    # Modified: Simplified warning condition
    if not processed_files_data and required_exts: 
        logger.warning(f"No documents were loaded from {path} with extensions {required_exts}. Check path and file types.")
    elif not required_exts:
        logger.warning(f"No required extensions specified for path {path}. No files processed.")
    return processed_files_data


async def get_dir_summaries(
    path: str, recursive: bool, required_exts: list, model: Model, 
    custom_prompt: str | None = None, progress_queue: Queue | None = None,
    research_topic_prompt: str | None = None,
    quick_topic_analysis_enabled: bool = False,
    full_doc_topic_analysis_enabled: bool = False
):
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "message": "Loading documents..."})
    
    loaded_file_info_list = load_documents(path, recursive, required_exts) 
    
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "total_docs_loaded": len(loaded_file_info_list), "message": "Documents loaded and processed."})
        if not loaded_file_info_list:
             progress_queue.put_nowait({"type": "status", "message": "No documents found to summarize."})
             return [] 

    docs_and_file_info_for_processing = []
    for file_info_item in loaded_file_info_list:
        doc_for_summary = Document(
            text=file_info_item["initial_summary_text"], 
            metadata=file_info_item["metadata"] 
        )
        docs_and_file_info_for_processing.append((doc_for_summary, file_info_item))


    await remove_deleted_files() 
    
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "total_docs_for_summary": len(docs_and_file_info_for_processing), "message": "Starting summarization and analysis process..."})

    all_processed_data = await get_summaries(
        docs_and_file_info_for_processing, model=model, 
        custom_prompt=custom_prompt, progress_queue=progress_queue,
        research_topic_prompt=research_topic_prompt,
        quick_topic_analysis_enabled=quick_topic_analysis_enabled,
        full_doc_topic_analysis_enabled=full_doc_topic_analysis_enabled
    )

    for item_data in all_processed_data: 
        if 'file_path' in item_data: 
             item_data["file_path"] = os.path.relpath(item_data["file_path"], path)
    
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "message": "Summarization and analysis complete for all documents."})
    return all_processed_data


async def run(
    directory_path: str, 
    recursive: bool, 
    required_exts: list, 
    llm_provider: str = "openai", 
    ollama_api_base_url: str | None = None, 
    custom_summarization_prompt: str | None = None,
    progress_queue: Queue | None = None,
    research_topic_prompt: str | None = None,
    quick_topic_analysis_enabled: bool = False,
    full_doc_topic_analysis_enabled: bool = False,
    semantic_search_enabled: bool = False 
):
    try:
        logger.info(f"Starting FileWizardAI run with analysis options: research_topic='{research_topic_prompt is not None}', "
                    f"quick_analysis={quick_topic_analysis_enabled}, full_analysis={full_doc_topic_analysis_enabled}, "
                    f"semantic_search={semantic_search_enabled}")
        if progress_queue:
            progress_queue.put_nowait({"type": "status", "message": "Initializing FileWizardAI..."})

        model = Model(llm_provider=llm_provider, ollama_api_base_url=ollama_api_base_url)
        
        if progress_queue:
            progress_queue.put_nowait({"type": "status", "message": "Starting directory analysis, document summarization, and topic analysis..."})
        
        processed_file_data_list = await get_dir_summaries(
            directory_path, recursive, required_exts, model=model, 
            custom_prompt=custom_summarization_prompt, progress_queue=progress_queue,
            research_topic_prompt=research_topic_prompt,
            quick_topic_analysis_enabled=quick_topic_analysis_enabled,
            full_doc_topic_analysis_enabled=full_doc_topic_analysis_enabled
        )
        
        if not processed_file_data_list: 
            logger.info("No files processed or summaries generated, concluding run.")
            if progress_queue:
                progress_queue.put_nowait({"type": "status", "message": "No files processed. File tree generation skipped."})
            return [] 

        if progress_queue:
            progress_queue.put_nowait({"type": "status", "message": "Generating file tree based on summaries..."})
        
        summaries_for_tree_generation = [
            {"file_path": item.get("file_path"), "summary": item.get("summary")}
            for item in processed_file_data_list if item.get("file_path") and item.get("summary")
        ]

        files_for_tree_output = await model.create_file_tree_api(summaries_for_tree_generation, progress_queue=progress_queue)
        
        if progress_queue:
            progress_queue.put_nowait({"type": "status", "message": "File tree generation complete."})
        
        if semantic_search_enabled and processed_file_data_list:
            logger.info(f"Semantic search enabled. Starting embedding pipeline for {len(processed_file_data_list)} files.")
            if progress_queue:
                progress_queue.put_nowait({"type": "status", "message": "Starting embedding generation and storage..."})
            
            embedded_files_count = 0
            for file_idx, file_data in enumerate(processed_file_data_list):
                relative_file_path = file_data.get('file_path') 
                if not relative_file_path:
                    logger.warning(f"Skipping embedding for file_data at index {file_idx} due to missing 'file_path'.")
                    continue

                absolute_file_path = os.path.join(directory_path, relative_file_path)
                
                logger.info(f"Processing file for embedding: {absolute_file_path}")
                if progress_queue:
                    progress_queue.put_nowait({
                        "type": "embedding_progress", 
                        "current_file_num": file_idx + 1,
                        "total_files": len(processed_file_data_list),
                        "file": relative_file_path,
                        "status": "starting_chunk_processing"
                    })

                text_chunks_for_file = file_data.get('text_chunks', [])
                if not text_chunks_for_file:
                    logger.info(f"No text chunks found for {absolute_file_path}. Skipping embedding for this file.")
                    db.insert_document_chunks(absolute_file_path, []) 
                    vector_store_service.delete_chunks_for_file(absolute_file_path)
                    continue

                db_chunk_objects = [{'chunk_order': i, 'chunk_text': text} for i, text in enumerate(text_chunks_for_file)]
                db.insert_document_chunks(absolute_file_path, db_chunk_objects) 

                persisted_chunks = db.get_document_chunks(absolute_file_path)
                if not persisted_chunks:
                    logger.warning(f"Failed to retrieve persisted chunks from SQLite for {absolute_file_path}. Skipping embedding for this file.")
                    continue
                
                chroma_chunks_data = []
                processed_chunks_for_file = 0
                for p_chunk in persisted_chunks:
                    chunk_text = p_chunk.get('chunk_text')
                    chunk_id_db = p_chunk.get('id')

                    if not chunk_text or chunk_id_db is None:
                        logger.warning(f"Skipping a chunk for {absolute_file_path} due to missing text or DB ID.")
                        continue
                    
                    if not embedding_service or not embedding_service.model:
                        logger.error("Embedding service or model not available. Halting embedding pipeline.")
                        if progress_queue: progress_queue.put_nowait({"type": "error", "message": "Embedding service not available."})
                        return files_for_tree_output 
                        
                    embedding_vector = embedding_service.get_embedding(chunk_text)

                    if embedding_vector:
                        try:
                            embedding_json_bytes = json.dumps(embedding_vector).encode('utf-8')
                            db.update_chunk_embedding(chunk_id_db, sqlite3.Binary(embedding_json_bytes))
                        except Exception as e_sql_update:
                            logger.error(f"Error updating SQLite chunk {chunk_id_db} with embedding for {absolute_file_path}: {e_sql_update}", exc_info=True)
                            continue 

                        chroma_chunks_data.append({
                            'chunk_id_db': chunk_id_db, 
                            'chunk_text': chunk_text,
                            'embedding': embedding_vector
                        })
                        processed_chunks_for_file +=1
                    else:
                        logger.warning(f"Failed to generate embedding for chunk ID {chunk_id_db} of file {absolute_file_path}.")
                
                if chroma_chunks_data:
                    logger.info(f"Adding {len(chroma_chunks_data)} chunk embeddings to vector store for {absolute_file_path}.")
                    add_to_chroma_success = vector_store_service.add_chunk_embeddings(absolute_file_path, chroma_chunks_data)
                    if add_to_chroma_success:
                        logger.info(f"Successfully added embeddings to vector store for {absolute_file_path}.")
                        embedded_files_count += 1
                    else:
                        logger.error(f"Failed to add embeddings to vector store for {absolute_file_path}.")
                elif text_chunks_for_file : 
                     logger.warning(f"No valid embeddings generated to store in ChromaDB for {absolute_file_path} despite having text chunks.")
                
                if progress_queue:
                     progress_queue.put_nowait({
                        "type": "embedding_progress",
                        "current_file_num": file_idx + 1,
                        "total_files": len(processed_file_data_list),
                        "file": relative_file_path,
                        "status": "completed_chunk_processing",
                        "processed_chunks_for_file": processed_chunks_for_file,
                        "total_chunks_in_file": len(text_chunks_for_file)
                    })

            logger.info(f"Embedding pipeline completed. Successfully processed and stored embeddings for {embedded_files_count}/{len(processed_file_data_list)} files.")
            if progress_queue:
                progress_queue.put_nowait({"type": "status", "message": f"Embedding generation and storage complete for {embedded_files_count} files."})
        elif semantic_search_enabled and not processed_file_data_list:
            logger.info("Semantic search enabled, but no files were processed to generate embeddings for.")
            if progress_queue:
                progress_queue.put_nowait({"type": "status", "message": "No files processed; embedding step skipped."})

        logger.info("FileWizardAI run completed successfully.")
        if progress_queue: 
            progress_queue.put_nowait({"type": "status", "message": "Process completed successfully."})
        return files_for_tree_output 
    
    except asyncio.CancelledError:
        logger.info("FileWizardAI run was cancelled.")
        if progress_queue:
            try:
                progress_queue.put_nowait({"type": "cancelled", "message": "Task cancelled during core run execution."})
            except Exception as e_q: # pragma: no cover
                logger.error(f"Failed to put cancellation message in queue: {e_q}")
        raise 
    except Exception as e:
        logger.error(f"Error during FileWizardAI run: {e}", exc_info=True)
        if progress_queue:
            try:
                progress_queue.put_nowait({"type": "error", "message": f"An unexpected error occurred in core run: {str(e)}"})
            except Exception as e_q: # pragma: no cover
                logger.error(f"Failed to put error message in queue: {e_q}")
        raise 


def update_file(root_path, item):
    src_file = root_path + "/" + item["src_path"]
    dst_file = root_path + "/" + item["dst_path"]
    dst_dir = os.path.dirname(dst_file)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if os.path.isfile(src_file):
        shutil.move(src_file, dst_file)
        new_hash = get_file_hash(dst_file) 
        db.update_file(src_file, dst_file, new_hash)


async def search_files(
    root_path: str, 
    recursive: bool, 
    required_exts: list, 
    search_query: str, 
    llm_provider: str = "openai", 
    ollama_api_base_url: str | None = None, 
    custom_summarization_prompt: str | None = None,
    progress_queue: Queue | None = None,
    research_topic_prompt: str | None = None,
    quick_topic_analysis_enabled: bool = False,
    full_doc_topic_analysis_enabled: bool = False 
):
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "message": "Search process started..."})

    model = Model(llm_provider=llm_provider, ollama_api_base_url=ollama_api_base_url)
    
    processed_file_data_list = await get_dir_summaries(
        root_path, recursive, required_exts, model=model, 
        custom_prompt=custom_summarization_prompt, progress_queue=progress_queue,
        research_topic_prompt=research_topic_prompt,
        quick_topic_analysis_enabled=quick_topic_analysis_enabled,
        full_doc_topic_analysis_enabled=full_doc_topic_analysis_enabled
    )
    
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "message": "Searching through summaries..."})
    
    summaries_for_search = [
        {"file_path": item.get("file_path"), "summary": item.get("summary")}
        for item in processed_file_data_list if item.get("file_path") and item.get("summary")
    ]
        
    files_found = await model.search_files_api(summaries_for_search, search_query, progress_queue=progress_queue) 
    
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "message": "Search process completed."})
        
    return files_found


def get_file_hash(file_path):
    # It's important that this function is robust to file not existing during hashing
    # e.g. if a file is deleted between discovery and processing.
    try:
        hash_func = hashlib.new('sha256')
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except FileNotFoundError:
        logger.error(f"File not found during hash calculation: {file_path}")
        return None # Or raise an error, or return a specific sentinel value
    except Exception as e: # pragma: no cover
        logger.error(f"Error calculating hash for {file_path}: {e}", exc_info=True)
        return None
