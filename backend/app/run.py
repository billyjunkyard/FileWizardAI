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
import shutil

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
from asyncio import Queue

db = SQLiteDB()


async def summarize_document(doc: Document, model: Model, custom_prompt: str | None = None, progress_queue: Queue | None = None):
    file_path = doc.metadata.get('file_path', 'Unknown file')
    logger.info(f"Processing file {file_path} for summarization.")
    if progress_queue:
        progress_queue.put_nowait({"type": "file_processing_start", "file": file_path, "stage": "summarization_db_check"})

    doc_hash = get_file_hash(file_path) # Assuming get_file_hash is synchronous
    if db.is_file_exist(file_path, doc_hash):
        summary = db.get_file_summary(file_path)
        if progress_queue:
            progress_queue.put_nowait({"type": "file_processing_update", "file": file_path, "status": "summary_from_db"})
    else:
        if progress_queue:
            # Pass file_path for logging within the API call if possible
            progress_queue.put_nowait({"type": "file_processing_update", "file": file_path, "status": "llm_summarization_start"})
        summary = await model.summarize_document_api(doc.text, custom_prompt=custom_prompt, progress_queue=progress_queue, file_path_for_logging=file_path)
        db.insert_file_summary(file_path, doc_hash, summary)
        if progress_queue:
            progress_queue.put_nowait({"type": "file_processing_update", "file": file_path, "status": "llm_summarization_complete"})
    
    if progress_queue:
        progress_queue.put_nowait({"type": "file_processing_end", "file": file_path, "stage": "summarization"})
    return {
        "file_path": file_path, # Ensure this key matches what the frontend/server expects
        "summary": summary
    }


async def summarize_image_document(doc: ImageDocument, model: Model, progress_queue: Queue | None = None):
    image_path = doc.image_path
    logger.info(f"Processing image {image_path} for summarization.")
    if progress_queue:
        progress_queue.put_nowait({"type": "file_processing_start", "file": image_path, "stage": "image_summarization_db_check"})

    image_hash = get_file_hash(image_path)
    if db.is_file_exist(image_path, image_hash):
        summary = db.get_file_summary(image_path)
        if progress_queue:
            progress_queue.put_nowait({"type": "file_processing_update", "file": image_path, "status": "image_summary_from_db"})
    else:
        if progress_queue:
            progress_queue.put_nowait({"type": "file_processing_update", "file": image_path, "status": "llm_image_summarization_start"})
        summary = await model.summarize_image_api(image_path=image_path, progress_queue=progress_queue)
        db.insert_file_summary(image_path, image_hash, summary)
        if progress_queue:
            progress_queue.put_nowait({"type": "file_processing_update", "file": image_path, "status": "llm_image_summarization_complete"})

    if progress_queue:
        progress_queue.put_nowait({"type": "file_processing_end", "file": image_path, "stage": "image_summarization"})
    return {
        "file_path": image_path, # Ensure this key matches
        "summary": summary
    }


async def dispatch_summarize_document(doc, model: Model, custom_prompt: str | None = None, progress_queue: Queue | None = None):
    # Check for cancellation before processing each document
    if progress_queue and hasattr(progress_queue, '_cancelled_event') and progress_queue._cancelled_event.is_set(): # Crude check
         logger.info(f"Summarization dispatch cancelled for doc: {doc.metadata.get('file_path', 'N/A')}")
         raise asyncio.CancelledError("Summarization dispatch cancelled by client request.")

    if isinstance(doc, ImageDocument):
        return await summarize_image_document(doc, model, progress_queue=progress_queue)
    elif isinstance(doc, Document):
        return await summarize_document(doc, model=model, custom_prompt=custom_prompt, progress_queue=progress_queue)
    else:
        # This case should ideally not be reached if load_documents filters correctly
        logger.warning(f"Unsupported document type encountered in dispatch: {type(doc)}")
        return None # Or raise error


async def get_summaries(documents, model: Model, custom_prompt: str | None = None, progress_queue: Queue | None = None):
    docs_summaries = []
    total_docs = len(documents)
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "message": f"Starting summarization of {total_docs} documents."})

    for i, doc in enumerate(documents):
        # Check for cancellation before processing each document
        # A more robust way to check for cancellation might be needed if the queue doesn't expose such an event.
        # For now, we rely on exceptions bubbling up or checks within dispatch_summarize_document.
        # Consider if server.py's task cancellation is sufficient.

        file_path_for_log = doc.metadata.get('file_path', doc.image_path if isinstance(doc, ImageDocument) else 'Unknown file')
        if progress_queue:
            progress_queue.put_nowait({
                "type": "progress_update", 
                "current_doc_index": i + 1, 
                "total_docs": total_docs,
                "file": file_path_for_log, 
                "status": "summarization_dispatch_started" # More specific status
            })
        
        summary_data = await dispatch_summarize_document(doc, model=model, custom_prompt=custom_prompt, progress_queue=progress_queue)
        if summary_data: # Ensure summary_data is not None (e.g. if dispatch_summarize_document returns None for unsupported types)
            docs_summaries.append(summary_data)

        if progress_queue:
            progress_queue.put_nowait({
                "type": "progress_update", 
                "current_doc_index": i + 1,
                "total_docs": total_docs,
                "file": file_path_for_log, 
                "status": "summarization_dispatch_completed" # More specific status
            })
    return docs_summaries


async def remove_deleted_files():
    file_paths = db.get_all_files()
    deleted_file_paths = [file_path for file_path in file_paths if not os.path.exists(file_path)]
    db.delete_records(deleted_file_paths)


def load_documents(path: str, recursive: bool, required_exts: list):
    logger.info(f"Attempting to load documents from {path} with extensions: {required_exts}")
    splitter = TokenTextSplitter(chunk_size=6144)
    documents = []

    # Separate PDF extensions
    pdf_exts = [ext for ext in required_exts if ext.lower() == '.pdf']
    other_exts = [ext for ext in required_exts if ext.lower() != '.pdf']

    if pdf_exts:
        from llama_index.core.readers import PDFReader  # Assuming this is the correct import
        
        file_pattern = "**/*.pdf" if recursive else "*.pdf"
        # Using Path.glob to find PDF files
        pdf_files = list(Path(path).glob(file_pattern))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process with PDFReader.")

        for pdf_file_path in pdf_files:
            file_path_str = str(pdf_file_path)
            logger.info(f"Processing PDF file: {file_path_str} with dedicated PDFReader")
            try:
                pdf_reader = PDFReader()
                # load_data expects a Path object for the file argument
                pdf_docs_list = pdf_reader.load_data(file=pdf_file_path)
                
                if not pdf_docs_list:
                    logger.warning(f"PDFReader returned no documents for {file_path_str}")
                    continue

                # Process each document loaded from the PDF
                for loaded_doc in pdf_docs_list:
                    if not loaded_doc.text or not loaded_doc.text.strip():
                        logger.warning(f"PDF page/document in {file_path_str} has no text content.")
                        # Add document with empty text if metadata is needed, or skip
                        documents.append(Document(text="", metadata={"file_path": file_path_str, "source_empty": True}))
                        continue
                    
                    try:
                        # Assuming PDFReader provides metadata similar to SimpleDirectoryReader,
                        # or we might need to construct it.
                        # For now, let's ensure file_path is in metadata.
                        current_metadata = loaded_doc.metadata or {}
                        if 'file_path' not in current_metadata:
                           current_metadata['file_path'] = file_path_str
                        
                        # Split text if necessary (PDFReader might return one doc per page or whole doc)
                        split_texts = splitter.split_text(loaded_doc.text)
                        for text_chunk in split_texts:
                            documents.append(Document(text=text_chunk, metadata=current_metadata))
                        logger.debug(f"Successfully processed and split PDF: {file_path_str}")
                    except Exception as e_split:
                        logger.error(f"Error splitting text for PDF file {file_path_str}: {e_split}")

            except Exception as e:
                logger.error(f"Error reading PDF file {file_path_str} with dedicated PDFReader: {e}")

    if other_exts:
        logger.info(f"Processing other file types {other_exts} with SimpleDirectoryReader.")
        reader = SimpleDirectoryReader(
            input_dir=path,
            recursive=recursive,
            required_exts=other_exts,
            errors='warn'  # Changed from 'ignore' to 'warn'
        )
        try:
            for docs_chunk in reader.iter_data():
                if not docs_chunk or not docs_chunk[0].metadata or 'file_path' not in docs_chunk[0].metadata:
                    logger.warning(f"Skipping a document chunk due to missing data or metadata in {path}.")
                    continue

                file_path_for_log = docs_chunk[0].metadata['file_path']
                logger.info(f"Processing file: {file_path_for_log} with SimpleDirectoryReader")

                # By default, llama index split files into multiple "documents" (docs_chunk)
                if len(docs_chunk) > 1:
                    try:
                        # Join all document contexts, then truncate by token count
                        full_text = "\n".join([d.text for d in docs_chunk if d.text])
                        if not full_text.strip():
                            logger.warning(f"File {file_path_for_log} has no text content after joining chunks.")
                            documents.append(Document(text="", metadata=docs_chunk[0].metadata))
                            continue
                        
                        text_chunks = splitter.split_text(full_text)
                        for chunk in text_chunks:
                            documents.append(Document(text=chunk, metadata=docs_chunk[0].metadata))
                    except Exception as e:
                        logger.error(f"Error splitting text for file {file_path_for_log}: {e}")
                elif docs_chunk: # Single document in the chunk
                    if not docs_chunk[0].text or not docs_chunk[0].text.strip():
                        logger.warning(f"File {file_path_for_log} has no text content.")
                        documents.append(Document(text="", metadata=docs_chunk[0].metadata))
                        continue
                    try:
                        text_chunks = splitter.split_text(docs_chunk[0].text)
                        for chunk in text_chunks:
                             documents.append(Document(text=chunk, metadata=docs_chunk[0].metadata))
                    except Exception as e:
                        logger.error(f"Error splitting text for single-doc file {file_path_for_log}: {e}")
                else:
                    logger.warning(f"Empty document chunk encountered for path {path} with extensions {other_exts}")
        except Exception as e:
            logger.error(f"Error during SimpleDirectoryReader processing for path {path} with extensions {other_exts}: {e}")
    
    if not pdf_exts and not other_exts and required_exts:
        logger.warning(f"Required extensions {required_exts} were specified, but resulted in no files to process.")
    elif not documents:
        logger.warning(f"No documents were loaded from {path} with extensions {required_exts}. Check path and file types.")

    return documents


async def get_dir_summaries(path: str, recursive: bool, required_exts: list, model: Model, custom_prompt: str | None = None, progress_queue: Queue | None = None):
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "message": "Loading documents..."})
    
    # load_documents is synchronous, so it will block here.
    # For true async progress during loading, load_documents would need to be async and yield progress.
    doc_dicts = load_documents(path, recursive, required_exts) 
    
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "total_docs_loaded": len(doc_dicts), "message": "Documents loaded."})
        if not doc_dicts:
             progress_queue.put_nowait({"type": "status", "message": "No documents found to summarize."})
             # Early exit if no documents
             return []


    await remove_deleted_files() # This is quick, no progress needed for now
    
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "total_docs_for_summary": len(doc_dicts), "message": "Starting summarization process..."})

    files_summaries = await get_summaries(doc_dicts, model=model, custom_prompt=custom_prompt, progress_queue=progress_queue)

    # Convert path to relative path
    for summary_item in files_summaries: # Renamed 'summary' to 'summary_item'
        if 'file_path' in summary_item: # Ensure 'file_path' exists
             summary_item["file_path"] = os.path.relpath(summary_item["file_path"], path)
    
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "message": "Summarization complete for all documents."})
    return files_summaries


async def run(
    directory_path: str, 
    recursive: bool, 
    required_exts: list, 
    llm_provider: str = "openai", 
    ollama_api_base_url: str | None = None, 
    custom_summarization_prompt: str | None = None,
    progress_queue: Queue | None = None # Added from server.py
):
    try:
        logger.info("Starting FileWizardAI run...")
        if progress_queue:
            progress_queue.put_nowait({"type": "status", "message": "Initializing FileWizardAI..."})

        model = Model(llm_provider=llm_provider, ollama_api_base_url=ollama_api_base_url)
        
        if progress_queue:
            progress_queue.put_nowait({"type": "status", "message": "Starting directory analysis and document summarization..."})
        
        summaries = await get_dir_summaries(
            directory_path, recursive, required_exts, model=model, 
            custom_prompt=custom_summarization_prompt, progress_queue=progress_queue
        )
        
        if not summaries: # If no summaries (e.g., no documents found or all failed), can end early.
            logger.info("No summaries generated, concluding run.")
            if progress_queue:
                progress_queue.put_nowait({"type": "status", "message": "No summaries generated. File tree generation skipped."})
            return [] # Return empty list as no files to process for tree

        if progress_queue:
            progress_queue.put_nowait({"type": "status", "message": "Generating file tree based on summaries..."})
        
        files = await model.create_file_tree_api(summaries, progress_queue=progress_queue)
        
        if progress_queue:
            progress_queue.put_nowait({"type": "status", "message": "File tree generation complete."})

        # The original code returned `files`, which is a list of {"src_path": ..., "dst_path": ...}
        # The tree structure was for local processing not the return value.
        # Keeping it consistent:
        
        logger.info("FileWizardAI run completed successfully.")
        if progress_queue: # Final status before completion if needed, though server.py sends task_completed
            progress_queue.put_nowait({"type": "status", "message": "Process completed successfully."})
        return files
    
    except asyncio.CancelledError:
        logger.info("FileWizardAI run was cancelled.")
        if progress_queue:
            # This message might not reach if the task is cancelled very abruptly.
            # server.py's event_generator should also handle this.
            try:
                progress_queue.put_nowait({"type": "cancelled", "message": "Task cancelled during core run execution."})
            except Exception as e_q:
                logger.error(f"Failed to put cancellation message in queue: {e_q}")
        raise # Re-raise to be handled by the server's event_generator
    except Exception as e:
        logger.error(f"Error during FileWizardAI run: {e}", exc_info=True)
        if progress_queue:
            try:
                progress_queue.put_nowait({"type": "error", "message": f"An unexpected error occurred in core run: {str(e)}"})
            except Exception as e_q:
                logger.error(f"Failed to put error message in queue: {e_q}")
        raise # Re-raise to ensure server knows task failed


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
    progress_queue: Queue | None = None # Added for consistency, though search is not SSE yet
):
    # Note: search_files is not currently an SSE endpoint, so progress_queue might not be used by the caller.
    # However, adding it for internal consistency if we decide to make search SSE later.
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "message": "Search process started..."})

    model = Model(llm_provider=llm_provider, ollama_api_base_url=ollama_api_base_url)
    
    # Pass progress_queue to get_dir_summaries if search were to be SSE.
    summaries = await get_dir_summaries(
        root_path, recursive, required_exts, model=model, 
        custom_prompt=custom_summarization_prompt, progress_queue=progress_queue 
    )
    
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "message": "Searching through summaries..."})
        
    files = await model.search_files_api(summaries, search_query, progress_queue=progress_queue) 
    
    if progress_queue:
        progress_queue.put_nowait({"type": "status", "message": "Search process completed."})
        
    return files


def get_file_hash(file_path):
    hash_func = hashlib.new('sha256')
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()
