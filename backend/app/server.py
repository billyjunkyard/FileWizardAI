from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .run import run, update_file, search_files
from .embedding_service import embedding_service
from .vector_store_service import vector_store_service
from .settings import Model as SettingsModel # Renamed to avoid conflict with 'Model' from Pydantic/FastAPI
import os
import subprocess
import platform
from fastapi.responses import FileResponse

import asyncio
import json
from sse_starlette.sse import EventSourceResponse
from typing import Dict, List # Added List
from asyncio import Task, Queue
import uuid
import logging
# import json # json is already imported by run.py, but good to have explicitly if needed directly in server

logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

active_tasks: Dict[str, Task] = {}

async def run_and_report_progress(
    progress_queue: Queue, 
    root_path: str, 
    recursive: bool, 
    required_exts: list, 
    llm_provider: str, 
    ollama_api_base_url: str | None, 
    custom_summarization_prompt: str | None
):
    """
    Wrapper to call the main 'run' function and pass its progress to the queue.
    The 'run' function in run.py will be modified to accept 'progress_queue'
    and put updates into it.
    """
    try:
        # The actual 'run' function from run.py needs to be modified
        # to accept progress_queue and use it.
        result = await run(
            directory_path=root_path,
            recursive=recursive,
            required_exts=required_exts,
            llm_provider=llm_provider,
            ollama_api_base_url=ollama_api_base_url,
            custom_summarization_prompt=custom_summarization_prompt,
            progress_queue=progress_queue # This new argument will be added to run.py's run function
        )
        return result
    except asyncio.CancelledError:
        logger.info(f"Task for {root_path} was cancelled by run_and_report_progress.")
        await progress_queue.put({"type": "cancelled", "message": "Task explicitly cancelled in wrapper."})
        raise # Re-raise to be handled by the event_generator
    except Exception as e:
        logger.error(f"Exception in run_and_report_progress for {root_path}: {e}", exc_info=True)
        await progress_queue.put({"type": "error", "message": f"Error during processing: {str(e)}"})
        raise # Re-raise to be handled by the event_generator


@app.get('/')
def get_angular_app():
    return FileResponse("app/static/index.html")


@app.get("/get_files")
async def get_files_sse(
    root_path: str, 
    recursive: bool, 
    required_exts: str, 
    llm_provider: str = Query("openai", description="The LLM provider to use, e.g., 'openai' or 'ollama'"), 
    ollama_api_base_url: str = Query(None, description="Ollama API base URL, e.g., http://localhost:11434/v1"), 
    custom_summarization_prompt: str = Query(None, description="Custom prompt for document summarization")
):
    if not os.path.exists(root_path):
        # EventSourceResponse doesn't handle HTTPExceptions well directly,
        # so we might need a different way if we want to return an error for this initial check.
        # For now, let it proceed and the error will be caught by the task.
        # A better approach might be to validate then return EventSourceResponse.
        # However, the prompt asks for the endpoint to return EventSourceResponse.
        # This will be handled by the task error reporting for now.
        pass

    task_id = str(uuid.uuid4())
    progress_queue = asyncio.Queue()
    parsed_required_exts = required_exts.split(';')

    async def event_generator():
        yield {"event": "task_started", "data": json.dumps({"task_id": task_id})}
        
        processing_task = asyncio.create_task(
            run_and_report_progress(
                progress_queue,
                root_path,
                recursive,
                parsed_required_exts,
                llm_provider,
                ollama_api_base_url,
                custom_summarization_prompt
            )
        )
        active_tasks[task_id] = processing_task
        
        try:
            while True:
                # Wait for either progress update or task completion
                get_progress_task = asyncio.create_task(progress_queue.get())
                
                done, pending = await asyncio.wait(
                    [get_progress_task, processing_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                if get_progress_task in done:
                    progress_update = get_progress_task.result()
                    if progress_update.get("type") == "error": # Error sent from run_and_report_progress
                        yield {"event": "task_error", "data": json.dumps({"error": progress_update.get("message", "Unknown error"), "task_id": task_id})}
                        # If a critical error happened in the task, we might want to break or ensure processing_task is awaited.
                        # For now, we assume processing_task will also complete.
                        # To be safe, ensure processing_task is cancelled if not already done.
                        if not processing_task.done():
                             processing_task.cancel()
                        break 
                    elif progress_update.get("type") == "cancelled":
                        yield {"event": "task_cancelled", "data": json.dumps({"task_id": task_id, "message": progress_update.get("message")})}
                        if not processing_task.done():
                             processing_task.cancel()
                        break
                    else:
                        yield {"event": "progress", "data": json.dumps(progress_update)}

                if processing_task in done:
                    try:
                        result = await processing_task # Get result or raise exception if not already handled
                        yield {"event": "task_completed", "data": json.dumps({"root_path": root_path, "items": result})}
                    except asyncio.CancelledError:
                        logger.info(f"Task {task_id} was cancelled by event_generator.")
                        yield {"event": "task_cancelled", "data": json.dumps({"task_id": task_id, "message": "Task was cancelled."})}
                    except Exception as e:
                        logger.error(f"Task {task_id} failed with exception: {e}", exc_info=True)
                        yield {"event": "task_error", "data": json.dumps({"error": str(e), "task_id": task_id})}
                    finally:
                        break # Exit the while loop as the main task is finished
            
        except asyncio.CancelledError: # Catch cancellation of the event_generator itself (e.g. client disconnects)
            logger.info(f"Event generator for task {task_id} cancelled (client disconnected). Cancelling processing task.")
            if not processing_task.done():
                processing_task.cancel()
            # Yielding here might not reach client if they disconnected
            # yield {"event": "task_cancelled", "data": json.dumps({"task_id": task_id, "message": "Client disconnected."})}
            # Re-raise is not strictly necessary here as we are exiting.
        except Exception as e:
            # Catch any other unexpected errors in the generator
            logger.error(f"Error in event_generator for task {task_id}: {e}", exc_info=True)
            yield {"event": "task_error", "data": json.dumps({"error": f"Unexpected error in event stream: {str(e)}", "task_id": task_id})}
        finally:
            if task_id in active_tasks:
                del active_tasks[task_id]
            logger.info(f"Event generator for task {task_id} finished.")

    if not os.path.exists(root_path): # Check again, as it's an async generator
        # This is tricky with EventSourceResponse. The headers are already sent.
        # One way is to send an error event immediately.
        async def error_event_generator():
            yield {"event": "task_error", "data": json.dumps({"error": f"Path doesn't exist: {root_path}", "task_id": task_id})}
        return EventSourceResponse(error_event_generator())
        
    return EventSourceResponse(event_generator())


@app.post("/update_files")
async def update_files(request: Request):
    data = await request.json()
    root_path = data.get('root_path')
    items = data.get('items')
    for item in items:
        try:
            update_file(root_path, item)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error while moving file: {e}")
    return {"message": "Files moved successfully"}


@app.post("/open_file")
async def open_file(request: Request):
    data = await request.json()
    file_path = data.get('file_path')
    if not os.path.exists(file_path):
        return HTTPException(status_code=404, detail=f"File doesn't exist: {file_path}")
    current_os = platform.system()
    try:
        if current_os == "Windows":
            os.startfile(file_path)
        elif current_os == "Darwin":
            subprocess.run(["open", file_path])
        elif current_os == "Linux":
            subprocess.run(["xdg-open", file_path])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while opening file: {e}")
    return {"message": "Files opened successfully"}


@app.get("/search_files")
async def get_search_files(root_path: str, recursive: bool, required_exts: str, search_query: str, llm_provider: str = Query("openai", description="The LLM provider to use, e.g., 'openai' or 'ollama'"), ollama_api_base_url: str = Query(None, description="Ollama API base URL, e.g., http://localhost:11434/v1"), custom_summarization_prompt: str = Query(None, description="Custom prompt for document summarization")):
    if not os.path.exists(root_path):
        return HTTPException(status_code=404, detail=f"Path doesn't exist: {root_path}")
    required_exts = required_exts.split(';')
    # Note: Search files is not SSE for now, as per subtask focusing on /get_files
    files = await search_files(root_path, recursive, required_exts, search_query, llm_provider=llm_provider, ollama_api_base_url=ollama_api_base_url, custom_summarization_prompt=custom_summarization_prompt)
    return files

@app.delete("/cancel_task/{task_id}")
async def cancel_task_endpoint(task_id: str):
    if task_id in active_tasks:
        task = active_tasks[task_id]
        if not task.done():
            task.cancel()
            # Wait for the task to acknowledge cancellation, but with a timeout
            try:
                await asyncio.wait_for(task, timeout=5.0) 
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for task {task_id} to cancel.")
            except asyncio.CancelledError:
                 logger.info(f"Task {task_id} was successfully cancelled by endpoint.")
            # The finally block in event_generator should remove it from active_tasks
            # but as a safeguard or if cancellation is very quick:
            if task_id in active_tasks: # check again as it might have been removed by its own finally block
                del active_tasks[task_id]
            return {"message": "Task cancellation requested", "task_id": task_id}
        else:
            # Task already completed, remove if still present (should be handled by finally block)
            if task_id in active_tasks:
                del active_tasks[task_id]
            return {"message": "Task already completed", "task_id": task_id}
    else:
        return HTTPException(status_code=404, detail=f"Task not found: {task_id}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/semantic_search/")
async def semantic_search_endpoint(
    query_text: str = Query(..., description="The text to search for."),
    top_n: int = Query(5, description="Number of top similar chunks to retrieve.", ge=1, le=50),
    file_paths_json: str = Query(None, description="Optional JSON string array of specific file paths to search within. Example: [\"/path/to/doc1.txt\", \"/path/to/doc2.pdf\"]")
):
    logger.info(f"Received semantic search request: query_text='{query_text[:50]}...', top_n={top_n}, file_paths_json='{file_paths_json}'")

    parsed_file_paths = None
    if file_paths_json:
        try:
            parsed_file_paths = json.loads(file_paths_json)
            if not isinstance(parsed_file_paths, list) or not all(isinstance(fp, str) for fp in parsed_file_paths):
                logger.error(f"Invalid format for file_paths_json. Expected list of strings. Got: {parsed_file_paths}")
                raise HTTPException(status_code=400, detail="Invalid format for file_paths_json. Must be a JSON array of strings.")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse file_paths_json: {file_paths_json}", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid JSON format for file_paths_json.")

    if not embedding_service or not embedding_service.model:
        logger.error("Embedding service or model not available for semantic search.")
        raise HTTPException(status_code=503, detail="Embedding service not available. Please try again later.")
    if not vector_store_service or not vector_store_service.collection:
        logger.error("Vector store service or collection not available for semantic search.")
        raise HTTPException(status_code=503, detail="Vector store service not available. Please try again later.")

    if not query_text.strip():
        logger.warning("Semantic search query text is empty.")
        return [] 

    query_embedding = None
    try:
        query_embedding = embedding_service.get_embedding(query_text)
    except Exception as e_embed: 
        logger.error(f"Error generating query embedding for '{query_text[:50]}...': {e_embed}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

    if query_embedding is None:
        logger.warning(f"Could not generate embedding for query text: '{query_text[:50]}...'")
        raise HTTPException(status_code=400, detail="Could not generate embedding for the provided query text.")

    logger.info(f"Generated query embedding (first 5 dims): {query_embedding[:5]}")

    try:
        search_results = vector_store_service.search_similar_chunks(
            query_embedding=query_embedding, top_n=top_n, file_paths=parsed_file_paths
        )
        logger.info(f"Semantic search found {len(search_results)} results for query '{query_text[:50]}...'.")
        return search_results
    except Exception as e_search:
        logger.error(f"Error during vector store search for query '{query_text[:50]}...': {e_search}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error performing search in vector store.")

@app.get("/answer_question/")
async def answer_question_endpoint(
    query_text: str = Query(..., description="The user's question."),
    top_n_chunks: int = Query(3, description="Number of relevant chunks to retrieve for context.", ge=1, le=10),
    file_paths_json: str = Query(None, description="Optional JSON string array of specific file paths to search within.")
):
    logger.info(f"Received Q&A request: query_text='{query_text[:50]}...', top_n_chunks={top_n_chunks}, file_paths_json='{file_paths_json}'")

    parsed_file_paths = None
    if file_paths_json:
        try:
            parsed_file_paths = json.loads(file_paths_json)
            if not isinstance(parsed_file_paths, list) or not all(isinstance(fp, str) for fp in parsed_file_paths):
                logger.error(f"Invalid format for file_paths_json in Q&A. Expected list of strings. Got: {parsed_file_paths}")
                raise HTTPException(status_code=400, detail="Invalid format for file_paths_json. Must be a JSON array of strings.")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse file_paths_json in Q&A: {file_paths_json}", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid JSON format for file_paths_json.")

    # Service availability checks
    if not embedding_service or not embedding_service.model:
        logger.error("Embedding service or model not available for Q&A.")
        raise HTTPException(status_code=503, detail="Embedding service not available.")
    if not vector_store_service or not vector_store_service.collection:
        logger.error("Vector store service or collection not available for Q&A.")
        raise HTTPException(status_code=503, detail="Vector store service not available.")
    
    # Instantiate Model from settings for LLM call - assuming default provider for now
    # In a more complex app, llm_provider might come from request or global config.
    try:
        model_instance = SettingsModel() # Uses defaults from .env or hardcoded in Settings
        if not model_instance.async_text_clients: # Check if clients were initialized
             logger.error("LLM text clients are not configured in Model settings for Q&A.")
             raise HTTPException(status_code=503, detail="LLM service not configured.")
    except Exception as e_model_init:
        logger.error(f"Failed to initialize Model for Q&A: {e_model_init}", exc_info=True)
        raise HTTPException(status_code=503, detail="LLM service initialization failed.")


    if not query_text.strip():
        logger.warning("Q&A query text is empty.")
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    # Generate query embedding
    query_embedding = None
    try:
        query_embedding = embedding_service.get_embedding(query_text)
    except Exception as e_embed:
        logger.error(f"Error generating query embedding for Q&A '{query_text[:50]}...': {e_embed}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate query embedding for Q&A.")
    
    if query_embedding is None:
        logger.warning(f"Could not generate embedding for Q&A query: '{query_text[:50]}...'")
        raise HTTPException(status_code=400, detail="Could not generate embedding for the Q&A query text.")

    # Search for relevant chunks
    relevant_chunks = []
    try:
        relevant_chunks = vector_store_service.search_similar_chunks(
            query_embedding=query_embedding,
            top_n=top_n_chunks,
            file_paths=parsed_file_paths
        )
    except Exception as e_search:
        logger.error(f"Error searching for relevant chunks for Q&A query '{query_text[:50]}...': {e_search}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error searching for relevant document chunks.")

    if not relevant_chunks:
        logger.info(f"No relevant chunks found for Q&A query: '{query_text[:50]}...'")
        return {"answer": "Could not find relevant documents to answer the question.", "source_chunks": []}

    # Construct context string
    context_string = "\n\n---\n\n".join([chunk['chunk_text'] for chunk in relevant_chunks if chunk.get('chunk_text')])
    logger.info(f"Constructed context string of length {len(context_string)} from {len(relevant_chunks)} chunks for Q&A query '{query_text[:50]}...'.")
    if not context_string.strip():
        logger.warning(f"Context string is empty after processing relevant chunks for Q&A query: '{query_text[:50]}...'")
        return {"answer": "Relevant document chunks found, but they contain no text to form an answer.", "source_chunks": relevant_chunks}

    # Call LLM to generate answer
    answer = None
    try:
        # Progress queue for LLM call is not directly used by this synchronous endpoint response for now.
        # If this were an SSE endpoint, it would be passed.
        answer = await model_instance.generate_answer_from_context(
            question=query_text,
            context=context_string,
            progress_queue=None # No SSE progress for this specific Q&A response
        )
    except Exception as e_llm:
        logger.error(f"Error generating answer from LLM for Q&A query '{query_text[:50]}...': {e_llm}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating answer from language model.")

    if answer is None:
        logger.warning(f"LLM did not return an answer for Q&A query: '{query_text[:50]}...'")
        # This case might be handled by the prompt ("I cannot answer..."), but if LLM returns nothing:
        answer = "The language model did not provide an answer based on the context."

    return {"answer": answer, "source_chunks": relevant_chunks}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
