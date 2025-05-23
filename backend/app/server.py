from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .run import run, update_file, search_files # 'run' will be modified to accept progress_queue
import os
import subprocess
import platform
from fastapi.responses import FileResponse

import asyncio
import json
from sse_starlette.sse import EventSourceResponse
from typing import Dict
from asyncio import Task, Queue
import uuid
import logging

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
