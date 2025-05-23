import time

from pydantic_settings import BaseSettings, SettingsConfigDict
from openai import AsyncOpenAI
import base64
import logging
import json
import sys
import requests
import logging

from asyncio import Queue # Added for type hinting
import logging # Ensure logging is imported if not already

# Use a logger specific to this module for better log filtering if needed
logger = logging.getLogger(__name__) 


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env')
    TEXT_API_END_POINT: str
    TEXT_MODEL_NAME: str
    TEXT_API_KEYS: list[str]
    IMAGE_API_END_POINT: str
    IMAGE_MODEL_NAME: str
    IMAGE_API_KEYS: list[str]
    LLM_PROVIDER: str = "openai"
    OLLAMA_API_BASE_URL: str = "http://localhost:11434/v1"


class Model:
    settings = Settings()

    TEXT_MODEL_NAME = settings.TEXT_MODEL_NAME
    IMAGE_MODEL_NAME = settings.IMAGE_MODEL_NAME
    IMAGE_API_KEYS = settings.IMAGE_API_KEYS # This should be settings.IMAGE_API_KEYS
    MAX_TOKEN_SIZE = 4000 # Increase or decrease based on the model context window size
    cnt_txt = 0
    cnt_img = 0

    def __init__(self, llm_provider: str = None, ollama_api_base_url: str = None):
        self.llm_provider = llm_provider if llm_provider else self.settings.LLM_PROVIDER
        self.ollama_api_base_url = ollama_api_base_url if ollama_api_base_url else self.settings.OLLAMA_API_BASE_URL

        if self.llm_provider == "ollama":
            self.TEXT_API_END_POINT = self.ollama_api_base_url
            self.async_text_clients = [AsyncOpenAI(base_url=self.TEXT_API_END_POINT, api_key="ollama")]
            # Assuming Ollama doesn't support image summarization with the current approach or requires a different setup
            logger.warning("Image summarization may not be fully supported with Ollama provider.")
            self.async_image_clients = [] # Placeholder, adjust if Ollama has image capabilities
            self.text_keys_count = 1 # Single client for Ollama
            self.image_keys_count = 0
        else: # openai or other providers
            self.TEXT_API_END_POINT = self.settings.TEXT_API_END_POINT
            self.TEXT_API_KEYS = self.settings.TEXT_API_KEYS
            self.async_text_clients = [AsyncOpenAI(base_url=self.TEXT_API_END_POINT, api_key=api_key)
                                       for api_key in self.TEXT_API_KEYS]
            self.IMAGE_API_END_POINT = self.settings.IMAGE_API_END_POINT
            # Corrected: Use settings.IMAGE_API_KEYS
            self.IMAGE_API_KEYS = self.settings.IMAGE_API_KEYS 
            self.async_image_clients = [AsyncOpenAI(base_url=self.IMAGE_API_END_POINT, api_key=api_key)
                                        for api_key in self.IMAGE_API_KEYS]
            self.text_keys_count = len(self.TEXT_API_KEYS)
            self.image_keys_count = len(self.IMAGE_API_KEYS)


    async def summarize_image_api(self, image_path, progress_queue: Queue | None = None):
        prompt = """
        Describe this image in the most concise way possible, capturing only the essential elements and details. 
        Aim for a very brief yet accurate summary.
        """
        attempt = 0
        summary = ""
        
        if progress_queue:
            progress_queue.put_nowait({"type": "llm_api_start", "file": image_path, "api_call": "summarize_image_api"})

        if self.llm_provider == "ollama" or not self.async_image_clients:
            logger.warning(f"Image summarization skipped for {image_path} as it's not supported by {self.llm_provider} or client not available.")
            if progress_queue:
                progress_queue.put_nowait({"type": "llm_api_skipped", "file": image_path, "api_call": "summarize_image_api", "reason": "Provider or client not available"})
            return "Image summarization not available for current provider."

        if hasattr(self, 'IMAGE_API_END_POINT') and "huggingface.co" in self.IMAGE_API_END_POINT.lower():
            endpoint_url = self.IMAGE_API_END_POINT.replace("v1", "models") + "/" + self.IMAGE_MODEL_NAME
            while attempt < 5:
                try:
                    if not self.IMAGE_API_KEYS or self.image_keys_count == 0: # Check if keys are available
                        raise ValueError("IMAGE_API_KEYS are not configured or empty for HuggingFace.")
                    headers = {"Authorization": f"Bearer {self.IMAGE_API_KEYS[self.cnt_img % self.image_keys_count]}"}
                    with open(image_path, "rb") as f:
                        data = f.read()
                    response = requests.post(endpoint_url, headers=headers, data=data)
                    response.raise_for_status() 
                    summary = response.json()[0]["generated_text"]
                    break
                except Exception as e:
                    logger.error(f"Error in summarize_image_api (HuggingFace) for {image_path}, attempt {attempt+1}: {e}")
                    attempt += 1
                    self.cnt_img += 1
                    if attempt >= 5 and progress_queue:
                        progress_queue.put_nowait({"type": "llm_api_failed_final", "file": image_path, "api_call": "summarize_image_api", "error": str(e)})
        else:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            while attempt < 5:
                try:
                    if not self.async_image_clients or self.image_keys_count == 0:
                         raise ValueError("AsyncOpenAI image clients are not configured or empty.")
                    chat_completion = await self.async_image_clients[
                        self.cnt_img % self.image_keys_count].chat.completions.create( 
                        model=Model.IMAGE_MODEL_NAME,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    },
                                ],
                            }
                        ],
                        timeout=30.0, # Added timeout
                        temperature=0,
                    )
                    summary = chat_completion.choices[0].message.content
                    break
                except Exception as e:
                    logger.error(f"Error in summarize_image_api (OpenAI-like) for {image_path}, attempt {attempt+1}: {e}")
                    attempt += 1
                    self.cnt_img += 1
                    if attempt >= 5 and progress_queue:
                         progress_queue.put_nowait({"type": "llm_api_failed_final", "file": image_path, "api_call": "summarize_image_api", "error": str(e)})
        
        if progress_queue:
            progress_queue.put_nowait({"type": "llm_api_end", "file": image_path, "api_call": "summarize_image_api", "success": bool(summary)})
        return summary

    async def summarize_document_api(self, doc_text, custom_prompt: str = None, progress_queue: Queue | None = None, file_path_for_logging: str = "Unknown file"):
        default_prompt = """
        You will be provided with the contents of a file. Provide a summary of the contents. 
        The purpose of the summary is to organize files based on their content. 
        To this end provide a concise but informative summary. Make the summary as specific to the file as possible.
        It is very important that you only provide the final output without any additional comments or remarks.
        """.strip()

        system_prompt = default_prompt
        if custom_prompt and custom_prompt.strip():
            system_prompt = custom_prompt.strip()
            logger.info(f"Using custom summarization prompt for {file_path_for_logging}: {system_prompt[:100]}...")
        else:
            logger.info(f"Using default summarization prompt for {file_path_for_logging}.")

        if progress_queue:
            progress_queue.put_nowait({"type": "llm_api_start", "file": file_path_for_logging, "api_call": "summarize_document_api"})

        attempt = 0
        summary = ""
        while attempt < 5:
            try:
                if not self.async_text_clients or self.text_keys_count == 0:
                     raise ValueError("AsyncOpenAI text clients are not configured or empty.")
                chat_completion = await self.async_text_clients[
                    self.cnt_txt % self.text_keys_count].chat.completions.create(
                    model=self.TEXT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": doc_text},
                    ],
                    stream=False,
                    temperature=0,
                    timeout=60.0, # Added timeout
                )
                summary = chat_completion.choices[0].message.content
                break
            except Exception as e:
                logger.error(f"Error in summarize_document_api for {file_path_for_logging}, attempt {attempt+1}: {e}")
                attempt += 1
                self.cnt_txt += 1
                if attempt >= 5 and progress_queue: 
                    progress_queue.put_nowait({"type": "llm_api_failed_final", "file": file_path_for_logging, "api_call": "summarize_document_api", "error": str(e)})
        
        if progress_queue:
            progress_queue.put_nowait({"type": "llm_api_end", "file": file_path_for_logging, "api_call": "summarize_document_api", "success": bool(summary)})
        return summary

    async def create_file_tree_api(self, summaries: list, progress_queue: Queue | None = None):
        tmp: list = []
        file_tree: list = []
        
        # Estimate total chunks for progress
        total_chunks_to_process = 0
        current_chunk_size = 0
        for s in summaries:
            s_size = sys.getsizeof(json.dumps(s)) / 4 # Approximate token size
            if current_chunk_size + s_size >= self.MAX_TOKEN_SIZE and current_chunk_size > 0:
                total_chunks_to_process += 1
                current_chunk_size = 0
            current_chunk_size += s_size
        if current_chunk_size > 0:
            total_chunks_to_process += 1
        
        if progress_queue:
            progress_queue.put_nowait({"type": "status", "message": f"Starting file tree generation. Estimated {total_chunks_to_process} chunk(s).", "total_chunks": total_chunks_to_process, "current_chunk_num": 0 })

        processed_chunks_count = 0
        for summary_item in summaries: 
            item_size = sys.getsizeof(json.dumps(summary_item)) / 4
            if (sys.getsizeof(json.dumps(tmp)) / 4) + item_size >= self.MAX_TOKEN_SIZE and tmp:
                processed_chunks_count += 1
                if progress_queue:
                    progress_queue.put_nowait({"type": "chunk_processing_start", "api_call": "create_file_tree_api_chunk", "chunk_num": processed_chunks_count, "total_chunks": total_chunks_to_process, "num_summaries_in_chunk": len(tmp)})
                chunk_result = await self.create_file_tree_api_chunk(tmp, progress_queue=progress_queue)
                file_tree.extend(chunk_result) 
                if progress_queue:
                     progress_queue.put_nowait({"type": "chunk_processing_end", "api_call": "create_file_tree_api_chunk", "chunk_num": processed_chunks_count, "results_count": len(chunk_result)})
                tmp = []
            tmp.append(summary_item)
        
        if tmp: 
            processed_chunks_count += 1
            if progress_queue:
                progress_queue.put_nowait({"type": "chunk_processing_start", "api_call": "create_file_tree_api_chunk", "chunk_num": processed_chunks_count, "total_chunks": total_chunks_to_process, "num_summaries_in_chunk": len(tmp)})
            chunk_result = await self.create_file_tree_api_chunk(tmp, progress_queue=progress_queue)
            file_tree.extend(chunk_result)
            if progress_queue:
                progress_queue.put_nowait({"type": "chunk_processing_end", "api_call": "create_file_tree_api_chunk", "chunk_num": processed_chunks_count, "results_count": len(chunk_result)})
        
        if progress_queue:
            progress_queue.put_nowait({"type": "status", "message": "File tree generation completed."})
        return file_tree

    async def create_file_tree_api_chunk(self, summaries: list, progress_queue: Queue | None = None):
        file_prompt = """
        You will be provided with list of source files and a summary of their contents.
        For each file,propose a new path and filename, using a directory structure that optimally organizes the files using known conventions and best practices.
        Follow good naming conventions. Here are a few guidelines
        - Think about your files : What related files are you working with?
        - Identify metadata (for example, date, sample, experiment) : What information is needed to easily locate a specific file?
        - Abbreviate or encode metadata
        - Use versioning : Are you maintaining different versions of the same file?
        - Think about how you will search for your files : What comes first?
        - Deliberately separate metadata elements : Avoid spaces or special characters in your file names
        If the file is already named well or matches a known convention, set the destination path to the same as the source path.

        Your response must be a JSON object with the following schema, dont add any extra text except the json:
        ```json
        {
            "files": [
                {
                    "src_path": "original file path",
                    "dst_path": "new file path under proposed directory structure with proposed file name"
                }
            ]
        }
        ```
        """.strip()
        
        if progress_queue:
            progress_queue.put_nowait({"type": "llm_api_start", "api_call": "create_file_tree_api_chunk", "num_summaries_in_chunk": len(summaries)})

        attempt = 0
        file_tree_chunk_result = [] 
        while attempt < 10: 
            try:
                if not self.async_text_clients or self.text_keys_count == 0:
                     raise ValueError("AsyncOpenAI text clients for file tree are not configured or empty.")
                chat_completion = await self.async_text_clients[
                    self.cnt_txt % self.text_keys_count].chat.completions.create(
                    messages=[
                        {"role": "system", "content": file_prompt},
                        {"role": "user", "content": json.dumps(summaries)},
                    ],
                    model=self.TEXT_MODEL_NAME,
                    stream=False,
                    temperature=0,
                    timeout=120.0, # Longer timeout for potentially larger chunks
                )
                result_text = chat_completion.choices[0].message.content
                result_text = result_text.replace("```json", "").replace("```", "").strip()
                file_tree_chunk_result = json.loads(result_text)["files"]
                break
            except Exception as e:
                logger.error(f"Error in create_file_tree_api_chunk, attempt {attempt+1}: {e}")
                attempt += 1
                self.cnt_txt += 1
                if attempt >= 10 and progress_queue:
                     progress_queue.put_nowait({"type": "llm_api_failed_final", "api_call": "create_file_tree_api_chunk", "error": str(e)})
                await asyncio.sleep(2) # Use asyncio.sleep
        
        if progress_queue:
            progress_queue.put_nowait({"type": "llm_api_end", "api_call": "create_file_tree_api_chunk", "success": bool(file_tree_chunk_result)})
        return file_tree_chunk_result

    async def search_files_api(self, summaries: list, search_query: str, progress_queue: Queue | None = None):
        tmp: list = []
        files: list = []
        
        total_search_chunks = 0
        current_search_chunk_size = 0
        for s in summaries:
            s_size = sys.getsizeof(json.dumps(s)) / 4
            if current_search_chunk_size + s_size >= self.MAX_TOKEN_SIZE and current_search_chunk_size > 0:
                total_search_chunks += 1
                current_search_chunk_size = 0
            current_search_chunk_size += s_size
        if current_search_chunk_size > 0:
            total_search_chunks += 1

        if progress_queue: # Though search isn't SSE yet, for internal consistency
            progress_queue.put_nowait({"type": "status", "message": f"Starting file search. Estimated {total_search_chunks} chunk(s).", "total_chunks": total_search_chunks, "current_chunk_num":0})

        processed_search_chunks_count = 0
        for summary_item in summaries: 
            item_size = sys.getsizeof(json.dumps(summary_item)) / 4
            if (sys.getsizeof(json.dumps(tmp))/4) + item_size >= self.MAX_TOKEN_SIZE and tmp:
                processed_search_chunks_count +=1
                if progress_queue:
                     progress_queue.put_nowait({"type": "chunk_processing_start", "api_call": "search_files_api_chunk", "chunk_num": processed_search_chunks_count, "total_chunks": total_search_chunks, "num_summaries_in_chunk": len(tmp)})
                chunk_result = await self.search_files_api_chunk(tmp, search_query, progress_queue=progress_queue)
                files.extend(chunk_result)
                if progress_queue:
                    progress_queue.put_nowait({"type": "chunk_processing_end", "api_call": "search_files_api_chunk", "chunk_num": processed_search_chunks_count, "results_count": len(chunk_result)})
                tmp = []
            tmp.append(summary_item)
        
        if tmp: 
            processed_search_chunks_count +=1
            if progress_queue:
                 progress_queue.put_nowait({"type": "chunk_processing_start", "api_call": "search_files_api_chunk", "chunk_num": processed_search_chunks_count, "total_chunks": total_search_chunks, "num_summaries_in_chunk": len(tmp)})
            chunk_result = await self.search_files_api_chunk(tmp, search_query, progress_queue=progress_queue)
            files.extend(chunk_result)
            if progress_queue:
                progress_queue.put_nowait({"type": "chunk_processing_end", "api_call": "search_files_api_chunk", "chunk_num": processed_search_chunks_count, "results_count": len(chunk_result)})
        
        if progress_queue:
            progress_queue.put_nowait({"type": "status", "message": "File search completed."})
        return files

    async def search_files_api_chunk(self, summaries: list, search_query: str, progress_queue: Queue | None = None):
        file_prompt = """
        You will be provided with list of source files and a summary of their contents:
        return the files that matches or have a similar content to this search query: """ + search_query + """

        Your response must be a JSON object with the following schema, dont add any extra text except the json:
        ```json
        {
        "files": [
                {
                    "file": "File that matches or have a similar content to the search query"
                }
            ]
        }
        """.strip()

        if progress_queue: # Though search isn't SSE, good for consistency
            progress_queue.put_nowait({"type": "llm_api_start", "api_call": "search_files_api_chunk", "num_summaries_in_chunk": len(summaries)})
        
        attempt = 0
        search_results_chunk = []
        # Using 5 attempts as in summarize_document_api for consistency
        while attempt < 5: 
            try:
                if not self.async_text_clients or self.text_keys_count == 0:
                     raise ValueError("AsyncOpenAI text clients for search are not configured or empty.")
                chat_completion = await self.async_text_clients[
                    self.cnt_txt % self.text_keys_count].chat.completions.create(
                    messages=[
                        {"role": "system", "content": file_prompt},
                        {"role": "user", "content": json.dumps(summaries)},
                    ],
                    model=self.TEXT_MODEL_NAME,
                    stream=False,
                    timeout=60.0, # Added timeout
                )
                result_text = chat_completion.choices[0].message.content
                search_results_chunk = json.loads(result_text)["files"]
                break
            except Exception as e:
                logger.error(f"Error in search_files_api_chunk, attempt {attempt+1}: {e}")
                attempt += 1
                self.cnt_txt += 1
                if attempt >= 5 and progress_queue:
                    progress_queue.put_nowait({"type": "llm_api_failed_final", "api_call": "search_files_api_chunk", "error": str(e)})
                await asyncio.sleep(2) # Use asyncio.sleep
        
        if progress_queue:
            progress_queue.put_nowait({"type": "llm_api_end", "api_call": "search_files_api_chunk", "success": bool(search_results_chunk)})
        return search_results_chunk


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;5;15m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
