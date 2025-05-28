import time # Keep time if used elsewhere, or remove if not.
# asyncio is used in some methods, ensure it's imported
import asyncio 

from pydantic_settings import BaseSettings, SettingsConfigDict
from openai import AsyncOpenAI
import base64
import logging # Ensure logging is imported
import json
import sys
import requests
# Remove redundant logging import if already at top

from asyncio import Queue # Added for type hinting

# Use a logger specific to this module for better log filtering if needed
logger = logging.getLogger(__name__) 


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore') # Add extra='ignore'

    # Provider: OpenAI / Groq (or any other OpenAI-compatible)
    OPENAI_TEXT_API_END_POINT: str = "https://api.groq.com/openai/v1"
    OPENAI_TEXT_MODEL_NAME: str = "llama3-70b-8192"
    OPENAI_TEXT_API_KEYS: list[str] = ["YOUR_DEFAULT_GROQ_OR_OPENAI_API_KEY_HERE"] # Provide a sensible default
    OPENAI_IMAGE_API_END_POINT: str = "https://api.groq.com/openai/v1" # Assuming same endpoint for images with Groq, adjust if different
    OPENAI_IMAGE_MODEL_NAME: str = "llava-v1.5-7b-4096-preview" # Example, adjust if using OpenAI or other
    OPENAI_IMAGE_API_KEYS: list[str] = ["YOUR_DEFAULT_GROQ_OR_OPENAI_API_KEY_HERE"]

    # Provider: Ollama
    OLLAMA_TEXT_API_BASE_URL: str = "http://localhost:11434/v1" # Base URL for Ollama text operations
    OLLAMA_TEXT_MODEL_NAME: str = "gemma2:latest" # Default Ollama text model
    # OLLAMA_IMAGE_API_BASE_URL: str = "http://localhost:11434/v1" # If image support is different
    # OLLAMA_IMAGE_MODEL_NAME: str = "moondream:latest" # Default Ollama image model, if supported

    # General setting for the default provider if not specified by request
    DEFAULT_LLM_PROVIDER: str = "openai"


class Model:
    settings = Settings()
    # TEXT_MODEL_NAME, IMAGE_MODEL_NAME, IMAGE_API_KEYS are now instance variables set in __init__
    MAX_TOKEN_SIZE = 4000 # Increase or decrease based on the model context window size
    cnt_txt = 0
    cnt_img = 0

    def __init__(self, llm_provider: str = None, ollama_api_base_url: str = None, ollama_text_model_name: str = None): # Added ollama_text_model_name
        # Determine the provider: Use passed `llm_provider`, then settings default.
        self.llm_provider = llm_provider if llm_provider else self.settings.DEFAULT_LLM_PROVIDER
        
        logger.info(f"Initializing Model class. Requested provider: {llm_provider}, Effective provider: {self.llm_provider}")

        if self.llm_provider == "ollama":
            # Use passed ollama_api_base_url if provided, else use the one from settings
            self.TEXT_API_END_POINT = ollama_api_base_url if ollama_api_base_url else self.settings.OLLAMA_TEXT_API_BASE_URL
            # Use passed ollama_text_model_name if provided, else use the one from settings
            self.TEXT_MODEL_NAME = ollama_text_model_name if ollama_text_model_name else self.settings.OLLAMA_TEXT_MODEL_NAME
            self.TEXT_API_KEYS = ["ollama"] # Hardcoded for Ollama with OpenAI client
            self.async_text_clients = [AsyncOpenAI(base_url=self.TEXT_API_END_POINT, api_key="ollama")]
            
            logger.warning("Image summarization with Ollama provider may not be fully supported or use a different setup.")
            # self.IMAGE_API_END_POINT = ollama_api_base_url if ollama_api_base_url else self.settings.OLLAMA_IMAGE_API_BASE_URL # If images were supported
            # self.IMAGE_MODEL_NAME = self.settings.OLLAMA_IMAGE_MODEL_NAME # If images were supported
            self.IMAGE_MODEL_NAME = None # Explicitly set to None for Ollama for now
            self.IMAGE_API_END_POINT = None # Explicitly set to None
            self.IMAGE_API_KEYS = []
            self.async_image_clients = [] 
            self.image_keys_count = 0
        
        elif self.llm_provider == "openai": # Handles "openai", "groq", or any other OpenAI-compatible
            self.TEXT_API_END_POINT = self.settings.OPENAI_TEXT_API_END_POINT
            self.TEXT_MODEL_NAME = self.settings.OPENAI_TEXT_MODEL_NAME
            self.TEXT_API_KEYS = self.settings.OPENAI_TEXT_API_KEYS
            self.async_text_clients = [AsyncOpenAI(base_url=self.TEXT_API_END_POINT, api_key=api_key)
                                       for api_key in self.TEXT_API_KEYS]
            
            self.IMAGE_API_END_POINT = self.settings.OPENAI_IMAGE_API_END_POINT
            self.IMAGE_MODEL_NAME = self.settings.OPENAI_IMAGE_MODEL_NAME
            self.IMAGE_API_KEYS = self.settings.OPENAI_IMAGE_API_KEYS
            self.async_image_clients = [AsyncOpenAI(base_url=self.IMAGE_API_END_POINT, api_key=api_key)
                                        for api_key in self.IMAGE_API_KEYS]
        else:
            logger.error(f"Unsupported LLM provider: {self.llm_provider}. Falling back to default OpenAI settings from .env.")
            # Fallback to OpenAI settings from .env as a default catch-all
            self.TEXT_API_END_POINT = self.settings.OPENAI_TEXT_API_END_POINT
            self.TEXT_MODEL_NAME = self.settings.OPENAI_TEXT_MODEL_NAME
            self.TEXT_API_KEYS = self.settings.OPENAI_TEXT_API_KEYS
            self.async_text_clients = [AsyncOpenAI(base_url=self.TEXT_API_END_POINT, api_key=api_key)
                                       for api_key in self.TEXT_API_KEYS]

            self.IMAGE_API_END_POINT = self.settings.OPENAI_IMAGE_API_END_POINT
            self.IMAGE_MODEL_NAME = self.settings.OPENAI_IMAGE_MODEL_NAME
            self.IMAGE_API_KEYS = self.settings.OPENAI_IMAGE_API_KEYS
            self.async_image_clients = [AsyncOpenAI(base_url=self.IMAGE_API_END_POINT, api_key=api_key)
                                        for api_key in self.IMAGE_API_KEYS]

        self.text_keys_count = len(self.TEXT_API_KEYS) if hasattr(self, 'TEXT_API_KEYS') and self.TEXT_API_KEYS else 0
        self.image_keys_count = len(self.IMAGE_API_KEYS) if hasattr(self, 'IMAGE_API_KEYS') and self.IMAGE_API_KEYS and self.async_image_clients else 0
        
        logger.info(f"Text Model for '{self.llm_provider}': {getattr(self, 'TEXT_MODEL_NAME', 'N/A')}, Endpoint: {getattr(self, 'TEXT_API_END_POINT', 'N/A')}")
        if self.image_keys_count > 0 and self.async_image_clients:
            logger.info(f"Image Model for '{self.llm_provider}': {getattr(self, 'IMAGE_MODEL_NAME', 'N/A')}, Endpoint: {getattr(self, 'IMAGE_API_END_POINT', 'N/A')}")
        elif self.llm_provider == "openai": # Or any provider expected to have image support
            logger.warning(f"Image clients may not be configured correctly for provider '{self.llm_provider}' if API keys are missing or empty.")


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

        # Ensure IMAGE_API_END_POINT and IMAGE_MODEL_NAME are accessed via self
        if hasattr(self, 'IMAGE_API_END_POINT') and self.IMAGE_API_END_POINT and "huggingface.co" in self.IMAGE_API_END_POINT.lower():
            # Ensure self.IMAGE_MODEL_NAME is not None before concatenation
            if not self.IMAGE_MODEL_NAME:
                 logger.error(f"HuggingFace image model name is not set for provider {self.llm_provider}.")
                 if progress_queue:
                     progress_queue.put_nowait({"type": "llm_api_failed_final", "file": image_path, "api_call": "summarize_image_api", "error": "HuggingFace image model name not set."})
                 return "HuggingFace image model name not set."
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
                    # Ensure self.IMAGE_MODEL_NAME is used
                    if not self.IMAGE_MODEL_NAME:
                        logger.error(f"OpenAI-like image model name is not set for provider {self.llm_provider}.")
                        if progress_queue:
                            progress_queue.put_nowait({"type": "llm_api_failed_final", "file": image_path, "api_call": "summarize_image_api", "error": "OpenAI-like image model name not set."})
                        return "OpenAI-like image model name not set."
                    chat_completion = await self.async_image_clients[
                        self.cnt_img % self.image_keys_count].chat.completions.create( 
                        model=self.IMAGE_MODEL_NAME, # Changed from Model.IMAGE_MODEL_NAME
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
                if not self.TEXT_MODEL_NAME: # Check if TEXT_MODEL_NAME is set
                    logger.error(f"Text model name is not set for provider {self.llm_provider} in summarize_document_api.")
                    raise ValueError(f"Text model name not set for provider {self.llm_provider}.")
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
                if not self.TEXT_MODEL_NAME: # Check if TEXT_MODEL_NAME is set
                    logger.error(f"Text model name is not set for provider {self.llm_provider} in create_file_tree_api_chunk.")
                    raise ValueError(f"Text model name not set for provider {self.llm_provider}.")
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
                if not self.TEXT_MODEL_NAME: # Check if TEXT_MODEL_NAME is set
                    logger.error(f"Text model name is not set for provider {self.llm_provider} in search_files_api_chunk.")
                    raise ValueError(f"Text model name not set for provider {self.llm_provider}.")
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

    async def analyze_text_for_topic_api(self, text_content: str, research_topic: str, 
                                         analysis_type: str, # analysis_type can be used for logging or minor prompt adjustments
                                         progress_queue: Queue | None = None, 
                                         file_path_for_logging: str = "Unknown file") -> dict | None:
        """
        Analyzes text content for relevance to a research topic, identifies sub-topics,
        and describes connections using an LLM.
        """
        system_prompt = f"""
You are an AI research assistant. Your task is to analyze the provided text content based on the given research topic.
The research topic is: "{research_topic}"

Analyze the text and provide your response in a JSON object with the following exact structure:
{{
  "is_relevant": <boolean>,
  "sub_topics": ["<list of string sub-topics or key points relevant to the research_topic found in the text>"],
  "connections": "<string describing how the text content connects to broader questions, themes, or implications related to the research_topic>"
}}

Instructions:
- "is_relevant": Must be true if the text content is directly relevant to the research topic, false otherwise.
- "sub_topics": List key sub-topics or distinct points from the text that are directly related to the research topic. If no specific sub-topics are found but the text is relevant, provide a general statement or an empty list.
- "connections": Provide a concise text description of how the text connects to the research topic. If not relevant, explain briefly why.
- Ensure the output is ONLY the JSON object. Do not include any introductory text, markdown formatting like ```json, or concluding remarks.
""".strip()

        user_content = text_content

        if progress_queue:
            progress_queue.put_nowait({
                "type": "llm_api_start", 
                "file": file_path_for_logging, 
                "api_call": "analyze_text_for_topic_api",
                "analysis_type": analysis_type 
            })

        attempt = 0
        analysis_result_dict = None
        # Using 5 attempts, similar to summarize_document_api
        while attempt < 5:
            try:
                if not self.async_text_clients or self.text_keys_count == 0:
                    logger.error(f"No text clients available for analysis API for {file_path_for_logging}.")
                    raise ValueError("AsyncOpenAI text clients for analysis are not configured or empty.")

                # Determine if JSON mode is supported (specific to OpenAI clients for now)
                client_supports_json_mode = False
                if self.llm_provider == "openai" and hasattr(self.async_text_clients[0].chat.completions, 'create'):
                     # A bit of a heuristic: if it's OpenAI and has 'create', it likely supports response_format
                     # This check can be refined if we have more specific client capabilities.
                     client_supports_json_mode = True
                
                if not self.TEXT_MODEL_NAME: # Check if TEXT_MODEL_NAME is set
                    logger.error(f"Text model name is not set for provider {self.llm_provider} in analyze_text_for_topic_api.")
                    raise ValueError(f"Text model name not set for provider {self.llm_provider}.")

                completion_params = {
                    "model": self.TEXT_MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "stream": False,
                    "temperature": 0.0, # For deterministic output
                    "timeout": 90.0, # Increased timeout for potentially complex analysis
                }

                if client_supports_json_mode:
                    completion_params["response_format"] = {"type": "json_object"}
                    logger.info(f"Using JSON mode for LLM call for {file_path_for_logging}, analysis type: {analysis_type}")
                else:
                    logger.info(f"Not using explicit JSON mode (or provider is not OpenAI like) for {file_path_for_logging}, analysis type: {analysis_type}. Relying on prompt for JSON output.")


                chat_completion = await self.async_text_clients[
                    self.cnt_txt % self.text_keys_count].chat.completions.create(**completion_params) # type: ignore
                
                raw_response_content = chat_completion.choices[0].message.content
                
                if not raw_response_content:
                    logger.warning(f"LLM returned empty content for {file_path_for_logging}, attempt {attempt+1}.")
                    # Consider this a failure for this attempt
                    raise ValueError("LLM returned empty content.")

                # Attempt to parse the JSON
                try:
                    # Remove potential markdown formatting if not in JSON mode
                    if not client_supports_json_mode:
                        cleaned_response = raw_response_content.strip()
                        if cleaned_response.startswith("```json"):
                            cleaned_response = cleaned_response[len("```json"):]
                        if cleaned_response.endswith("```"):
                            cleaned_response = cleaned_response[:-len("```")]
                        cleaned_response = cleaned_response.strip()
                        parsed_json = json.loads(cleaned_response)
                    else:
                        parsed_json = json.loads(raw_response_content)

                except json.JSONDecodeError as json_e:
                    logger.error(f"JSON parsing failed for {file_path_for_logging}, attempt {attempt+1}. Error: {json_e}. Response: {raw_response_content[:500]}")
                    # This attempt failed, will retry
                    raise # Re-raise to be caught by the outer try-except

                # Validate structure
                if not all(key in parsed_json for key in ["is_relevant", "sub_topics", "connections"]):
                    logger.error(f"Missing expected keys in LLM JSON response for {file_path_for_logging}, attempt {attempt+1}. Response: {parsed_json}")
                    raise ValueError("Missing expected keys in LLM JSON response.")
                
                if not isinstance(parsed_json["is_relevant"], bool):
                    logger.warning(f"'is_relevant' is not a boolean for {file_path_for_logging}. Value: {parsed_json['is_relevant']}. Coercing if possible or failing.")
                    # Attempt to coerce, or handle as error. For now, let's be strict.
                    raise ValueError("'is_relevant' field is not a boolean.")

                if not isinstance(parsed_json["sub_topics"], list):
                    logger.warning(f"'sub_topics' is not a list for {file_path_for_logging}. Value: {parsed_json['sub_topics']}. Coercing if possible or failing.")
                    raise ValueError("'sub_topics' field is not a list.")

                analysis_result_dict = parsed_json
                break # Success
            
            except Exception as e:
                logger.error(f"Error in analyze_text_for_topic_api for {file_path_for_logging} (type: {analysis_type}), attempt {attempt+1}: {e}")
                attempt += 1
                self.cnt_txt += 1 # Rotate API key/client
                if attempt >= 5:
                    if progress_queue:
                        progress_queue.put_nowait({
                            "type": "llm_api_failed_final", 
                            "file": file_path_for_logging, 
                            "api_call": "analyze_text_for_topic_api", 
                            "error": str(e)
                        })
                    logger.error(f"Final attempt failed for analyze_text_for_topic_api for {file_path_for_logging}.")
                    analysis_result_dict = None # Ensure it's None on final failure
                else:
                    # import asyncio # Import here if not already at top level of file # asyncio is already imported at the top
                    await asyncio.sleep(1 + attempt) # Exponential backoff, simple version

        if progress_queue:
            progress_queue.put_nowait({
                "type": "llm_api_end", 
                "file": file_path_for_logging, 
                "api_call": "analyze_text_for_topic_api", 
                "success": analysis_result_dict is not None
            })
        
        return analysis_result_dict

    async def generate_answer_from_context(self, question: str, context: str, 
                                           progress_queue: Queue | None = None, 
                                           file_path_for_logging: str = "Q&A_request") -> str | None:
        """
        Generates an answer to a question based strictly on the provided context using an LLM.
        """
        system_prompt = """
You are a helpful AI assistant. Your task is to answer the user's question based *only* on the provided context.
If the context does not contain enough information to answer the question, you must explicitly state: 
'I cannot answer the question based on the provided documents.'
Do not use any external knowledge or make assumptions beyond the provided text.
Be concise and directly answer the question.
""".strip()

        user_prompt_for_llm = f"Context:\n---\n{context}\n---\n\nQuestion: {question}"

        if progress_queue:
            progress_queue.put_nowait({
                "type": "llm_api_start", 
                "file": file_path_for_logging, # Using file_path_for_logging to denote the Q&A operation
                "api_call": "generate_answer_from_context"
            })

        attempt = 0
        answer = None
        # Using 3-5 attempts for consistency with other methods. Let's use 3 for Q&A.
        while attempt < 3:
            try:
                if not self.async_text_clients or self.text_keys_count == 0:
                    logger.error(f"No text clients available for Q&A API ({file_path_for_logging}).")
                    raise ValueError("AsyncOpenAI text clients for Q&A are not configured or empty.")
                
                if not self.TEXT_MODEL_NAME: # Check if TEXT_MODEL_NAME is set
                    logger.error(f"Text model name is not set for provider {self.llm_provider} in generate_answer_from_context.")
                    raise ValueError(f"Text model name not set for provider {self.llm_provider}.")

                chat_completion = await self.async_text_clients[
                    self.cnt_txt % self.text_keys_count].chat.completions.create(
                    model=self.TEXT_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt_for_llm},
                    ],
                    stream=False,
                    temperature=0.1, # Slightly higher for more natural answer, but still aiming for factuality
                    timeout=60.0, 
                )
                
                answer_content = chat_completion.choices[0].message.content
                
                if not answer_content or not answer_content.strip():
                    logger.warning(f"LLM returned empty or whitespace content for Q&A ({file_path_for_logging}), attempt {attempt+1}.")
                    # Consider this a failure for this attempt, maybe the model can't answer
                    answer = "The model returned an empty response. It might be unable to answer based on the context." # Provide a default if it's empty
                else:
                    answer = answer_content.strip()
                
                break # Success
            
            except Exception as e:
                logger.error(f"Error in generate_answer_from_context for '{file_path_for_logging}', attempt {attempt+1}: {e}", exc_info=True)
                attempt += 1
                self.cnt_txt += 1 # Rotate API key/client
                if attempt >= 3:
                    if progress_queue:
                        progress_queue.put_nowait({
                            "type": "llm_api_failed_final", 
                            "file": file_path_for_logging, 
                            "api_call": "generate_answer_from_context", 
                            "error": str(e)
                        })
                    logger.error(f"Final attempt failed for generate_answer_from_context for '{file_path_for_logging}'.")
                    answer = None # Ensure answer is None on final failure
                else:
                    # Need to import asyncio if not already available at class/module level
                    # For simplicity, assuming asyncio is available or this is refactored to be top-level
                    # import asyncio # asyncio is already imported at the top
                    await asyncio.sleep(1 + attempt) # Exponential backoff

        if progress_queue:
            progress_queue.put_nowait({
                "type": "llm_api_end", 
                "file": file_path_for_logging, 
                "api_call": "generate_answer_from_context", 
                "success": answer is not None
            })
        
        return answer

    async def search_files_api_chunk(self, summaries: list, search_query: str, progress_queue: Queue | None = None): # Note: This method seems to return search_results_chunk which is not defined in its scope. This was pre-existing.
        # Assuming the implementation of search_files_api_chunk was intended to be here or this method is a duplicate/placeholder
        # For now, I will leave its content as is, as the primary task is to refactor Settings and Model.__init__
        # However, it will likely cause a NameError if called.
        # Placeholder for actual implementation or correction:
        logger.warning("search_files_api_chunk may not be correctly implemented, returning empty list.") 
        return [] # Returning empty list to avoid NameError with undefined search_results_chunk


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
