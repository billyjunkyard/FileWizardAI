import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';


@Injectable({
  providedIn: 'root'
})
export class DataService {

  private apiUrl = 'http://localhost:8000'; // This should align with your backend

  constructor(private http: HttpClient) { // HttpClient might still be used by other methods
  }

  public getFormattedFiles(params: HttpParams): Observable<any> {
    const url = `${this.apiUrl}/get_files?${params.toString()}`; // Construct URL with params

    return new Observable(observer => {
      const eventSource = new EventSource(url);

      eventSource.onmessage = (event) => {
        // console.log('SSE onmessage event:', event); // For debugging
        if (event.data) {
          try {
            const eventData = JSON.parse(event.data);
            // console.log('SSE parsed data:', eventData); // For debugging
            
            // Based on server.py, the backend sends events like:
            // {"event": "task_started", "data": {"task_id": "..."}}
            // {"event": "progress", "data": {"type": "...", "message": "..."}}
            // {"event": "task_completed", "data": {"root_path": "...", "items": [...]}}
            // The event.data from EventSource is the stringified content of the "data" field in the SSE structure,
            // which itself is a JSON string. So eventData is already the actual payload object.
            
            observer.next(eventData); // Pass the parsed data payload

            // Check if the event signifies the end of the stream
            if (eventData.event === 'task_completed' || eventData.event === 'task_error' || eventData.event === 'task_cancelled') {
              observer.complete(); // Signal completion of the stream
              eventSource.close();
              // console.log(`SSE stream completed with event: ${eventData.event}`); // For debugging
            }
          } catch (e) {
            console.error('Error parsing SSE event data:', e, 'Raw data:', event.data);
            observer.error({ message: 'Error parsing SSE event data.', rawData: event.data });
            eventSource.close();
          }
        } else {
          console.warn('SSE event without data:', event);
        }
      };

      eventSource.onerror = (error) => {
        console.error('SSE error event:', error);
        let errorMsg = 'Error connecting to Server-Sent Events endpoint.';
        
        if (eventSource.readyState === EventSource.CLOSED) {
          errorMsg = 'SSE connection was closed by the server or due to a network error.';
           // For CLOSED state, it's often better to complete rather than error,
           // unless this indicates an unexpected closure.
           // If the stream is meant to complete, this might be handled by a specific terminal event.
           // If it's an abrupt closure, erroring is appropriate.
           // Given the terminal events above, an onerror when CLOSED might be an unexpected termination.
        } else if (eventSource.readyState === EventSource.CONNECTING) {
          // This means it failed and might be retrying, or has permanently failed after retries.
          errorMsg = 'SSE connection failed. The server might be down or unreachable.';
        }
        
        observer.error({ message: errorMsg, originalError: error });
        eventSource.close(); // Ensure it's closed on error
      };

      // Return a teardown logic function that will be called when the observer unsubscribes
      return () => {
        if (eventSource.readyState !== EventSource.CLOSED) {
          eventSource.close();
          // console.log('SSE EventSource closed due to unsubscription.'); // For debugging
        }
      };
    });
  }

  updateStructure(newStructureBody: any): Observable<any> {
    return this.http.post<any>(this.apiUrl + "/update_files", newStructureBody);
  }

  getSearchFiles(params: HttpParams) { // Assuming this might also be SSE or needs similar update later
    // For now, keeping it as is, but if it's also SSE, it would need the same pattern
    return this.http.get(this.apiUrl + "/search_files", { params: params });
  }

  openFile(fileToOpen: any): Observable<any> {
    return this.http.post<any>(this.apiUrl + "/open_file", { file_path: fileToOpen });
  }

  semanticSearch(queryText: string, topN: number, filePaths?: string[]): Observable<any> {
    let params = new HttpParams()
      .set('query_text', queryText)
      .set('top_n', topN.toString());

    if (filePaths && filePaths.length > 0) {
      params = params.set('file_paths_json', JSON.stringify(filePaths));
    }
    return this.http.get<any>(`${this.apiUrl}/semantic_search/`, { params });
  }

  answerQuestion(
    queryText: string, 
    topN: number, 
    filePaths?: string[], 
    llmProvider?: string, 
    ollamaApiBaseUrl?: string, 
    ollamaTextModel?: string
  ): Observable<{ answer: string, source_chunks: any[] }> {
    let params = new HttpParams()
      .set('query_text', queryText)
      .set('top_n_chunks', topN.toString()); 

    if (filePaths && filePaths.length > 0) {
      params = params.set('file_paths_json', JSON.stringify(filePaths));
    }

    if (llmProvider) {
      params = params.set('llm_provider', llmProvider);
      if (llmProvider === 'ollama') {
        if (ollamaApiBaseUrl) {
          params = params.set('ollama_api_base_url', ollamaApiBaseUrl);
        }
        if (ollamaTextModel) {
          params = params.set('ollama_text_model_name', ollamaTextModel);
        }
      }
    }
    return this.http.get<{ answer: string, source_chunks: any[] }>(`${this.apiUrl}/answer_question/`, { params });
  }

  getOllamaModels(ollamaApiBaseUrl: string): Observable<{models: string[]}> {
    const params = new HttpParams().set('ollama_api_base_url', ollamaApiBaseUrl);
    return this.http.get<{models: string[]}>(`${this.apiUrl}/ollama/models`, { params });
  }
}
