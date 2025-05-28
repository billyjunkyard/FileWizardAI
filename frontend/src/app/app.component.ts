import { Component, HostBinding, Input, QueryList, ViewChildren } from '@angular/core';
import { DataService } from './data.service';
import { HttpParams } from "@angular/common/http";
import { FolderTreeComponent } from './components/folder-tree.component';
import { NgModel } from '@angular/forms';

interface ExtensionGroup {
  name: string;
  icon: string;
  extensions: string[];
  selected: number;
  total: number;
  expanded: boolean;
}

@Component({
  selector: 'app-root',
  template: `
    <div class="app-container">
      <button class="theme-toggle" (click)="toggleTheme()">
        <mat-icon>{{ isDarkTheme ? 'light_mode' : 'dark_mode' }}</mat-icon>
        {{ isDarkTheme ? 'Light' : 'Dark' }} Mode
      </button>

      <div class="header">
        <div class="header-content">
          <h1>Welcome to FileWizard AI</h1>
          <p class="subtitle">Intelligent file management at your fingertips</p>
          <a class="github-link" href="https://github.com/AIxHunter/FileWizardAI/tree/main" target="_blank">
            <div class="github-icons">
              <svg class="github-logo" height="24" viewBox="0 0 16 16" width="24">
                <path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
              </svg>
              <span>View on GitHub</span>
              <mat-icon class="arrow">arrow_forward</mat-icon>
            </div>
          </a>
        </div>
        <div class="header-background"></div>
      </div>

      <div class="content">
        <div class="main-section">
          <div class="section-header">
            <div class="icon-title">
              <mat-icon>folder_open</mat-icon>
              <div>
                <h2>File Structure Manager</h2>
                <p>Configure and organize your files intelligently</p>
              </div>
            </div>
          </div>

          <div class="input-section">
            <mat-form-field appearance="outline" class="root-path-field">
              <mat-label>Root Path</mat-label>
              <input matInput [(ngModel)]="rootPath" (ngModelChange)="onPathChange($event)" placeholder="Click the folder icon to select a directory">
              <mat-icon matSuffix>folder_open</mat-icon>
            </mat-form-field>
          </div>

          <div class="llm-config-section">
            <mat-form-field appearance="outline" class="llm-provider-field">
              <mat-label>LLM Provider</mat-label>
              <mat-select [(ngModel)]="selectedLLMProvider" (selectionChange)="onLlmProviderChange()">
                <mat-option *ngFor="let provider of llmProviders" [value]="provider.value">
                  {{provider.viewValue}}
                </mat-option>
              </mat-select>
            </mat-form-field>

            <mat-form-field appearance="outline" class="ollama-url-field" *ngIf="selectedLLMProvider === 'ollama'">
              <mat-label>Ollama API Base URL</mat-label>
              <input matInput [(ngModel)]="ollamaApiBaseUrl" 
                     (ngModelChange)="checkOllamaUrlAndFetchModels()"
                     placeholder="e.g., http://localhost:11434">
              <mat-icon matSuffix *ngIf="ollamaApiUrlStatus === 'loading'">hourglass_empty</mat-icon>
              <mat-icon matSuffix *ngIf="ollamaApiUrlStatus === 'success'" style="color: green;">check_circle</mat-icon>
              <mat-icon matSuffix *ngIf="ollamaApiUrlStatus === 'error'" style="color: red;" [matTooltip]="ollamaApiErrorMsg || 'Error connecting to Ollama'">error</mat-icon>
            </mat-form-field>
            <div *ngIf="selectedLLMProvider === 'ollama' && ollamaApiUrlStatus === 'error' && ollamaApiErrorMsg" class="ollama-error-message">
              {{ ollamaApiErrorMsg }}
            </div>

            <mat-form-field appearance="outline" class="ollama-model-select-field" *ngIf="selectedLLMProvider === 'ollama' && ollamaApiUrlStatus === 'success' && ollamaModels.length > 0">
              <mat-label>Ollama Text Model</mat-label>
              <mat-select [(ngModel)]="selectedOllamaTextModel">
                <mat-option *ngFor="let model of ollamaModels" [value]="model">
                  {{ model }}
                </mat-option>
              </mat-select>
            </mat-form-field>
            <div *ngIf="selectedLLMProvider === 'ollama' && ollamaApiUrlStatus === 'success' && ollamaModels.length === 0" class="ollama-no-models-message">
              No Ollama models found at the specified address. Please check your Ollama setup.
            </div>


            <mat-form-field appearance="outline" class="custom-prompt-field">
              <mat-label>Custom Summarization Prompt (Optional)</mat-label>
              <textarea matInput
                        [(ngModel)]="customSummarizationPrompt"
                        rows="3"
                        placeholder="Enter your custom prompt for summarizing documents. If left empty, a default prompt will be used."></textarea>
            </mat-form-field>
          </div>

          <!-- Deep Analysis Section -->
          <div class="deep-analysis-section">
            <mat-checkbox [(ngModel)]="deepAnalysisModeEnabled" color="primary" class="deep-analysis-mode-check">
              Enable Deep Analysis Mode
            </mat-checkbox>

            <div *ngIf="deepAnalysisModeEnabled" class="deep-analysis-options">
              <mat-form-field appearance="outline" class="research-topic-field">
                <mat-label>Research Topic/Question for Analysis</mat-label>
                <input matInput [(ngModel)]="researchTopicPrompt" placeholder="Enter topic or question for analysis features">
              </mat-form-field>

              <mat-checkbox [(ngModel)]="quickTopicAnalysisEnabled" 
                            [disabled]="!researchTopicPrompt || !researchTopicPrompt.trim()"
                            color="primary" 
                            matTooltip="Analyzes the summary of the document. Requires a Research Topic.">
                Quick Topic Analysis (uses document summary)
              </mat-checkbox>
              <mat-checkbox [(ngModel)]="fullDocTopicAnalysisEnabled" 
                            [disabled]="!researchTopicPrompt || !researchTopicPrompt.trim()"
                            color="primary" 
                            matTooltip="Analyzes the full text of the document. Requires a Research Topic.">
                In-depth Topic Analysis (uses full document text)
              </mat-checkbox>
              <mat-checkbox [(ngModel)]="semanticSearchEnabled" 
                            color="primary"
                            matTooltip="Processes all document chunks for semantic search and Q&A. Can be time-consuming.">
                Enable Semantic Search & Q&A
              </mat-checkbox>
            </div>
          </div>
          <!-- End Deep Analysis Section -->

          <div class="extensions-section">
            <div class="extensions-header">
              <h3>File Extensions</h3>
              <div class="extension-actions">
                <button mat-button color="primary" (click)="selectAll()">
                  <mat-icon>select_all</mat-icon>
                  Select All
                </button>
                <button mat-button color="warn" (click)="clearAll()">
                  <mat-icon>clear_all</mat-icon>
                  Clear
                </button>
              </div>
            </div>

            <div class="extension-groups">
              <div *ngFor="let group of extensionGroups" class="extension-group">
                <div class="group-header" (click)="toggleGroup(group)">
                  <div class="group-info">
                    <mat-icon>{{group.icon}}</mat-icon>
                    <span>{{group.name}}</span>
                  </div>
                  <div class="group-count">
                    {{group.selected}}/{{group.total}}
                    <mat-icon class="expand-icon" [class.expanded]="group.expanded">expand_more</mat-icon>
                  </div>
                </div>
                <div class="group-content" [class.expanded]="group.expanded">
                  <mat-checkbox *ngFor="let ext of group.extensions"
                              [checked]="isExtensionSelected(ext)"
                              (change)="toggleExtension(ext, group)"
                              color="primary">
                    {{ext}}
                  </mat-checkbox>
                </div>
              </div>
            </div>
          </div>

          <div class="actions-section">
            <mat-checkbox [(ngModel)]="isRecursive" color="primary" class="subdirectories-check">
              Include Subdirectories
            </mat-checkbox>
            
            <button mat-flat-button color="primary" (click)="getFiles()" class="get-files-btn">
              <mat-icon>search</mat-icon>
              GET FILES
            </button>
          </div>
        </div>

        <div class="search-section"> 
          <div class="section-header">
            <div class="icon-title">
              <mat-icon>search</mat-icon>
              <div>
                <h2>File Search</h2>
                <p>Search and locate files in your directory</p>
              </div>
            </div>
          </div>

          <app-search-files [rootPath]="rootPath" 
                           [isRecursive]="isRecursive" 
                           [filesExts]="filesExts">
          </app-search-files>
          
          <div class="semantic-qa-section">
            <div class="section-header">
                <div class="icon-title">
                  <mat-icon>travel_explore</mat-icon> 
                  <div>
                    <h2>Semantic Document Search & Q&A</h2>
                    <p>Ask questions or find similar content across your documents.</p>
                  </div>
                </div>
            </div>

            <mat-form-field appearance="outline" class="search-query-field">
              <mat-label>Enter Search Query or Question</mat-label>
              <textarea matInput [(ngModel)]="searchQueryText" rows="3" placeholder="Type your semantic query or question here..."></textarea>
            </mat-form-field>

            <div class="search-controls">
              <mat-form-field appearance="outline" class="top-n-field">
                <mat-label>Top N Results</mat-label>
                <input matInput type="number" [(ngModel)]="searchTopN" min="1" max="20">
              </mat-form-field>

              <mat-form-field appearance="outline" class="file-filter-field">
                <mat-label>Filter by File Paths (Optional)</mat-label>
                <input matInput [(ngModel)]="searchFilePathsInput" placeholder="e.g., path/to/file1.txt, another/doc.pdf">
                <mat-hint>Comma-separated relative paths.</mat-hint>
              </mat-form-field>
            </div>

            <div class="search-buttons">
              <button mat-stroked-button color="primary" (click)="performSemanticSearch()" [disabled]="isLoadingSearch" class="action-button">
                <mat-icon>search</mat-icon> Semantic Search
              </button>
              <button mat-stroked-button color="accent" (click)="performQuestionAnswering()" [disabled]="isLoadingSearch" class="action-button">
                <mat-icon>question_answer</mat-icon> Get Answer (Q&A)
              </button>
            </div>
            
            <div *ngIf="isLoadingSearch" class="loading-indicator">
              <mat-progress-spinner mode="indeterminate" diameter="30"></mat-progress-spinner>
              <span>Processing your query...</span>
            </div>

            <div *ngIf="searchError" class="error-message search-error-message">
              <mat-icon>error</mat-icon> {{searchError}}
            </div>

            <div *ngIf="searchResults && searchResults.length > 0" class="results-area semantic-results-area">
              <h3>Semantic Search Results:</h3>
              <div *ngFor="let result of searchResults" class="result-item chunk-item-card">
                <p><strong>File:</strong> {{result.file_path}} (Chunk ID: {{result.chunk_id_db}})</p>
                <p><strong>Distance:</strong> {{result.distance?.toFixed(4)}}</p>
                <p class="chunk-text-display"><strong>Text:</strong> {{result.chunk_text}}</p>
              </div>
            </div>

            <div *ngIf="qaAnswer" class="results-area qa-results-area">
              <h3>Answer:</h3>
              <p class="qa-answer-text">{{qaAnswer.answer}}</p>
              <div *ngIf="qaAnswer.source_chunks && qaAnswer.source_chunks.length > 0" class="source-chunks-area">
                <h4>Source Chunks:</h4>
                <div *ngFor="let chunk of qaAnswer.source_chunks" class="result-item chunk-item-card">
                  <p><strong>File:</strong> {{chunk.file_path}} (Chunk ID: {{chunk.chunk_id_db}})</p>
                  <p><strong>Distance:</strong> {{chunk.distance?.toFixed(4)}}</p>
                  <p class="chunk-text-display"><strong>Text:</strong> {{chunk.chunk_text}}</p>
                </div>
              </div>
            </div>
          </div>


          <div class="trees-container" *ngIf="srcPaths">
            <div class="structure-panel">
              <app-folder-tree [paths]="srcPaths" 
                             [rootPath]="rootPath"
                             [headline]="'Current Structure'"
                             [index]=0 
                             (notify)="onNotify($event)">
              </app-folder-tree>
            </div>

            <div class="structure-panel">
              <app-folder-tree [paths]="dstPaths" 
                             [rootPath]="rootPath"
                             [headline]="'Optimized Structure'"
                             [index]=1 
                             (notify)="onNotify($event)">
              </app-folder-tree>
            </div>
          </div>

          <div class="update-section" *ngIf="original_files">
            <button mat-flat-button color="primary" (click)="updateStructure()">
              <mat-icon>auto_fix_high</mat-icon>
              Update Structure
            </button>

            <div class="messages">
              <div *ngIf="successMessage" class="success-message">
                <mat-icon>check_circle</mat-icon>
                {{successMessage}}
              </div>
              <div *ngIf="errorMessage" class="error-message">
                <mat-icon>error</mat-icon>
                {{errorMessage}}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="file-details-section" *ngIf="original_files && original_files.items && original_files.items.length > 0">
        <div class="section-header">
          <div class="icon-title">
            <mat-icon>insights</mat-icon> 
            <div>
              <h2>File Analysis Details</h2>
              <p>Detailed summary and topic analysis for processed files</p>
            </div>
          </div>
        </div>
        <div *ngFor="let file of original_files.items" class="file-item-card">
          <h4>{{ file.file_path }}</h4> 
          <p><strong>Summary:</strong> {{ file.summary || 'N/A' }}</p>
          
          <div *ngIf="file.research_topic" class="analysis-subsection">
            <h5>Research Topic Analysis</h5>
            <p><strong>Analyzed for Topic:</strong> {{ file.research_topic }}</p>
            <p><strong>Analysis Scope:</strong> {{ file.analysis_type || 'N/A' }}</p>
            <p><strong>Relevant to Topic:</strong> 
              <span [ngClass]="{'relevant': file.is_topic_relevant, 'not-relevant': !file.is_topic_relevant && file.is_topic_relevant !== null}">
                {{ file.is_topic_relevant === null || file.is_topic_relevant === undefined ? 'N/A' : (file.is_topic_relevant ? 'Yes' : 'No') }}
              </span>
            </p>
            
            <div *ngIf="file.sub_topics && file.sub_topics.length > 0">
              <strong>Key Sub-topics:</strong>
              <ul>
                <li *ngFor="let sub_topic of file.sub_topics">{{ sub_topic }}</li>
              </ul>
            </div>
            <p *ngIf="!file.sub_topics || file.sub_topics.length === 0"><strong>Key Sub-topics:</strong> N/A</p>
            
            <p><strong>Connections to Broader Themes:</strong> {{ file.topic_connections || 'N/A' }}</p>
          </div>
          <mat-divider *ngIf="!last"></mat-divider>
        </div>
      </div>
    </div>
  `,
  styles: [`
    :host {
      display: block;
      min-height: 100vh;
      background-color: var(--background);
      color: var(--text-primary);
      font-family: 'Roboto', sans-serif;
    }

    .app-container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 2rem;
      animation: fadeIn 0.5s ease;
    }

    .ollama-url-field .mat-icon { 
      cursor: default; 
    }
    .ollama-error-message {
      color: red; 
      font-size: 0.75em; 
      margin-top: -0.85em; 
      margin-left: 0.5em; 
      margin-bottom: 0.5em;
    }
    .ollama-model-select-field {
      width: 100%;
      margin-bottom: 1rem; 
    }
    .ollama-no-models-message {
      font-size: 0.8em;
      color: var(--text-secondary); 
      margin-top: -0.75em;
      margin-bottom: 0.5em;
      padding: 0.5em;
      background-color: rgba(var(--primary-rgb), 0.05); 
      border-radius: 4px;
    }


    .deep-analysis-section {
      background-color: rgba(var(--primary-rgb), 0.03); 
      padding: 1.5rem;
      border-radius: var(--radius-md, 8px); 
      margin-bottom: 2rem;
      border: 1px solid rgba(var(--primary-rgb), 0.1);
    }

    .deep-analysis-mode-check {
      margin-bottom: 1rem; 
    }

    .deep-analysis-options {
      display: flex;
      flex-direction: column;
      gap: 1rem; 
      padding-left: 1rem; 
      border-left: 2px solid rgba(var(--primary-rgb), 0.2); 
      margin-top: 1rem;
    }

    .deep-analysis-options mat-form-field,
    .deep-analysis-options mat-checkbox {
      width: 100%;
    }
    
    .research-topic-field input {
       color: var(--text-primary); 
    }

    .file-details-section {
      margin-top: 2rem;
      background: var(--surface);
      border-radius: var(--radius-lg);
      padding: 2rem;
      border: 1px solid var(--border);
    }

    .file-item-card {
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      background-color: rgba(var(--background-rgb), 0.5); 
      border-radius: var(--radius-md);
      border: 1px solid rgba(var(--border-rgb, var(--primary-rgb)), 0.1); 
      box-shadow: var(--shadow-sm, 0 1px 3px rgba(0,0,0,0.05)); 
    }

    .file-item-card h4 {
      color: var(--primary);
      margin-top: 0;
      margin-bottom: 1rem;
      font-size: 1.1rem;
      font-weight: 500;
      word-break: break-all; 
    }

    .file-item-card p {
      margin-bottom: 0.5rem;
      font-size: 0.9rem;
      line-height: 1.6;
    }
    
    .analysis-subsection {
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px dashed rgba(var(--primary-rgb), 0.2);
    }

    .analysis-subsection h5 {
      font-size: 1rem;
      font-weight: 500;
      color: var(--primary);
      margin-bottom: 0.75rem;
    }

    .analysis-subsection ul {
      padding-left: 1.5rem;
      margin-top: 0.5rem;
    }
    .analysis-subsection li {
      margin-bottom: 0.25rem;
    }

    .relevant {
      color: #4CAF50; /* Green for Yes */
      font-weight: bold;
    }
    .not-relevant {
      color: #F44336; /* Red for No */
      font-weight: bold;
    }
    
    /* Semantic Search & Q&A Styles */
    .semantic-qa-section {
      margin-top: 2.5rem; /* Space above this new section */
      padding: 1.5rem;
      background-color: rgba(var(--primary-rgb), 0.02); /* Very subtle background */
      border-radius: var(--radius-md);
      border: 1px solid rgba(var(--primary-rgb), 0.08);
    }

    .search-query-field {
      width: 100%;
      margin-bottom: 1rem;
    }
    .search-query-field textarea {
       color: var(--text-primary);
    }


    .search-controls {
      display: flex;
      gap: 1rem;
      margin-bottom: 1rem;
      align-items: flex-start; /* Align items to the top */
    }

    .top-n-field {
      width: 120px; /* Smaller width for Top N */
    }
    .top-n-field input {
       color: var(--text-primary);
    }


    .file-filter-field {
      flex-grow: 1; /* Allow file filter to take remaining space */
    }
    .file-filter-field input {
       color: var(--text-primary);
    }

    .search-buttons {
      display: flex;
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    
    .action-button mat-icon {
        margin-right: 8px; /* Space between icon and text */
    }

    .loading-indicator {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin: 1rem 0;
      color: var(--text-secondary);
    }
    
    .search-error-message {
        margin-top: 1rem; /* Ensure spacing from buttons */
    }

    .results-area {
      margin-top: 1.5rem;
    }
    
    .results-area h3 {
        font-size: 1.2rem;
        color: var(--primary);
        margin-bottom: 1rem;
    }
    .results-area h4 {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .chunk-item-card {
      padding: 1rem;
      margin-bottom: 1rem;
      background-color: var(--surface-variant);
      border-radius: var(--radius-sm);
      border: 1px solid var(--border);
    }
    .chunk-item-card p {
        margin-bottom: 0.3rem;
        font-size: 0.85rem;
    }
    .chunk-text-display {
        white-space: pre-wrap; /* Preserve whitespace and wrap text */
        background-color: rgba(var(--background-rgb), 0.7);
        padding: 0.5rem;
        border-radius: var(--radius-sm);
        max-height: 150px; /* Limit height of chunk display */
        overflow-y: auto; /* Allow scrolling for long chunks */
    }
    
    .qa-answer-text {
        font-size: 1rem;
        line-height: 1.7;
        padding: 1rem;
        background-color: var(--surface-variant);
        border-radius: var(--radius-md);
        white-space: pre-wrap; /* To respect newlines in the answer */
    }
    .source-chunks-area {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
    }


    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .header {
      text-align: center;
      margin-bottom: 4rem;
      padding: 4rem 2rem;
      position: relative;
      overflow: hidden;
      background: linear-gradient(180deg, 
        rgba(0, 191, 165, 0.03) 0%,
        rgba(100, 255, 218, 0.02) 100%
      );
      backdrop-filter: blur(10px);
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .header-content {
      position: relative;
      z-index: 2;
      max-width: 800px;
      margin: 0 auto;
    }

    .header-background {
      position: absolute;
      top: -50%;
      left: -50%;
      right: -50%;
      bottom: -50%;
      background: 
        radial-gradient(circle at 20% 30%, rgba(0, 191, 165, 0.03) 0%, transparent 70%),
        radial-gradient(circle at 80% 70%, rgba(100, 255, 218, 0.03) 0%, transparent 70%);
      transform-origin: center;
      animation: gentleRotate 30s linear infinite;
      z-index: 1;
      filter: blur(30px);
    }

    .header h1 {
      font-size: 4rem;
      font-weight: 700;
      margin: 0;
      background: linear-gradient(135deg, 
        rgba(0, 191, 165, 1) 0%,
        rgba(100, 255, 218, 1) 50%,
        rgba(0, 191, 165, 1) 100%
      );
      background-size: 200% auto;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.02em;
      line-height: 1.1;
      animation: shimmerText 8s linear infinite;
      text-shadow: 0 2px 10px rgba(0, 191, 165, 0.2);
    }

    .subtitle {
      color: var(--text-secondary);
      font-size: 1.3rem;
      margin-top: 1.5rem;
      font-weight: 400;
      letter-spacing: 0.02em;
      opacity: 0.9;
      animation: fadeInUp 0.8s ease 0.2s both;
      text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }

    .github-link {
      display: inline-flex;
      align-items: center;
      gap: 0.75rem;
      color: var(--text-primary);
      text-decoration: none;
      margin-top: 2.5rem;
      padding: 0.75rem 1.75rem;
      border-radius: 50px;
      background: rgba(0, 191, 165, 0.1);
      border: 1px solid rgba(0, 191, 165, 0.2);
      transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
      animation: fadeInUp 0.8s ease 0.4s both;
      font-weight: 500;
      position: relative;
      overflow: hidden;
      backdrop-filter: blur(5px);

      .github-icons {
        display: flex;
        align-items: center;
        gap: 0.75rem;
      }

      .github-logo {
        color: var(--primary);
        transition: all 0.3s ease;
        animation: floatIcon 6s ease-in-out infinite;
      }

      span {
        font-size: 0.95rem;
        letter-spacing: 0.02em;
      }

      .arrow {
        font-size: 18px;
        transition: all 0.3s ease;
        opacity: 0;
        transform: translateX(-10px);
        color: var(--primary);
        margin-left: 4px;
      }

      &:hover {
        color: var(--primary);
        background: rgba(0, 191, 165, 0.15);
        border-color: rgba(0, 191, 165, 0.5);
        transform: translateY(-2px);
        padding-right: 2.25rem;
        box-shadow: 
          0 4px 20px rgba(0, 191, 165, 0.2),
          0 0 0 2px rgba(0, 191, 165, 0.1);

        &::before {
          animation: shimmer 1s forwards;
        }

        .github-logo {
          transform: scale(1.1);
          filter: drop-shadow(0 0 8px rgba(0, 191, 165, 0.5));
        }

        .arrow {
          opacity: 1;
          transform: translateX(0);
        }
      }
    }

    @keyframes gentleRotate {
      from {
        transform: rotate(0deg);
      }
      to {
        transform: rotate(360deg);
      }
    }

    @keyframes shimmerText {
      to {
        background-position: 200% center;
      }
    }

    @keyframes floatIcon {
      0%, 100% {
        transform: translateY(0);
      }
      50% {
        transform: translateY(-3px);
      }
    }

    @keyframes shimmer {
      to {
        transform: translateX(100%);
      }
    }

    .content {
      display: grid;
      grid-template-columns: 1fr 1.5fr;
      gap: 2rem;
      margin-bottom: 2rem;
    }

    .main-section, .search-section {
      background: var(--surface);
      border-radius: var(--radius-lg);
      padding: 2rem;
      border: 1px solid var(--border);
      height: fit-content;
      transition: transform 0.3s ease, box-shadow 0.3s ease;

      &:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
      }
    }

    .section-header {
      margin-bottom: 2rem;
    }

    .icon-title {
      display: flex;
      align-items: center;
      gap: 1rem;

      mat-icon {
        font-size: 2rem;
        width: 2rem;
        height: 2rem;
        color: var(--primary);
        animation: gentleFloat 6s ease-in-out infinite;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
      }

      &:hover mat-icon {
        animation: elegantPulse 2s ease-in-out infinite;
      }
    }

    .icon-title h2 {
      margin: 0;
      font-size: 1.5rem;
      font-weight: 500;
      color: var(--text-primary);
    }

    .icon-title p {
      margin: 0.25rem 0 0 0;
      color: var(--text-secondary);
      font-size: 0.9rem;
    }

    .input-section {
      margin-bottom: 2rem;
    }

    .root-path-field, .llm-provider-field, .ollama-url-field, .ollama-model-select-field, .custom-prompt-field {
      width: 100%;
      margin-bottom: 1rem; 
      color: var(--text-primary);
    }

    .root-path-field input, .ollama-url-field input, .custom-prompt-field textarea {
      color: var(--text-primary);
    }
    
    .llm-config-section {
        margin-bottom: 2rem;
    }

    .search-text {
      color: var(--text-primary);
    }

    .extensions-section {
      margin-bottom: 2rem;
    }

    .extensions-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
    }

    .extensions-header h3 {
      margin: 0;
      font-size: 1rem;
      font-weight: 500;
      color: var(--text-primary);
    }

    .extension-actions {
      display: flex;
      gap: 0.5rem;
    }

    .extension-groups {
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
    }

    .extension-group {
      border-bottom: 1px solid var(--border);
      margin-bottom: 1rem;

      &:last-child {
        border-bottom: none;
        margin-bottom: 0;
      }
    }

    .group-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.75rem 1rem;
      cursor: pointer;
      border-radius: var(--radius-sm);
      transition: all 0.3s ease;

      .group-info {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex: 1;

        mat-icon {
          font-size: 20px;
          width: 20px;
          height: 20px;
          color: var(--primary);
          transition: transform 0.3s ease;
          display: flex;
          align-items: center;
          justify-content: center;
          animation: gentleFloat 6s ease-in-out infinite;
        }

        span {
          font-size: 0.9rem;
          color: var(--text-primary);
          line-height: 20px;
        }
      }

      &:hover {
        background: rgba(var(--primary-rgb), 0.05);
        transform: translateX(4px);

        .group-info mat-icon {
          transform: scale(1.1);
        }
      }
    }

    .group-content {
      display: none;
      padding: 0.5rem;
      gap: 0.5rem;
      flex-wrap: wrap;

      &.expanded {
        display: flex;
        animation: expandContent 0.3s ease;
      }
    }

    @keyframes expandContent {
      from {
        opacity: 0;
        transform: translateY(-10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .actions-section {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 2rem;
    }

    .get-files-btn {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: all 0.3s ease;
      position: relative;

      &:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);

        mat-icon {
          transform: rotate(180deg);
        }
      }

      mat-icon {
        transition: transform 0.3s ease;
      }
    }

    .current-structure, .optimized-structure {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 1rem;
      margin-top: 1rem;
      min-height: 200px;
      max-height: 500px;
      overflow-y: auto;
    }

    .current-structure h3, .optimized-structure h3 {
      margin: 0 0 1rem 0;
      font-size: 1rem;
      font-weight: 500;
      color: var(--text-primary);
      border-bottom: 2px solid var(--primary);
      padding-bottom: 0.5rem;
    }

    .trees-container {
      display: flex;
      flex-direction: column;
      gap: 2rem;
      margin-top: 2rem;
    }

    .structure-panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 1rem;
      margin-top: 1rem;
      min-height: 200px;
      max-height: 500px;
      overflow-y: auto;
    }

    .structure-panel h3 {
      margin: 0 0 1rem 0;
      font-size: 1rem;
      font-weight: 500;
      color: var(--text-primary);
      border-bottom: 2px solid var(--primary);
      padding-bottom: 0.5rem;
    }

    .update-section {
      display: flex;
      flex-direction: column;
      align-items: center;
      margin-top: 2rem;
      gap: 1rem;
    }

    .update-section button {
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      position: relative;
      overflow: hidden;

      &::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
          90deg,
          transparent,
          rgba(255, 255, 255, 0.2),
          transparent
        );
        transform: translateX(-100%);
      }

      &:hover::after {
        transform: translateX(100%);
        transition: transform 0.8s ease;
      }

      mat-icon {
        animation: gentleFloat 7s ease-in-out infinite;
      }

      &:hover mat-icon {
        animation: smoothRotate 3s linear infinite;
      }
    }

    .messages {
      width: 100%;
      max-width: 500px;
    }

    .success-message, .error-message {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 1rem;
      border-radius: var(--radius-sm);
      margin-top: 0.5rem;
      position: relative;
      overflow: hidden;
      animation: slideIn 0.3s ease;
    }

    @keyframes slideIn {
      from {
        opacity: 0;
        transform: translateX(-20px);
      }
      to {
        opacity: 1;
        transform: translateX(0);
      }
    }

    .success-message {
      background: rgba(46, 125, 50, 0.1);

      mat-icon {
        animation: elegantPulse 3s ease-in-out infinite;
      }

      &::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
          90deg,
          transparent,
          rgba(46, 125, 50, 0.1),
          transparent
        );
        animation: shimmer 2s infinite;
        background-size: 200% 100%;
      }
    }

    .error-message {
      background: rgba(198, 40, 40, 0.1);

      mat-icon {
        animation: gentleFloat 4s ease-in-out infinite;
      }

      &::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
          90deg,
          transparent,
          rgba(211, 47, 47, 0.1),
          transparent
        );
        animation: shimmer 2s infinite;
        background-size: 200% 100%;
      }
    }

    /* Material Overrides */
    ::ng-deep {
      .mat-form-field-appearance-outline .mat-form-field-outline {
        color: var(--border);
      }

      .mat-form-field-appearance-outline.mat-focused .mat-form-field-outline-thick {
        color: var(--primary);
      }

      .mat-form-field-label {
        color: var(--text-secondary);
      }

      .mat-checkbox-checked.mat-primary .mat-checkbox-background {
        background-color: var(--primary);
      }

      .mat-button.mat-primary {
        color: var(--primary);
      }
    }

    /* Theme Variables */
    :root {
      --background: #ffffff;
      --background-rgb: 255, 255, 255;
      --surface: #f9f9f9;
      --surface-variant: #f5f5f5;
      --primary: #00BFA5;
      --primary-rgb: 0, 191, 165; 
      --text-primary: #333333;
      --text-secondary: #666666;
      --border: #e0e0e0;
      --border-rgb: 224, 224, 224; 
      --radius-lg: 12px;
      --radius-sm: 4px;
      --radius-md: 8px; 
      --shadow-sm: 0 1px 2px rgba(0,0,0,0.04); 
      --shadow-md: 0 2px 8px rgba(0, 0, 0, 0.05);
      --shadow-lg: 0 4px 16px rgba(0, 0, 0, 0.1);
      --hover: #f0f0f0;
    }

    :root[data-theme="dark"] {
      --background: #333333;
      --background-rgb: 51, 51, 51;
      --surface: #444444;
      --surface-variant: #555555;
      --primary: #00BFA5;
      --primary-rgb: 0, 191, 165; 
      --text-primary: #ffffff;
      --text-secondary: #cccccc;
      --border: #666666;
      --border-rgb: 102, 102, 102; 
    }

    @media (max-width: 1024px) {
      .content {
        grid-template-columns: 1fr;
      }

      .trees-container {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 768px) {
      .app-container {
        padding: 1rem;
      }

      .header h1 {
        font-size: 2rem;
      }

      .actions-section {
        flex-direction: column;
        gap: 1rem;
      }

      .get-files-btn {
        width: 100%;
      }
    }

    @keyframes gentleFloat {
      0%, 100% {
        transform: translateY(0);
      }
      50% {
        transform: translateY(-2px);
      }
    }
  `]
})
export class AppComponent {

  @ViewChildren(FolderTreeComponent) childComponents!: QueryList<FolderTreeComponent>;

  extensionGroups: ExtensionGroup[] = [
    { name: 'Documents', icon: 'description', extensions: ['.pdf', '.doc', '.docx', '.txt', '.md'], selected: 0, total: 5, expanded: false },
    { name: 'Images', icon: 'image', extensions: ['.jpg', '.jpeg', '.png', '.gif', '.svg'], selected: 0, total: 5, expanded: false },
    { name: 'Audio', icon: 'audiotrack', extensions: ['.mp3', '.wav', '.ogg', '.m4a', '.flac'], selected: 0, total: 5, expanded: false },
    { name: 'Video', icon: 'movie', extensions: ['.mp4', '.avi', '.mkv', '.mov', '.wmv'], selected: 0, total: 5, expanded: false },
    { name: 'Archives', icon: 'folder_zip', extensions: ['.zip', '.rar', '.7z', '.tar', '.gz'], selected: 0, total: 5, expanded: false },
    { name: 'Code', icon: 'code', extensions: ['.js', '.ts', '.py', '.java', '.html', '.css', '.json', '.php', '.cpp'], selected: 0, total: 9, expanded: false },
    { name: 'Data', icon: 'storage', extensions: ['.csv', '.xlsx', '.xml', '.sql', '.db', '.json'], selected: 0, total: 6, expanded: false }
  ];

  original_files: any; 
  srcPaths: any;
  dstPaths: any;
  rootPath: string = "";
  isRecursive: boolean = false;
  successMessage: string = '';
  errorMessage: string = '';
  isLoading: boolean = false;
  filesExts: string[] = [];
  isDarkTheme = false;
  
  selectedLLMProvider: string = 'openai'; 
  ollamaApiBaseUrl: string = 'http://localhost:11434'; 
  llmProviders = [
    {value: 'openai', viewValue: 'OpenAI/Groq API'}, 
    {value: 'ollama', viewValue: 'Ollama (Local)'}
  ];
  customSummarizationPrompt: string = '';

  // Ollama specific properties
  ollamaApiUrlStatus: 'none' | 'loading' | 'success' | 'error' = 'none';
  ollamaApiErrorMsg: string | null = null;
  ollamaModels: string[] = [];
  selectedOllamaTextModel: string = ''; 

  deepAnalysisModeEnabled: boolean = false;
  researchTopicPrompt: string = '';
  quickTopicAnalysisEnabled: boolean = false;
  fullDocTopicAnalysisEnabled: boolean = false;
  semanticSearchEnabled: boolean = false;

  searchQueryText: string = '';
  searchTopN: number = 3; 
  searchFilePathsInput: string = ''; 
  
  searchResults: any[] = [];
  qaAnswer: { answer: string, source_chunks: any[] } | null = null; 

  isLoadingSearch: boolean = false;
  searchError: string | null = null;

  constructor(private dataService: DataService) {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
      this.isDarkTheme = true;
      document.documentElement.setAttribute('data-theme', 'dark');
    }
    this.updateSelectedCounts();
  }

  toggleGroup(group: ExtensionGroup) {
    group.expanded = !group.expanded;
  }

  updateSelectedCounts() {
    this.extensionGroups.forEach(group => {
      group.selected = group.extensions.filter(ext => this.filesExts.includes(ext)).length;
    });
  }

  selectAll() {
    this.filesExts = this.extensionGroups.flatMap(group => group.extensions);
    this.updateSelectedCounts();
  }

  clearAll() {
    this.filesExts = [];
    this.updateSelectedCounts();
  }

  isExtensionSelected(ext: string): boolean {
    return this.filesExts.includes(ext);
  }

  toggleExtension(ext: string, group: ExtensionGroup) {
    const index = this.filesExts.indexOf(ext);
    if (index === -1) {
      this.filesExts.push(ext);
    } else {
      this.filesExts.splice(index, 1);
    }
    group.selected = group.extensions.filter(ext => this.filesExts.includes(ext)).length;
  }

  onPathChange(value: string) {
    this.rootPath = value.replaceAll("\\\\", "/");
  }

  onLlmProviderChange() {
    if (this.selectedLLMProvider === 'ollama') {
      this.checkOllamaUrlAndFetchModels();
    } else {
      this.ollamaApiUrlStatus = 'none'; 
      this.ollamaModels = [];
      this.selectedOllamaTextModel = ''; 
      this.ollamaApiErrorMsg = null;
    }
  }

  checkOllamaUrlAndFetchModels() {
    if (!this.ollamaApiBaseUrl || !this.ollamaApiBaseUrl.trim() || this.selectedLLMProvider !== 'ollama') {
      this.ollamaApiUrlStatus = 'none';
      this.ollamaModels = [];
      this.selectedOllamaTextModel = '';
      this.ollamaApiErrorMsg = null;
      return;
    }

    this.ollamaApiUrlStatus = 'loading';
    this.ollamaApiErrorMsg = null;
    this.ollamaModels = []; 
    this.selectedOllamaTextModel = '';

    this.dataService.getOllamaModels(this.ollamaApiBaseUrl.trim()).subscribe({
      next: (response) => {
        this.ollamaApiUrlStatus = 'success';
        this.ollamaModels = response.models || [];
        if (this.ollamaModels.length > 0) {
           if (!this.selectedOllamaTextModel || !this.ollamaModels.includes(this.selectedOllamaTextModel)) {
             this.selectedOllamaTextModel = this.ollamaModels[0];
           }
        } else {
          this.ollamaApiErrorMsg = 'URL is reachable, but no Ollama models found.';
          this.selectedOllamaTextModel = ''; 
        }
      },
      error: (err) => {
        this.ollamaApiUrlStatus = 'error';
        this.ollamaModels = [];
        this.selectedOllamaTextModel = '';
        if (err.error && typeof err.error.detail === 'string') {
          this.ollamaApiErrorMsg = err.error.detail;
        } else if (err.statusText && typeof err.statusText === 'string' && err.statusText !== 'OK') {
          this.ollamaApiErrorMsg = `Error: ${err.status} - ${err.statusText}`;
        } else if (typeof err.message === 'string') {
          this.ollamaApiErrorMsg = err.message;
        } else {
          this.ollamaApiErrorMsg = 'Failed to connect or fetch models. Check URL and ensure Ollama is running and accessible (CORS might be an issue if Ollama is remote and not configured).';
        }
      }
    });
  }

  getFiles(): void {
    // Reset states
    this.srcPaths = null;
    this.dstPaths = null;
    this.original_files = null;
    this.successMessage = '';
    this.errorMessage = '';
    this.isLoading = true; // Set loading true at the beginning

    let params = new HttpParams();
    params = params.set("root_path", this.rootPath);
    params = params.set("recursive", this.isRecursive.toString()); // Ensure boolean is string for HttpParams
    params = params.set("required_exts", this.filesExts.join(';'));
    params = params.set("llm_provider", this.selectedLLMProvider);

    if (this.selectedLLMProvider === 'ollama') {
      if (this.ollamaApiBaseUrl && this.ollamaApiBaseUrl.trim()) {
        params = params.set("ollama_api_base_url", this.ollamaApiBaseUrl.trim());
      }
      if (this.selectedOllamaTextModel) { 
        params = params.set("ollama_text_model_name", this.selectedOllamaTextModel);
      }
    }

    if (this.customSummarizationPrompt && this.customSummarizationPrompt.trim() !== '') {
      params = params.set("custom_summarization_prompt", this.customSummarizationPrompt);
    }

    if (this.deepAnalysisModeEnabled) {
      if (this.researchTopicPrompt && this.researchTopicPrompt.trim() !== '') {
        params = params.set("research_topic_prompt", this.researchTopicPrompt);
        params = params.set("quick_topic_analysis_enabled", this.quickTopicAnalysisEnabled.toString());
        params = params.set("full_doc_topic_analysis_enabled", this.fullDocTopicAnalysisEnabled.toString());
      } else {
        // Ensure these are explicitly false if no topic prompt, as per backend expectations
        params = params.set("quick_topic_analysis_enabled", "false");
        params = params.set("full_doc_topic_analysis_enabled", "false");
      }
      params = params.set("semantic_search_enabled", this.semanticSearchEnabled.toString());
    } else {
        // Ensure these are explicitly false if deep analysis is off
        params = params.set("quick_topic_analysis_enabled", "false");
        params = params.set("full_doc_topic_analysis_enabled", "false");
        params = params.set("semantic_search_enabled", "false");
    }
    
    // console.log('getFiles params:', params.toString()); // For debugging

    this.dataService.getFormattedFiles(params).subscribe({
      next: (eventData) => {
        // console.log('SSE event received in component:', eventData); // For debugging
        if (eventData.event === 'task_started') {
          this.successMessage = `Task started (ID: ${eventData.data?.task_id || 'N/A'}). Waiting for progress...`;
          this.errorMessage = ''; // Clear previous errors
        } else if (eventData.event === 'progress') {
          if (eventData.data && eventData.data.type === 'status') {
              this.successMessage = eventData.data.message || 'Processing...';
          } else if (eventData.data && eventData.data.type === 'file_processed') { // Example of a more specific progress event
              this.successMessage = `Processed: ${eventData.data.file_path} (${eventData.data.current_file}/${eventData.data.total_files})`;
          }
        } else if (eventData.event === 'task_completed') {
          this.isLoading = false;
          this.successMessage = 'File processing completed successfully!';
          this.errorMessage = '';

          const backendResponseData = eventData.data; 
          let processedItems = backendResponseData.items;

          if (processedItems && Array.isArray(processedItems)) {
            processedItems = processedItems.map((file: any) => {
              let parsedSubTopics = file.sub_topics;
              if (parsedSubTopics && typeof parsedSubTopics === 'string') {
                try {
                  parsedSubTopics = JSON.parse(parsedSubTopics);
                } catch (e) {
                  console.error('Error parsing sub_topics for file:', file.file_path, e);
                }
              }
              if (!Array.isArray(parsedSubTopics)) {
                parsedSubTopics = parsedSubTopics ? [String(parsedSubTopics)] : [];
              }
              return {
                ...file,
                file_path: file.file_path.replaceAll("\\\\", "/"),
                sub_topics: parsedSubTopics,
                dst_path: file.dst_path ? file.dst_path.replaceAll("\\\\", "/") : null
              };
            });
          } else {
            console.warn('No items received in task_completed event or items is not an array:', backendResponseData);
            processedItems = [];
          }
          
          this.original_files = { items: processedItems, root_path: this.rootPath };

          const res = processedItems.map((item: any) => ({
            src_path: `${this.rootPath}/${item.file_path}`, 
            dst_path: item.dst_path ? `${this.rootPath}/${item.dst_path}` : null
          }));
          this.srcPaths = res.map((r: any) => r.src_path);
          this.dstPaths = res.filter((r: any) => r.dst_path !== null).map((r: any) => r.dst_path);

        } else if (eventData.event === 'task_error') {
          this.isLoading = false;
          this.errorMessage = eventData.data?.error || 'An unknown error occurred during processing.';
          this.successMessage = '';
          console.error('Task error from backend:', eventData.data);
        } else if (eventData.event === 'task_cancelled') {
          this.isLoading = false;
          this.errorMessage = eventData.data?.message || 'Task was cancelled.';
          this.successMessage = '';
          console.warn('Task cancelled from backend:', eventData.data);
        }
      },
      error: (err) => {
        this.isLoading = false;
        console.error('Error subscribing to SSE for getFiles:', err);
        this.errorMessage = err.message || 'Failed to connect or process file stream. Check console for details.';
        this.successMessage = '';
        this.original_files = { items: [], root_path: this.rootPath };
        this.srcPaths = null;
        this.dstPaths = null;
      },
      complete: () => {
        if (this.isLoading) { 
            this.isLoading = false;
        }
      }
    });
  }


  updateStructure(): void {
    if (!this.original_files || !this.original_files.items_for_update_structure) { 
        console.error("Data for updateStructure is not available in the expected format.");
        this.errorMessage = "Cannot update structure: required data is missing.";
        return;
    }
    
    this.dataService.updateStructure(this.original_files).subscribe({ 
        next: data => {
            this.successMessage = 'Files re-structured successfully.';
        },
        error: error => {
            console.error(error);
            this.errorMessage = 'An error occurred while moving data.';
        }
    });
  }

  onNotify(value: any): void {
    const index = 1 - value.index; 
    const path = value.path; 
    const root_path = this.original_files.root_path;
    let matchingFilePath = "";
    const itemsForSearch = this.original_files && this.original_files.items ? this.original_files.items : [];

    if (value.index === 0) { 
      const foundItem = itemsForSearch.find((file: any) => root_path + "/" + file.src_path === path);
      if (foundItem && foundItem.dst_path) { 
        matchingFilePath = root_path + "/" + foundItem.dst_path;
      }
    } else { 
      const foundItem = itemsForSearch.find((file: any) => file.dst_path && root_path + "/" + file.dst_path === path);
      if (foundItem) { 
        matchingFilePath = root_path + "/" + foundItem.src_path;
      }
    }
    if(matchingFilePath){ 
        this.childComponents.toArray()[index].highlightFile(matchingFilePath);
    } else {
        console.warn("No matching file path found for highlighting in the other tree for path:", path);
    }
  }

  toggleTheme() {
    this.isDarkTheme = !this.isDarkTheme;
    document.documentElement.setAttribute('data-theme', this.isDarkTheme ? 'dark' : 'light');
    localStorage.setItem('theme', this.isDarkTheme ? 'dark' : 'light');
  }

  private parseFilePaths(): string[] | undefined {
    if (this.searchFilePathsInput && this.searchFilePathsInput.trim() !== '') {
      return this.searchFilePathsInput.split(',').map(fp => fp.trim()).filter(fp => fp !== '');
    }
    return undefined;
  }

  performSemanticSearch(): void {
    if (!this.searchQueryText.trim()) {
      this.searchError = "Please enter a search query.";
      this.searchResults = [];
      this.qaAnswer = null;
      return;
    }
    this.isLoadingSearch = true;
    this.searchError = null;
    this.searchResults = [];
    this.qaAnswer = null;
    
    const filePaths = this.parseFilePaths();

    this.dataService.semanticSearch(this.searchQueryText, this.searchTopN, filePaths)
      .subscribe({
        next: (results) => {
          this.searchResults = results;
          this.isLoadingSearch = false;
          if (results.length === 0) {
            this.searchError = "No relevant chunks found for your query.";
          }
        },
        error: (err) => {
          console.error('Error during semantic search:', err);
          this.searchError = err.error?.detail || err.message || 'Failed to perform semantic search.';
          this.isLoadingSearch = false;
        }
      });
  }

  performQuestionAnswering(): void {
    if (!this.searchQueryText.trim()) {
      this.searchError = "Please enter a question.";
      this.searchResults = [];
      this.qaAnswer = null;
      return;
    }
    this.isLoadingSearch = true;
    this.searchError = null;
    this.searchResults = [];
    this.qaAnswer = null;

    const filePaths = this.parseFilePaths();
    
    this.dataService.answerQuestion(
        this.searchQueryText, 
        this.searchTopN, 
        filePaths, 
        this.selectedLLMProvider, 
        this.ollamaApiBaseUrl.trim(), 
        this.selectedOllamaTextModel 
    ).subscribe({
        next: (response) => {
          this.qaAnswer = response; 
          this.isLoadingSearch = false;
        },
        error: (err) => {
          console.error('Error during question answering:', err);
          this.searchError = err.error?.detail || err.message || 'Failed to get an answer.';
          this.isLoadingSearch = false;
        }
      });
  }
}
