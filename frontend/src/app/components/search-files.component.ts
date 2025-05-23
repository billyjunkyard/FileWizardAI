import { Component, Input } from '@angular/core';
import { HttpParams } from "@angular/common/http";
import { DataService } from "../data.service";

@Component({
  selector: 'app-search-files',
  template: `
    <mat-form-field class="example-full-width">
      <mat-label>Search text</mat-label>
      <textarea matInput placeholder="Search text" [(ngModel)]="searchQuery"></textarea>
    </mat-form-field>

    <div class="llm-config-section">
      <mat-form-field appearance="outline" class="llm-provider-field">
        <mat-label>LLM Provider</mat-label>
        <mat-select [(ngModel)]="selectedLLMProvider">
          <mat-option *ngFor="let provider of llmProviders" [value]="provider.value">
            {{provider.viewValue}}
          </mat-option>
        </mat-select>
      </mat-form-field>

      <mat-form-field appearance="outline" class="ollama-url-field" *ngIf="selectedLLMProvider === 'ollama'">
        <mat-label>Ollama API Base URL</mat-label>
        <input matInput [(ngModel)]="ollamaApiBaseUrl" placeholder="e.g., http://localhost:11434/v1">
      </mat-form-field>

      <mat-form-field appearance="outline" class="custom-prompt-field">
        <mat-label>Custom Summarization Prompt (Optional)</mat-label>
        <textarea matInput
                  [(ngModel)]="customSummarizationPrompt"
                  rows="3"
                  placeholder="Enter your custom prompt for summarizing documents. If left empty, a default prompt will be used."></textarea>
      </mat-form-field>
    </div>

    <button mat-raised-button (click)="searchFiles()" style="margin-left: 1%">Search</button>
    <div class="spinner-container">
        <mat-spinner *ngIf="isLoading"></mat-spinner>
    </div>
    <app-folder-tree title="Searched files" *ngIf="files" [paths]="files" [rootPath]="rootPath" [index]=2></app-folder-tree>
  `,
  styles: [`
    .example-full-width {
      width: 100%;
    }
    .llm-config-section {
      margin-top: 1rem;
      margin-bottom: 1rem;
    }
    .llm-provider-field, .ollama-url-field, .custom-prompt-field {
      width: 100%;
      margin-bottom: 1rem;
    }
    .custom-prompt-field textarea {
      color: var(--text-primary); /* Ensure text color matches theme */
    }
    .spinner-container {
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100%; /* or a specific height if needed */
    }
  `]
})
export class SearchFilesComponent {
  searchQuery: string = "";
  files: any;
  isLoading: boolean = false;
  @Input() rootPath: string = "";
  @Input() isRecursive: boolean = false;
  @Input() filesExts: string[] = [];

  selectedLLMProvider: string = 'openai';
  ollamaApiBaseUrl: string = 'http://localhost:11434/v1';
  llmProviders = [{value: 'openai', viewValue: 'OpenAI/Groq API'}, {value: 'ollama', viewValue: 'Ollama (Local)'}];
  customSummarizationPrompt: string = '';

  constructor(private dataService: DataService) {
  }

  searchFiles(): void {
    this.files = null;
    this.isLoading = true;
    let params = new HttpParams();
    params = params.set("root_path", this.rootPath);
    params = params.set("recursive", this.isRecursive);
    params = params.set("required_exts", this.filesExts.join(';'));
    params = params.set("search_query", this.searchQuery);
    params = params.set("llm_provider", this.selectedLLMProvider);
    if (this.selectedLLMProvider === 'ollama') {
      params = params.set("ollama_api_base_url", this.ollamaApiBaseUrl);
    }
    if (this.customSummarizationPrompt && this.customSummarizationPrompt.trim() !== '') {
      params = params.set("custom_summarization_prompt", this.customSummarizationPrompt);
    }
    this.dataService
      .getSearchFiles(params)
      .subscribe((data: any) => {
        if (!this.rootPath.endsWith('/')) this.rootPath += '/';
        this.files = data.map((item: any) => this.rootPath + item.file);
        this.isLoading = false;
      })
  }
}
