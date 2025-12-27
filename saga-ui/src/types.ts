/**
 * Shared types for Saga UI
 */

export interface Thread {
  id: string;
  title: string;
  preview: string;
  messageCount: number;
  updatedAt: Date;
  source: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  source?: string;
  metadata?: Record<string, unknown>;
}

export interface CaptureRequest {
  source: string;
  role: string;
  content: string;
  thread_id?: string;
  metadata?: Record<string, unknown>;
}

export interface SearchResult {
  messages: Message[];
  total: number;
}

export interface PerplexityRequest {
  query: string;
  thread_id?: string;
  inject_context?: boolean;
}
