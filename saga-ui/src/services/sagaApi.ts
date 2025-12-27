/**
 * Saga API Client
 * Connects the UI to the Saga FastAPI backend
 */

import type { CaptureRequest, Message, PerplexityRequest, SearchResult, Thread } from '../types';

const API_BASE = 'http://localhost:8421';

class SagaApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  /**
   * Check if the Saga backend is available
   */
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await fetch(`${this.baseUrl}/health`);
    if (!response.ok) {
      throw new Error('Saga backend is not available');
    }
    return response.json();
  }

  /**
   * List all conversation threads
   */
  async listThreads(limit: number = 50): Promise<Thread[]> {
    const response = await fetch(`${this.baseUrl}/threads?limit=${limit}`);
    if (!response.ok) {
      throw new Error('Failed to fetch threads');
    }
    const data = await response.json();
    
    // Transform API response to Thread interface
    return (data.threads || []).map((t: Record<string, unknown>) => ({
      id: t.thread_id || t.id,
      title: t.title || `Thread ${String(t.thread_id || '').slice(0, 8)}`,
      preview: t.first_message || t.last_message || t.preview || '',
      messageCount: t.message_count || 0,
      updatedAt: new Date(String(t.updated_at || t.last_updated || Date.now())),
      source: t.source || 'saga-ui'
    }));
  }

  /**
   * Get messages for a specific thread
   */
  async getThread(threadId: string): Promise<Message[]> {
    const response = await fetch(`${this.baseUrl}/thread/${threadId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch thread messages');
    }
    const data = await response.json();
    
    return (data.messages || []).map((m: Record<string, unknown>) => ({
      id: m.id || crypto.randomUUID(),
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: m.content,
      timestamp: new Date(String(m.timestamp || Date.now())),
      source: m.source,
      metadata: m.metadata
    }));
  }

  /**
   * Capture/store a new message
   */
  async captureMessage(request: CaptureRequest): Promise<{ message_id: string; thread_id: string }> {
    const response = await fetch(`${this.baseUrl}/capture`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    
    if (!response.ok) {
      throw new Error('Failed to capture message');
    }
    
    return response.json();
  }

  /**
   * Ask Perplexity with automatic context injection
   */
  async askPerplexity(request: PerplexityRequest): Promise<ReadableStream<Uint8Array> | null> {
    const response = await fetch(`${this.baseUrl}/perplexity/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    
    if (!response.ok) {
      throw new Error('Failed to get response from Perplexity');
    }
    
    return response.body;
  }

  /**
   * Search across all conversations
   */
  async searchMemory(query: string, options?: {
    searchType?: 'keyword' | 'semantic';
    sources?: string[];
    limit?: number;
  }): Promise<SearchResult> {
    const response = await fetch(`${this.baseUrl}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        search_type: options?.searchType || 'keyword',
        sources: options?.sources,
        limit: options?.limit || 20,
      }),
    });
    
    if (!response.ok) {
      throw new Error('Search failed');
    }
    
    const data = await response.json();
    return {
      messages: (data.results || []).map((m: Record<string, unknown>) => ({
        id: m.id || crypto.randomUUID(),
        role: m.role === 'assistant' ? 'assistant' : 'user',
        content: m.content,
        timestamp: new Date(String(m.timestamp || Date.now())),
        source: m.source,
      })),
      total: data.total || data.results?.length || 0,
    };
  }

  /**
   * Get recent context across all sources
   */
  async getRecentContext(limit: number = 10, sources?: string[]): Promise<Message[]> {
    const params = new URLSearchParams({ limit: String(limit) });
    if (sources) {
      params.set('sources', sources.join(','));
    }
    
    const response = await fetch(`${this.baseUrl}/context/recent?${params}`);
    if (!response.ok) {
      throw new Error('Failed to fetch context');
    }
    
    const data = await response.json();
    return (data.messages || []).map((m: Record<string, unknown>) => ({
      id: m.id || crypto.randomUUID(),
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: m.content,
      timestamp: new Date(String(m.timestamp || Date.now())),
      source: m.source,
    }));
  }
}

// Export singleton instance
export const sagaApi = new SagaApiClient();

// Export class for custom instances
export { SagaApiClient };

