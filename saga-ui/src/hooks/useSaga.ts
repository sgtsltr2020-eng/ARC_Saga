/**
 * React hooks for Saga API integration
 */

import { useCallback, useEffect, useState } from 'react';
import { sagaApi } from '../services/sagaApi';
import type { Message, Thread } from '../types';

/**
 * Hook for managing threads
 */
export function useThreads() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchThreads = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await sagaApi.listThreads();
      setThreads(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch threads');
      // Keep showing mock data if API fails
      setThreads([
        {
          id: '1',
          title: 'Phase 8: MAE Foundation',
          preview: 'Implementing FQL Gateway and Governor...',
          messageCount: 42,
          updatedAt: new Date(),
          source: 'saga-ui'
        },
        {
          id: '2',
          title: 'Synthesis Engine Refactor',
          preview: 'Adding dual-citation requirement...',
          messageCount: 18,
          updatedAt: new Date(Date.now() - 3600000),
          source: 'saga-ui'
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchThreads();
  }, [fetchThreads]);

  return { threads, loading, error, refetch: fetchThreads };
}

/**
 * Hook for managing chat messages in a thread
 */
export function useChat(threadId: string | null) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch thread messages
  const fetchMessages = useCallback(async () => {
    if (!threadId) {
      setMessages([]);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const data = await sagaApi.getThread(threadId);
      setMessages(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch messages');
      // Show mock data if API fails
      setMessages([
        {
          id: '1',
          role: 'user',
          content: 'Can you help me implement the FQL Gateway?',
          timestamp: new Date(Date.now() - 120000),
        },
        {
          id: '2',
          role: 'assistant',
          content: `I'll help you implement the FQL Gateway. Here's the plan:

1. **Create FQL Schema** - Define Pydantic models for FQL packets
2. **Implement Validation** - Add validate_proposal() method to Mimiry
3. **Wire to Warden** - Make Warden enforce FQL protocol

\`\`\`python
from pydantic import BaseModel
from enum import Enum

class FQLAction(str, Enum):
    VALIDATE_PATTERN = "VALIDATE_PATTERN"
    PROPOSE_REFACTOR = "PROPOSE_REFACTOR"
\`\`\`

Would you like me to continue?`,
          timestamp: new Date(Date.now() - 60000),
        },
      ]);
    } finally {
      setLoading(false);
    }
  }, [threadId]);

  useEffect(() => {
    fetchMessages();
  }, [fetchMessages]);

  // Send a new message
  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    // Optimistically add user message
    setMessages(prev => [...prev, userMessage]);
    setSending(true);

    try {
      // Capture user message
      const result = await sagaApi.captureMessage({
        source: 'saga-ui',
        role: 'user',
        content,
        thread_id: threadId || undefined,
      });

      // Get AI response via Perplexity
      const stream = await sagaApi.askPerplexity({
        query: content,
        thread_id: result.thread_id,
        inject_context: true,
      });

      if (stream) {
        // Handle streaming response
        const reader = stream.getReader();
        const decoder = new TextDecoder();
        let assistantContent = '';

        const assistantMessage: Message = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: '',
          timestamp: new Date(),
        };

        setMessages(prev => [...prev, assistantMessage]);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          assistantContent += chunk;

          setMessages(prev => 
            prev.map(m => 
              m.id === assistantMessage.id 
                ? { ...m, content: assistantContent }
                : m
            )
          );
        }

        // Capture assistant response
        await sagaApi.captureMessage({
          source: 'saga-ui',
          role: 'assistant',
          content: assistantContent,
          thread_id: result.thread_id,
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
      
      // Add fallback response if API fails
      const fallbackMessage: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: 'I apologize, but I encountered an error connecting to the backend. Please ensure the Saga server is running on port 8421.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, fallbackMessage]);
    } finally {
      setSending(false);
    }
  }, [threadId]);

  return { 
    messages, 
    loading, 
    sending, 
    error, 
    sendMessage, 
    refetch: fetchMessages 
  };
}

/**
 * Hook for checking backend health
 */
export function useBackendHealth() {
  const [isConnected, setIsConnected] = useState(false);
  const [checking, setChecking] = useState(true);

  const checkHealth = useCallback(async () => {
    try {
      setChecking(true);
      await sagaApi.healthCheck();
      setIsConnected(true);
    } catch {
      setIsConnected(false);
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  return { isConnected, checking, checkHealth };
}

/**
 * Hook for searching memory
 */
export function useSearch() {
  const [results, setResults] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const search = useCallback(async (query: string) => {
    if (!query.trim()) {
      setResults([]);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const data = await sagaApi.searchMemory(query);
      setResults(data.messages);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  }, []);

  return { results, loading, error, search };
}
