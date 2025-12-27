import {
    Bot,
    Check,
    Code2,
    Copy,
    Loader2,
    Paperclip,
    RotateCcw,
    Send,
    Sparkles,
    User
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface ChatAreaProps {
  messages: Message[];
  onSendMessage: (content: string) => Promise<void>;
  sending?: boolean;
  threadId?: string | null;
}

export function ChatArea({ messages, onSendMessage, sending = false, threadId }: ChatAreaProps) {
  const [input, setInput] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || sending) return;

    const content = input;
    setInput('');
    await onSendMessage(content);
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const threadTitle = threadId 
    ? `Thread ${threadId.slice(0, 8)}...` 
    : 'New Conversation';

  return (
    <div style={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: '#09090b'
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 24px',
        borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        backdropFilter: 'blur(12px)',
        backgroundColor: 'rgba(9, 9, 11, 0.8)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%)',
            border: '1px solid rgba(99, 102, 241, 0.2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Sparkles size={18} color="#818cf8" />
          </div>
          <div>
            <h1 style={{ 
              fontSize: '15px', 
              fontWeight: 600, 
              color: '#fafafa', 
              margin: 0,
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              Saga Assistant
              <span style={{
                fontSize: '10px',
                fontWeight: 500,
                color: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                padding: '2px 8px',
                borderRadius: '4px',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>Pro</span>
            </h1>
            <p style={{ fontSize: '12px', color: '#71717a', margin: 0, marginTop: '2px' }}>
              {threadTitle}
            </p>
          </div>
        </div>
        <button 
          style={{
            padding: '8px 12px',
            borderRadius: '8px',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            background: 'rgba(255, 255, 255, 0.03)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            transition: 'all 0.15s ease'
          }}
          title="Regenerate"
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.06)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)';
          }}
        >
          <RotateCcw size={14} color="#71717a" />
          <span style={{ fontSize: '13px', color: '#a1a1aa' }}>Regenerate</span>
        </button>
      </div>

      {/* Messages */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '24px 32px'
      }}>
        {messages.length === 0 ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: '#52525b',
            textAlign: 'center'
          }}>
            <div style={{
              width: '64px',
              height: '64px',
              borderRadius: '16px',
              background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
              border: '1px solid rgba(99, 102, 241, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              marginBottom: '16px'
            }}>
              <Sparkles size={28} color="#6366f1" />
            </div>
            <h2 style={{ 
              fontSize: '18px', 
              fontWeight: 600, 
              color: '#e4e4e7', 
              margin: 0,
              marginBottom: '8px'
            }}>
              How can I help you today?
            </h2>
            <p style={{ 
              fontSize: '14px', 
              color: '#71717a', 
              margin: 0,
              maxWidth: '400px'
            }}>
              Ask me anything about your codebase, debugging, architecture, or any programming questions.
            </p>
          </div>
        ) : (
          messages.map((message) => (
            <div 
              key={message.id} 
              style={{ marginBottom: '28px' }}
              onMouseEnter={(e) => {
                const actions = e.currentTarget.querySelector('.message-actions') as HTMLElement;
                if (actions) actions.style.opacity = '1';
              }}
              onMouseLeave={(e) => {
                const actions = e.currentTarget.querySelector('.message-actions') as HTMLElement;
                if (actions) actions.style.opacity = '0';
              }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
                {/* Avatar */}
                <div style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: '10px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                  background: message.role === 'user' 
                    ? 'rgba(255, 255, 255, 0.05)' 
                    : 'linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)',
                  border: message.role === 'user'
                    ? '1px solid rgba(255, 255, 255, 0.06)'
                    : '1px solid rgba(99, 102, 241, 0.2)'
                }}>
                  {message.role === 'user' 
                    ? <User size={16} color="#a1a1aa" />
                    : <Bot size={16} color="#818cf8" />
                  }
                </div>

                {/* Content */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                    <span style={{ 
                      fontSize: '14px', 
                      fontWeight: 600, 
                      color: message.role === 'user' ? '#e4e4e7' : '#fafafa' 
                    }}>
                      {message.role === 'user' ? 'You' : 'Saga'}
                    </span>
                    <span style={{ fontSize: '12px', color: '#52525b' }}>
                      {formatTime(message.timestamp)}
                    </span>
                  </div>
                  <div style={{ 
                    fontSize: '14px', 
                    color: '#d4d4d8', 
                    whiteSpace: 'pre-wrap',
                    lineHeight: 1.7
                  }}>
                    {renderContent(message.content)}
                  </div>
                </div>

                {/* Actions */}
                <div 
                  className="message-actions"
                  style={{ 
                    opacity: 0, 
                    transition: 'opacity 0.15s',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}
                >
                  <button 
                    onClick={() => copyToClipboard(message.content, message.id)}
                    style={{
                      padding: '8px',
                      borderRadius: '8px',
                      border: '1px solid rgba(255, 255, 255, 0.06)',
                      background: 'rgba(255, 255, 255, 0.03)',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      transition: 'all 0.15s ease'
                    }}
                    title="Copy"
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.06)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)';
                    }}
                  >
                    {copiedId === message.id 
                      ? <Check size={14} color="#22c55e" />
                      : <Copy size={14} color="#71717a" />
                    }
                  </button>
                </div>
              </div>
            </div>
          ))
        )}

        {sending && (
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '16px' }}>
            <div style={{
              width: '36px',
              height: '36px',
              borderRadius: '10px',
              background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%)',
              border: '1px solid rgba(99, 102, 241, 0.2)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Bot size={16} color="#818cf8" />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', paddingTop: '8px' }}>
              <Loader2 size={16} color="#6366f1" style={{ animation: 'spin 1s linear infinite' }} />
              <span style={{ fontSize: '13px', color: '#71717a' }}>Thinking...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div style={{
        padding: '20px 32px 24px',
        borderTop: '1px solid rgba(255, 255, 255, 0.06)',
        backgroundColor: 'rgba(9, 9, 11, 0.95)'
      }}>
        <form onSubmit={handleSubmit}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'flex-end', 
            gap: '12px',
            padding: '12px 16px',
            backgroundColor: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            transition: 'all 0.2s ease'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <button 
                type="button"
                style={{
                  padding: '8px',
                  borderRadius: '8px',
                  border: 'none',
                  background: 'transparent',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'background 0.15s'
                }}
                title="Attach file"
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                }}
              >
                <Paperclip size={18} color="#71717a" />
              </button>
              <button 
                type="button"
                style={{
                  padding: '8px',
                  borderRadius: '8px',
                  border: 'none',
                  background: 'transparent',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'background 0.15s'
                }}
                title="Insert code"
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                }}
              >
                <Code2 size={18} color="#71717a" />
              </button>
            </div>

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Ask Saga anything..."
              rows={1}
              disabled={sending}
              style={{
                flex: 1,
                padding: '8px 0',
                backgroundColor: 'transparent',
                border: 'none',
                fontSize: '14px',
                color: '#fafafa',
                resize: 'none',
                outline: 'none',
                fontFamily: 'inherit',
                lineHeight: 1.5,
                opacity: sending ? 0.5 : 1
              }}
            />

            <button
              type="submit"
              disabled={!input.trim() || sending}
              style={{
                width: '40px',
                height: '40px',
                borderRadius: '12px',
                border: 'none',
                background: !input.trim() || sending 
                  ? 'rgba(99, 102, 241, 0.3)' 
                  : 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                cursor: !input.trim() || sending ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: !input.trim() || sending 
                  ? 'none' 
                  : '0 4px 12px rgba(99, 102, 241, 0.3)'
              }}
            >
              {sending ? (
                <Loader2 size={18} color="white" style={{ animation: 'spin 1s linear infinite' }} />
              ) : (
                <Send size={18} color="white" />
              )}
            </button>
          </div>
        </form>
        <p style={{
          fontSize: '11px',
          color: '#52525b',
          textAlign: 'center',
          marginTop: '12px'
        }}>
          Saga may make mistakes. Verify important information.
        </p>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString('en-US', { 
    hour: 'numeric', 
    minute: '2-digit',
    hour12: true 
  });
}

function renderContent(content: string): React.ReactNode {
  const parts = content.split(/(```[\s\S]*?```)/g);
  
  return parts.map((part, index) => {
    if (part.startsWith('```') && part.endsWith('```')) {
      const lines = part.slice(3, -3).split('\n');
      const language = lines[0] || 'text';
      const code = lines.slice(1).join('\n');
      
      return (
        <pre 
          key={index} 
          style={{
            margin: '16px 0',
            padding: '16px 20px',
            backgroundColor: 'rgba(0, 0, 0, 0.3)',
            borderRadius: '12px',
            border: '1px solid rgba(255, 255, 255, 0.06)',
            overflowX: 'auto'
          }}
        >
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '12px',
            paddingBottom: '12px',
            borderBottom: '1px solid rgba(255, 255, 255, 0.06)'
          }}>
            <span style={{ 
              fontSize: '11px', 
              color: '#71717a',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              fontWeight: 500
            }}>{language}</span>
            <button
              style={{
                padding: '4px 8px',
                borderRadius: '6px',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                background: 'rgba(255, 255, 255, 0.03)',
                fontSize: '11px',
                color: '#a1a1aa',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}
              onClick={() => navigator.clipboard.writeText(code)}
            >
              <Copy size={12} />
              Copy
            </button>
          </div>
          <code style={{ 
            fontSize: '13px', 
            fontFamily: "'JetBrains Mono', 'Fira Code', monospace", 
            color: '#e4e4e7',
            lineHeight: 1.6
          }}>
            {code}
          </code>
        </pre>
      );
    }
    return <span key={index}>{part}</span>;
  });
}
