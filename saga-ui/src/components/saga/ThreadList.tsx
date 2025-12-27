import {
    Loader2,
    MessageSquare,
    MoreHorizontal,
    Plus,
    Search
} from 'lucide-react';
import { useState } from 'react';

interface Thread {
  id: string;
  title: string;
  preview: string;
  messageCount: number;
  updatedAt: Date;
  source?: string;
}

interface ThreadListProps {
  threads: Thread[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNewThread: () => void;
  loading?: boolean;
}

export function ThreadList({ threads, activeId, onSelect, onNewThread, loading }: ThreadListProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  const filteredThreads = threads.filter(t => 
    t.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div style={{
      height: '100%',
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: '#0f0f12'
    }}>
      {/* Header */}
      <div style={{
        padding: '20px 16px 16px',
        borderBottom: '1px solid rgba(255, 255, 255, 0.06)'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '16px'
        }}>
          <h2 style={{
            fontSize: '13px',
            fontWeight: 600,
            color: '#a1a1aa',
            margin: 0,
            textTransform: 'uppercase',
            letterSpacing: '0.5px'
          }}>Threads</h2>
          <button 
            onClick={onNewThread}
            style={{
              width: '28px',
              height: '28px',
              borderRadius: '8px',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              background: 'rgba(255, 255, 255, 0.03)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.15s ease'
            }}
            title="New Thread"
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(99, 102, 241, 0.15)';
              e.currentTarget.style.borderColor = 'rgba(99, 102, 241, 0.3)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)';
              e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.08)';
            }}
          >
            <Plus size={14} color="#a1a1aa" />
          </button>
        </div>
        
        {/* Search */}
        <div style={{ position: 'relative' }}>
          <Search 
            size={14} 
            style={{
              position: 'absolute',
              left: '12px',
              top: '50%',
              transform: 'translateY(-50%)',
              color: '#52525b'
            }} 
          />
          <input
            type="text"
            placeholder="Search threads..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              width: '100%',
              paddingLeft: '36px',
              paddingRight: '12px',
              paddingTop: '10px',
              paddingBottom: '10px',
              fontSize: '13px',
              backgroundColor: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.06)',
              borderRadius: '10px',
              color: '#fafafa',
              outline: 'none',
              transition: 'all 0.15s ease'
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = 'rgba(99, 102, 241, 0.5)';
              e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.06)';
              e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.03)';
            }}
          />
        </div>
      </div>

      {/* Thread List */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '8px'
      }}>
        {loading ? (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100px',
            color: '#71717a'
          }}>
            <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} />
          </div>
        ) : filteredThreads.length === 0 ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '150px',
            color: '#52525b',
            textAlign: 'center',
            padding: '0 20px'
          }}>
            <MessageSquare size={32} style={{ marginBottom: '12px', opacity: 0.5 }} />
            <p style={{ fontSize: '13px', margin: 0 }}>
              {searchQuery ? 'No threads match your search' : 'No threads yet'}
            </p>
            <button
              onClick={onNewThread}
              style={{
                marginTop: '12px',
                padding: '8px 16px',
                fontSize: '12px',
                color: '#818cf8',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                border: '1px solid rgba(99, 102, 241, 0.2)',
                borderRadius: '8px',
                cursor: 'pointer'
              }}
            >
              Start a conversation
            </button>
          </div>
        ) : (
          filteredThreads.map((thread) => (
            <div
              key={thread.id}
              onClick={() => onSelect(thread.id)}
              onMouseEnter={() => setHoveredId(thread.id)}
              onMouseLeave={() => setHoveredId(null)}
              style={{
                padding: '14px 12px',
                marginBottom: '4px',
                borderRadius: '12px',
                cursor: 'pointer',
                backgroundColor: activeId === thread.id 
                  ? 'rgba(99, 102, 241, 0.12)' 
                  : hoveredId === thread.id 
                    ? 'rgba(255, 255, 255, 0.03)' 
                    : 'transparent',
                border: activeId === thread.id 
                  ? '1px solid rgba(99, 102, 241, 0.2)' 
                  : '1px solid transparent',
                transition: 'all 0.15s ease',
                position: 'relative'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                <div style={{
                  width: '36px',
                  height: '36px',
                  borderRadius: '10px',
                  backgroundColor: activeId === thread.id 
                    ? 'rgba(99, 102, 241, 0.2)' 
                    : 'rgba(255, 255, 255, 0.04)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                  transition: 'all 0.15s ease'
                }}>
                  <MessageSquare 
                    size={16} 
                    color={activeId === thread.id ? '#818cf8' : '#71717a'} 
                  />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'space-between',
                    marginBottom: '4px'
                  }}>
                    <h3 style={{
                      fontSize: '14px',
                      fontWeight: 500,
                      color: activeId === thread.id ? '#fafafa' : '#e4e4e7',
                      margin: 0,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      flex: 1
                    }}>
                      {thread.title}
                    </h3>
                    {hoveredId === thread.id && (
                      <button 
                        onClick={(e) => e.stopPropagation()}
                        style={{ 
                          padding: '4px', 
                          background: 'none', 
                          border: 'none', 
                          cursor: 'pointer',
                          borderRadius: '4px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                      >
                        <MoreHorizontal size={14} color="#71717a" />
                      </button>
                    )}
                  </div>
                  <p style={{
                    fontSize: '12px',
                    color: '#71717a',
                    margin: 0,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}>
                    {thread.preview}
                  </p>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    marginTop: '8px',
                    fontSize: '11px',
                    color: '#52525b'
                  }}>
                    <span>{thread.messageCount} messages</span>
                    <span style={{ color: '#3f3f46' }}>•</span>
                    <span>{formatRelativeTime(thread.updatedAt)}</span>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
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

function formatRelativeTime(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return 'Just now';
  if (minutes < 60) return `${minutes}m ago`;
  if (hours < 24) return `${hours}h ago`;
  return `${days}d ago`;
}
