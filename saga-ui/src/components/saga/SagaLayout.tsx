import { Wifi, WifiOff } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useBackendHealth, useChat, useThreads } from '../../hooks/useSaga';
import { ChatArea } from './ChatArea';
import { Sidebar } from './Sidebar';
import { ThreadList } from './ThreadList';

export function SagaLayout() {
  const [activeTab, setActiveTab] = useState('chat');
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [threadPanelWidth, setThreadPanelWidth] = useState(280);
  const [isResizing, setIsResizing] = useState(false);

  // API hooks
  const { threads, loading: threadsLoading, refetch: refetchThreads } = useThreads();
  const { messages, sending, sendMessage, refetch: refetchMessages } = useChat(activeThreadId);
  const { isConnected, checking } = useBackendHealth();

  // Set first thread as active when threads load
  useEffect(() => {
    if (threads.length > 0 && !activeThreadId) {
      setActiveThreadId(threads[0].id);
    }
  }, [threads, activeThreadId]);

  const handleMouseDown = () => {
    setIsResizing(true);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isResizing) return;
    const newWidth = Math.max(220, Math.min(400, e.clientX - 72));
    setThreadPanelWidth(newWidth);
  };

  const handleMouseUp = () => {
    setIsResizing(false);
  };

  const handleThreadSelect = (threadId: string) => {
    setActiveThreadId(threadId);
  };

  const handleSendMessage = async (content: string) => {
    await sendMessage(content);
    refetchThreads(); // Refresh thread list after sending
  };

  const handleNewThread = () => {
    setActiveThreadId(null);
    // Clearing activeThreadId will create a new thread on first message
  };

  return (
    <div 
      style={{
        display: 'flex',
        height: '100%',
        width: '100%',
        backgroundColor: '#09090b',
        position: 'relative'
      }}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Connection Status Indicator */}
      <div style={{
        position: 'absolute',
        top: '12px',
        right: '12px',
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '6px 12px',
        borderRadius: '8px',
        backgroundColor: isConnected 
          ? 'rgba(34, 197, 94, 0.1)' 
          : 'rgba(239, 68, 68, 0.1)',
        border: `1px solid ${isConnected 
          ? 'rgba(34, 197, 94, 0.2)' 
          : 'rgba(239, 68, 68, 0.2)'}`,
        transition: 'all 0.3s ease'
      }}>
        {checking ? (
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: '#f59e0b',
            animation: 'pulse 1s infinite'
          }} />
        ) : isConnected ? (
          <Wifi size={14} color="#22c55e" />
        ) : (
          <WifiOff size={14} color="#ef4444" />
        )}
        <span style={{
          fontSize: '11px',
          fontWeight: 500,
          color: isConnected ? '#22c55e' : '#ef4444'
        }}>
          {checking ? 'Connecting...' : isConnected ? 'Connected' : 'Offline'}
        </span>
      </div>

      {/* Icon Sidebar - 72px fixed */}
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Thread List Panel */}
      <div style={{
        width: `${threadPanelWidth}px`,
        flexShrink: 0,
        height: '100%',
        overflow: 'hidden',
        borderRight: '1px solid rgba(255, 255, 255, 0.06)'
      }}>
        <ThreadList 
          threads={threads}
          activeId={activeThreadId}
          onSelect={handleThreadSelect}
          onNewThread={handleNewThread}
          loading={threadsLoading}
        />
      </div>

      {/* Resize Handle */}
      <div 
        onMouseDown={handleMouseDown}
        style={{
          width: '6px',
          marginLeft: '-3px',
          marginRight: '-3px',
          cursor: 'col-resize',
          backgroundColor: isResizing ? '#6366f1' : 'transparent',
          transition: isResizing ? 'none' : 'background-color 0.15s',
          flexShrink: 0,
          zIndex: 10,
          position: 'relative'
        }}
        onMouseEnter={(e) => {
          if (!isResizing) {
            (e.currentTarget.querySelector('.handle-line') as HTMLElement)?.style.setProperty('opacity', '1');
          }
        }}
        onMouseLeave={(e) => {
          if (!isResizing) {
            (e.currentTarget.querySelector('.handle-line') as HTMLElement)?.style.setProperty('opacity', '0');
          }
        }}
      >
        <div 
          className="handle-line"
          style={{
            position: 'absolute',
            left: '50%',
            top: '50%',
            transform: 'translate(-50%, -50%)',
            width: '2px',
            height: '40px',
            backgroundColor: '#6366f1',
            borderRadius: '2px',
            opacity: isResizing ? 1 : 0,
            transition: 'opacity 0.15s'
          }}
        />
      </div>

      {/* Main Chat Area */}
      <div style={{ 
        flex: 1, 
        minWidth: 0,
        height: '100%',
        overflow: 'hidden'
      }}>
        <ChatArea 
          messages={messages}
          onSendMessage={handleSendMessage}
          sending={sending}
          threadId={activeThreadId}
        />
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}

export default SagaLayout;
