import { useState } from 'react';
import { ChatArea } from './ChatArea';
import { ThreadList } from './ThreadList';

export function ChatWorkspace() {
  const [activeThreadId, setActiveThreadId] = useState('1');
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const [isResizing, setIsResizing] = useState(false);

  const handleMouseDown = () => {
    setIsResizing(true);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isResizing) return;
    const newWidth = Math.max(200, Math.min(400, e.clientX));
    setSidebarWidth(newWidth);
  };

  const handleMouseUp = () => {
    setIsResizing(false);
  };

  return (
    <div 
      style={{
        display: 'flex',
        height: '100%',
        width: '100%'
      }}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Left Thread List Panel */}
      <div style={{ 
        width: `${sidebarWidth}px`, 
        flexShrink: 0,
        height: '100%',
        overflow: 'hidden'
      }}>
        <ThreadList 
          activeId={activeThreadId} 
          onSelect={setActiveThreadId} 
        />
      </div>

      {/* Resize Handle */}
      <div 
        onMouseDown={handleMouseDown}
        style={{
          width: '4px',
          cursor: 'col-resize',
          backgroundColor: isResizing ? 'var(--saga-accent)' : 'var(--saga-border)',
          transition: isResizing ? 'none' : 'background-color 0.15s',
          flexShrink: 0
        }}
        onMouseEnter={(e) => {
          if (!isResizing) e.currentTarget.style.backgroundColor = 'var(--saga-accent)';
        }}
        onMouseLeave={(e) => {
          if (!isResizing) e.currentTarget.style.backgroundColor = 'var(--saga-border)';
        }}
      />

      {/* Main Chat Area */}
      <div style={{ 
        flex: 1, 
        minWidth: 0,
        height: '100%',
        overflow: 'hidden'
      }}>
        <ChatArea />
      </div>
    </div>
  );
}
