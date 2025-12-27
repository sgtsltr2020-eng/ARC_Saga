import {
    Bell,
    FolderOpen,
    HelpCircle,
    MessageSquare,
    Search,
    Settings,
    Sparkles,
    User
} from 'lucide-react';
import { useState } from 'react';

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  const [hoveredTab, setHoveredTab] = useState<string | null>(null);

  const topTabs = [
    { id: 'chat', icon: MessageSquare, label: 'Chat' },
    { id: 'files', icon: FolderOpen, label: 'Files' },
    { id: 'search', icon: Search, label: 'Search' },
    { id: 'sparks', icon: Sparkles, label: 'Sparks' },
  ];

  const bottomTabs = [
    { id: 'notifications', icon: Bell, label: 'Notifications' },
    { id: 'help', icon: HelpCircle, label: 'Help' },
    { id: 'settings', icon: Settings, label: 'Settings' },
    { id: 'profile', icon: User, label: 'Profile' },
  ];

  const renderTab = (tab: typeof topTabs[0]) => {
    const isActive = activeTab === tab.id;
    const isHovered = hoveredTab === tab.id;
    const Icon = tab.icon;

    return (
      <button
        key={tab.id}
        onClick={() => onTabChange(tab.id)}
        onMouseEnter={() => setHoveredTab(tab.id)}
        onMouseLeave={() => setHoveredTab(null)}
        title={tab.label}
        style={{
          width: '44px',
          height: '44px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: '12px',
          border: 'none',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          backgroundColor: isActive 
            ? 'rgba(99, 102, 241, 0.15)' 
            : isHovered 
              ? 'rgba(255, 255, 255, 0.05)' 
              : 'transparent',
          position: 'relative'
        }}
      >
        <Icon 
          size={22} 
          style={{
            color: isActive ? '#818cf8' : '#71717a',
            transition: 'color 0.2s ease'
          }}
        />
        {isActive && (
          <div style={{
            position: 'absolute',
            left: '-2px',
            top: '50%',
            transform: 'translateY(-50%)',
            width: '4px',
            height: '24px',
            backgroundColor: '#6366f1',
            borderRadius: '0 4px 4px 0'
          }} />
        )}
      </button>
    );
  };

  return (
    <div style={{
      width: '72px',
      height: '100%',
      backgroundColor: '#18181b',
      borderRight: '1px solid rgba(255, 255, 255, 0.06)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '16px 0',
      gap: '8px'
    }}>
      {/* Logo */}
      <div style={{
        width: '40px',
        height: '40px',
        borderRadius: '10px',
        background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: '16px',
        boxShadow: '0 4px 12px rgba(99, 102, 241, 0.3)'
      }}>
        <span style={{ 
          color: 'white', 
          fontSize: '18px', 
          fontWeight: 700 
        }}>S</span>
      </div>

      {/* Top tabs */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
        flex: 1
      }}>
        {topTabs.map(renderTab)}
      </div>

      {/* Bottom tabs */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '4px'
      }}>
        {bottomTabs.map(renderTab)}
      </div>
    </div>
  );
}
