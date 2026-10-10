
import React, { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  LiveKitRoom,
  RoomAudioRenderer,
  ControlBar,
  useVoiceAssistant,
  useConnectionState,
  useRoomContext,
  useLocalParticipant,
  useParticipants,
} from '@livekit/components-react';
import { getToken } from '../api';
import './VoiceRoom.css';

// Helper to generate random room name
const generateRandomRoomName = () => {
  const prefixes = ['room', 'voice', 'meeting', 'session'];
  const prefix = prefixes[Math.floor(Math.random() * prefixes.length)];
  const random = Math.random().toString(36).substring(2, 7);
  return `${prefix}-${random}`;
};

const VoiceRoom = () => {
  const [roomName, setRoomName] = useState(() => generateRandomRoomName());
  // const [identity, setIdentity] = useState(`user-${Math.random().toString(36).substring(2, 8)}`);
  const [identity, setIdentity] = useState(`default_user`);
  const [isConnected, setIsConnected] = useState(false);
  const [token, setToken] = useState(null);
  const [url, setUrl] = useState(null);
  const [status, setStatus] = useState('Disconnected');
  const [selectedDoc, setSelectedDoc] = useState(null);

  // Load selected document from localStorage
  useEffect(() => {
    const docId = localStorage.getItem('selectedDocumentId');
    const docName = localStorage.getItem('selectedDocumentName');
    if (docId && docName) {
      setSelectedDoc({ id: docId, name: docName });
    }
  }, []);

  const clearSelectedDoc = () => {
    setSelectedDoc(null);
    localStorage.removeItem('selectedDocumentId');
    localStorage.removeItem('selectedDocumentName');
  };

  const handleConnect = async () => {
    try {
      console.log("[UI] Connect button clicked");
      console.log("[UI] Room name:", roomName);
      console.log("[UI] Identity:", identity);
      console.log("[UI] Selected Doc:", selectedDoc);
      
      setStatus('Connecting...');
      
      // Prepare metadata
      const metadata = selectedDoc ? {
        user_id: identity,
        document_id: selectedDoc.id,
        document_name: selectedDoc.name
      } : {
        user_id: identity,
        document_id: 'default_doc'
      };
      
      const data = await getToken(identity, roomName, metadata);
      
      console.log("[TOKEN RESPONSE]", data);
      console.log("[LIVEKIT URL]", data.url);
      
      setToken(data.token);
      setUrl(data.url);
      setIsConnected(true);
      setStatus('Connected');
    } catch (error) {
      console.error('[ERROR] Failed to connect:', error);
      setStatus('Connection failed');
    }
  };

  return (
    <div className="voice-room">
      <header className="voice-room-header fade-in">
        <div className="logo">
          <div className="logo-icon">🎙️</div>
          <div className="logo-text">
            <h1>AI Voice Agent</h1>
            <p>Real-time voice assistant</p>
          </div>
        </div>
        <div className="header-right">
          {selectedDoc && (
            <div className="selected-doc-badge">
              <span className="doc-icon">📄</span>
              <span className="doc-name">{selectedDoc.name}</span>
              <button className="clear-doc-btn" onClick={clearSelectedDoc} title="Clear document selection">
                ×
              </button>
            </div>
          )}
          <Link to="/documents" className="manage-docs-button">
            Manage Documents
          </Link>
          <div className="connection-status">
            <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}></div>
            <span>{status}</span>
          </div>
        </div>
      </header>

      <main className="voice-room-main">
        {!isConnected ? (
          <div className="join-form fade-in">
            <div className="form-card glass">
              <h2>Join a Room</h2>
              {selectedDoc ? (
                <div className="doc-info">
                  <div className="doc-info-icon">📄</div>
                  <div className="doc-info-text">
                    <strong>Active Document:</strong>
                    <div>{selectedDoc.name}</div>
                  </div>
                </div>
              ) : (
                <div className="doc-warning">
                  ⚠️ No document selected. The AI will use default document.
                </div>
              )}
              <div className="form-group">
                <label htmlFor="roomName">Room Name</label>
                <input
                  type="text"
                  id="roomName"
                  value={roomName}
                  onChange={(e) => setRoomName(e.target.value)}
                  placeholder="Enter room name"
                  className="input-field"
                />
              </div>
              <div className="form-group">
                <label htmlFor="identity">Your Identity</label>
                <input
                  type="text"
                  id="identity"
                  value={identity}
                  onChange={(e) => setIdentity(e.target.value)}
                  placeholder="Enter your identity"
                  className="input-field"
                />
              </div>
              <button onClick={handleConnect} className="join-button">
                Join Room
              </button>
            </div>
          </div>
        ) : (
          <LiveKitRoom
            token={token}
            serverUrl={url}
            connect={true}
            video={false}
            audio={true}
            style={{ width: '100%', height: '100%' }}
          >
            <ConnectedView setStatus={setStatus} roomNameFromParent={roomName} selectedDoc={selectedDoc} />
            <RoomAudioRenderer />
            <div className="control-bar-wrapper">
              <ControlBar />
            </div>
          </LiveKitRoom>
        )}
      </main>
    </div>
  );
};

// Connected View Component
const ConnectedView = ({ setStatus, roomNameFromParent, selectedDoc }) => {
  const connectionState = useConnectionState();
  const voiceAssistant = useVoiceAssistant();
  
  // Log ALL keys from voiceAssistant to find correct state property
  useEffect(() => {
    console.log("[VOICE ASSISTANT FULL OBJECT]", voiceAssistant);
    if (voiceAssistant) {
      console.log("[VOICE ASSISTANT KEYS]", Object.keys(voiceAssistant));
    }
  }, [voiceAssistant]);

  // Try all possible state properties
  let voiceState = 'idle';
  if (voiceAssistant) {
    if (voiceAssistant.state) voiceState = voiceAssistant.state;
    if (voiceAssistant.agent?.state) voiceState = voiceAssistant.agent.state;
    if (voiceAssistant.agentState) voiceState = voiceAssistant.agentState;
    if (voiceAssistant.agent_state) voiceState = voiceAssistant.agent_state;
  }

  // Custom state with detailed info
  const [detailedState, setDetailedState] = useState({
    status: 'idle',
    toolName: null
  });

  const messages = voiceAssistant?.messages || [];
  const transcriptEndRef = useRef(null);
  const roomContext = useRoomContext();
  const { localParticipant, microphoneTrack } = useLocalParticipant();
  const remoteParticipants = useParticipants();

  // Listen to data packets for agent status
  useEffect(() => {
    if (!roomContext) return;

    const handleDataReceived = (payload, participant, kind) => {
      try {
        const data = JSON.parse(new TextDecoder().decode(payload));
        if (data.type === "agent_status") {
          console.log("[AGENT STATUS RECEIVED]", data);
          setDetailedState({
            status: data.status,
            toolName: data.tool_name || null
          });
        }
      } catch (e) {
        console.error("[DATA PARSE ERROR]", e);
      }
    };

    roomContext.on('dataReceived', handleDataReceived);

    return () => {
      roomContext.off('dataReceived', handleDataReceived);
    };
  }, [roomContext]);

  useEffect(() => {
    console.log("[FINAL VOICE ASSISTANT STATE]", voiceState);
    console.log("[DETAILED AGENT STATE]", detailedState);
  }, [voiceState, detailedState]);

  // Log room info and LIVEKIT URL match with worker
  useEffect(() => {
    if (roomContext) {
      console.log("[ROOM CONTEXT]", roomContext);
      console.log("[ROOM NAME FROM LIVEKIT]", roomContext.name);
      console.log("[ROOM NAME FROM UI]", roomNameFromParent);
    }
  }, [roomContext, roomNameFromParent]);

  // Log connection state continuously
  useEffect(() => {
    console.log("[ROOM CONNECTION STATE]", connectionState);
    setStatus(connectionState === 'connected' ? 'Connected' : connectionState);
  }, [connectionState, setStatus]);

  // Log room and local participant/track info with diagnostics
  useEffect(() => {
    if (localParticipant) {
      console.log("[LOCAL PARTICIPANT]", localParticipant);
      if (microphoneTrack) {
        console.log("[MIC TRACK] Published Successfully", {
          isMuted: microphoneTrack.isMuted,
          name: microphoneTrack.trackName,
          kind: microphoneTrack.kind
        });
      }
    }
  }, [localParticipant, microphoneTrack]);

  // Log remote participants (should include the agent!)
  useEffect(() => {
    console.log("[REMOTE PARTICIPANTS COUNT]", remoteParticipants.length);
    if (remoteParticipants.length > 0) {
      remoteParticipants.forEach((rp) => {
        console.log("[REMOTE PARTICIPANT]", rp.identity, rp);
      });
    }
  }, [remoteParticipants]);

  // Debugging logs for room events
  useEffect(() => {
    if (!roomContext) return;

    const handleParticipantConnected = (participant) => {
      console.log("[DEBUG] Participant joined:", participant.identity);
    };

    const handleParticipantDisconnected = (participant) => {
      console.log("[DEBUG] Participant left:", participant.identity);
    };

    const handleTrackSubscribed = (track, publication, participant) => {
      console.log("[DEBUG] Track subscribed:", participant.identity, track.kind);
    };

    const handleTrackUnsubscribed = (track, publication, participant) => {
      console.log("[DEBUG] Track unsubscribed:", participant.identity, track.kind);
    };

    roomContext.on('participantConnected', handleParticipantConnected);
    roomContext.on('participantDisconnected', handleParticipantDisconnected);
    roomContext.on('trackSubscribed', handleTrackSubscribed);
    roomContext.on('trackUnsubscribed', handleTrackUnsubscribed);

    return () => {
      roomContext.off('participantConnected', handleParticipantConnected);
      roomContext.off('participantDisconnected', handleParticipantDisconnected);
      roomContext.off('trackSubscribed', handleTrackSubscribed);
      roomContext.off('trackUnsubscribed', handleTrackUnsubscribed);
    };
  }, [roomContext]);

  // Auto scroll to bottom of transcripts
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Helper to get icon and text based on detailed state
  const getDetailedStateIcon = (detailedState) => {
    switch (detailedState.status) {
      case 'listening': return '👂';
      case 'speaking': return '🔊';
      case 'thinking': return '🤔';
      case 'calling_tool': return '🔧';
      case 'generating_response': return '✨';
      case 'tool_failed': return '❌';
      default: return '🎧';
    }
  };

  const getDetailedStateText = (detailedState) => {
    switch (detailedState.status) {
      case 'listening': return 'Listening...';
      case 'speaking': return 'Speaking...';
      case 'thinking': return 'Thinking...';
      case 'calling_tool':
        return `Please wait, Calling Tool${detailedState.toolName ? `: ${detailedState.toolName}` : ''}...`;
      case 'generating_response': return 'Generating Response...';
      case 'tool_failed': return 'Tool Failed';
      default: return 'Ready to talk';
    }
  };

  // Separate agent participant from others: PRIORITIZE ditto-avatar-worker and anam-avatar-agent!
  const agentParticipant = remoteParticipants.find(
    (p) => p.identity === 'ditto-avatar-worker' || p.identity === 'anam-avatar-agent'
  ) || remoteParticipants.find(
    (p) => 
      p.identity.startsWith('agent-') || 
      p.identity === 'voice-agent-rag'
  );
  // Filter out avatar participant from badge list entirely!
  const otherParticipants = remoteParticipants.filter(
    (p) => 
      p.identity !== 'ditto-avatar-worker' &&
      p.identity !== 'anam-avatar-agent' &&
      (!agentParticipant || p.identity !== agentParticipant.identity)
  );
  // Ensure no duplicates (in case remoteParticipants includes local)
  const seenIdentities = new Set();
  const uniqueParticipants = [];
  if (localParticipant && !seenIdentities.has(localParticipant.identity)) {
    seenIdentities.add(localParticipant.identity);
    uniqueParticipants.push(localParticipant);
  }
  otherParticipants.forEach(p => {
    if (!seenIdentities.has(p.identity)) {
      seenIdentities.add(p.identity);
      uniqueParticipants.push(p);
    }
  });
  const allHumanParticipants = uniqueParticipants;

  return (
    <div className="connected-view">
      {/* Selected Document Display (below header) */}
      {/* {selectedDoc && (
        <div className="selected-doc-display glass">
          <span className="doc-icon">📄</span>
          <span className="doc-name">{selectedDoc.name}</span>
        </div>
      )} */}

      {/* Main content: Avatar and status */}
      <div className="main-content">
        {/* Avatar Container */}
        <div className="avatar-container glass">
          {agentParticipant ? (
            <ParticipantView key={agentParticipant.identity} participant={agentParticipant} isAgent={true} />
          ) : (
            <div className="avatar-placeholder">
              <div className="avatar-placeholder-icon">🤖</div>
            </div>
          )}
        </div>

        {/* Status Display */}
        <div className={`status-display ${detailedState.status}`}>
          <div className="status-icon">{getDetailedStateIcon(detailedState)}</div>
          <div className="status-text">{getDetailedStateText(detailedState)}</div>
          {/* Animated indicator based on state */}
          <div className="status-animator">
            {detailedState.status === 'listening' && (
              <div className="wave-animation">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="wave-bar" style={{ animationDelay: `${i * 0.1}s` }}></div>
                ))}
              </div>
            )}
            {detailedState.status === 'thinking' && (
              <div className="dots-animation">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            )}
            {detailedState.status === 'speaking' && (
              <div className="wave-animation speaking">
                {[...Array(7)].map((_, i) => (
                  <div key={i} className="wave-bar" style={{ animationDelay: `${i * 0.08}s` }}></div>
                ))}
              </div>
            )}
            {detailedState.status === 'calling_tool' && (
              <div className="spinner-animation"></div>
            )}
          </div>
        </div>
      </div>

      {/* Participants as badges */}
      <div className="participants-badges">
        <div className="participants-label">Connected</div>
        <div className="badges-list">
          {allHumanParticipants.map((participant, idx) => (
            <div key={participant.sid || participant.identity || idx} className="participant-badge">
              <div className="badge-icon">👤</div>
              <span>{participant.identity}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Message Component (still exists for possible future use)
const Message = ({ message }) => {
  if (!message || !message.text) return null;
  
  const isUser = message.role === 'user' || message.type === 'user';
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <div className={`message ${isUser ? 'user-message' : 'agent-message'} slide-in`}>
      <div className="message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="message-content">
        <div className="message-header">
          <span className="message-author">{isUser ? 'You' : 'Assistant'}</span>
          <span className="message-time">{time}</span>
        </div>
        <div className="message-text">{message.text}</div>
      </div>
    </div>
  );
};

// Participant View Component (updated)
const ParticipantView = ({ participant, isAgent }) => {
  const videoRef = useRef(null);
  const audioRef = useRef(null);

  useEffect(() => {
    const handleTrackPublished = (publication) => {
      console.log("[ParticipantView] Track published:", publication.trackSid, publication.kind);
    };

    const handleTrackSubscribed = (track, publication) => {
      console.log("[ParticipantView] Track subscribed:", track.sid, track.kind);
      if (track.kind === 'video' && videoRef.current) {
        track.attach(videoRef.current);
      }
      if (track.kind === 'audio' && audioRef.current) {
        track.attach(audioRef.current);
      }
    };

    const handleTrackUnsubscribed = (track, publication) => {
      console.log("[ParticipantView] Track unsubscribed:", track.sid, track.kind);
      if (track.kind === 'video' && videoRef.current) {
        track.detach(videoRef.current);
      }
      if (track.kind === 'audio' && audioRef.current) {
        track.detach(audioRef.current);
      }
    };

    participant.on('trackPublished', handleTrackPublished);
    participant.on('trackSubscribed', handleTrackSubscribed);
    participant.on('trackUnsubscribed', handleTrackUnsubscribed);

    // Attach already subscribed tracks
    if (participant.tracks) {
      const subscribedTracks = Array.from(participant.tracks.values());
      subscribedTracks.forEach((publication) => {
        if (publication.track) {
          handleTrackSubscribed(publication.track, publication);
        }
      });
    }

    return () => {
      participant.off('trackPublished', handleTrackPublished);
      participant.off('trackSubscribed', handleTrackSubscribed);
      participant.off('trackUnsubscribed', handleTrackUnsubscribed);
    };
  }, [participant]);

  if (isAgent) {
    return (
      <div className="agent-video-container">
        <video ref={videoRef} className="agent-video" autoPlay playsInline muted />
        <audio ref={audioRef} autoPlay />
      </div>
    );
  }

  return null;
};

export default VoiceRoom;
