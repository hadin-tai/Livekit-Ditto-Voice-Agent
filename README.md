# AI Voice Agent - Phase 1

A production-grade real-time voice assistant using LiveKit, Deepgram, Google Gemini 2.5 Flash Lite, and Cartesia.

## Architecture
```
React Frontend
    │
    ▼
FastAPI Backend
    │
    ▼
LiveKit Cloud
    │
    ▼
LiveKit Agent Worker
    │
┌──────────────┐
│ Deepgram STT│
└──────────────┘
    │
    ▼
Gemini 2.5 Flash Lite
    │
    ▼
Cartesia TTS
    │
    ▼
LiveKit Room
```

## Setup Instructions

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm or yarn
- LiveKit Cloud account
- Deepgram API key
- Google API key
- Cartesia API key

### Backend Setup
1. Navigate to backend directory:
```bash
cd backend
```

2. Copy .env.example to .env and fill in your API keys:
```bash
cp .env.example .env
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup
1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

## Run Instructions

### Start FastAPI Backend (Terminal 1)
```bash
cd backend
uvicorn app.main:app --reload
```
Backend will run at http://localhost:8000

### Start LiveKit Agent Worker (Terminal 2)
```bash
cd backend
python -m agent.worker dev
```

### Start Frontend (Terminal 3)
```bash
cd frontend
npm run dev
```
Frontend will run at http://localhost:5173

## Testing Instructions
1. Open http://localhost:5173 in your browser
2. Enter a room name and identity
3. Click "Join Room"
4. Allow microphone permissions
5. Speak into your microphone
6. The AI will respond with generated speech

## Troubleshooting Guide

### Backend won't start
- Check if .env file has all required variables
- Verify Python version is 3.11+
- Check if dependencies are installed correctly

### Agent won't connect
- Ensure LiveKit API keys are correct in .env
- Verify LiveKit Cloud project is active
- Check agent logs for errors

### Frontend can't connect to room
- Ensure backend is running on port 8000
- Check browser console for errors
- Verify LiveKit URL is correct

### No speech recognition
- Ensure microphone permissions are granted
- Check Deepgram API key
- Verify browser supports WebRTC

### No AI response
- Check Google API key
- Verify Cartesia API key
- Check agent logs for LLM/TTS errors

## API Endpoints

### GET /health
Returns health status
```json
{ "status": "ok" }
```

### POST /token
Generates LiveKit access token
- Input: `{ "identity": "user123", "room_name": "test-room" }`
- Output: `{ "token": "...", "url": "..." }`
