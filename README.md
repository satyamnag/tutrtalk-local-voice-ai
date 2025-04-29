<div align="center">
  <img src="./voice-assistant-frontend/.github/assets/app-icon.png" alt="App Icon" width="80" />
  <h1>🧠 Local Voice Agent</h1>
  <p>A full-stack, Dockerized AI voice assistant with speech, text, and voice synthesis powered by <a href="https://livekit.io/">LiveKit</a>.</p>
  <img src="./voice-assistant-frontend/.github/assets/frontend-screenshot.jpeg" alt="Screenshot" width="600" />
</div>

---

## 🧩 Overview

This repo contains everything needed to run a real-time AI voice assistant locally using:

- 🎙️ **LiveKit Agents** for STT ↔ LLM ↔ TTS
- 🧠 **Ollama** for running local LLMs
- 🗣️ **Kokoro** for TTS voice synthesis
- 👂 **Whisper (via VoxBox)** for speech-to-text
- 💬 **Next.js + Tailwind** frontend UI
- 🐳 Fully containerized via Docker Compose

## 🏁 Quick Start

```bash
./test.sh
```

This script:
- Cleans up existing containers
- Builds all services
- Launches the full stack (agent, LLM, STT, TTS, frontend, and signaling server)

Once it's up, visit [http://localhost:3000](http://localhost:3000) in your browser to start chatting.

## 📦 Architecture

```
[Frontend] → [LiveKit Room] ← [Agent]
                        ↘        ↙
              [Whisper]   [Ollama]   [Kokoro]
```

Each service is containerized and communicates over a shared Docker network:
- `livekit`: WebRTC signaling server
- `agent`: Custom Python agent with LiveKit SDK
- `whisper`: Speech-to-text using `vox-box` and Whisper model
- `ollama`: Local LLM provider (e.g., `gemma3:4b`)
- `kokoro`: TTS engine for speaking responses
- `frontend`: React-based client using LiveKit components

## 🧠 Agent Instructions

Your agent lives in [`agent/myagent.py`](./agent/myagent.py). It uses:
- `openai.STT` → routes to Whisper
- `openai.LLM` → routes to Ollama
- `groq.TTS` → routes to Kokoro
- `silero.VAD` → for voice activity detection

All metrics from each component are logged for debugging.

## 🔐 Environment Variables

You can find environment examples in:
- [`/.env`](./.env)
- [`/agent/.env`](./agent/.env)
- [`/voice-assistant-frontend/.env.example`](./voice-assistant-frontend/.env.example)

These provide keys and internal URLs for each service. Most keys are placeholders for local dev use.

## 🧪 Testing & Dev

To test or redeploy:

```bash
docker-compose down -v --remove-orphans
docker-compose up --build
```

The services will restart and build fresh containers.

## 🧰 Project Structure

```
.
├── agent/                     # Python voice agent
├── ollama/                    # LLM serving
├── whisper/                   # Whisper via vox-box
├── livekit/                   # Signaling server
├── voice-assistant-frontend/ # Next.js UI client
└── docker-compose.yml         # Brings it all together
```

## 📷 Screenshots

![UI Screenshot](./voice-assistant-frontend/.github/assets/frontend-screenshot.jpeg)

## 🛠️ Requirements

- Docker + Docker Compose
- No GPU required (uses CPU-based models)
- Recommended RAM: 18GB+

## 🙌 Credits

- Built with ❤️ by [LiveKit](https://livekit.io/)
- Uses [LiveKit Agents](https://docs.livekit.io/agents/)
- Local LLMs via [Ollama](https://ollama.com/)
- TTS via [Kokoro](https://github.com/remsky/kokoro)
