import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero, groq

load_dotenv()

logger = logging.getLogger("local-agent")
logger.setLevel(logging.INFO)

class SimpleAgent(Agent):
    def __init__(self) -> None:
        stt = openai.STT(base_url="http://whisper:80/v1", model="Systran/faster-whisper-small")
        llm = openai.LLM(base_url="http://ollama:11434/v1", model="gemma3:4b")
        tts = groq.TTS(base_url="http://kokoro:8880/v1", model="kokoro", voice="af_nova")
        vad_inst = silero.VAD.load()
        super().__init__(
            instructions="""
                You are a helpful agent. When the user speaks, you listen and respond.
                Never ever use emojis. Everything you say should be in plain text, since it will be spoken out loud.
                Keep your responses short and concise. Never more than a sentence or two.
            """,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad_inst
        )

        def llm_metrics_wrapper(metrics):
            import asyncio
            asyncio.create_task(self.on_llm_metrics_collected(metrics))
        llm.on("metrics_collected", llm_metrics_wrapper)

        def stt_metrics_wrapper(metrics):
            import asyncio
            asyncio.create_task(self.on_stt_metrics_collected(metrics))
        stt.on("metrics_collected", stt_metrics_wrapper)

        def eou_metrics_wrapper(metrics):
            import asyncio
            asyncio.create_task(self.on_eou_metrics_collected(metrics))
        stt.on("eou_metrics_collected", eou_metrics_wrapper)

        def tts_metrics_wrapper(metrics):
            import asyncio
            asyncio.create_task(self.on_tts_metrics_collected(metrics))
        tts.on("metrics_collected", tts_metrics_wrapper)

        def vad_metrics_wrapper(event):
            import asyncio
            asyncio.create_task(self.on_vad_event(event))
        vad_inst.on("metrics_collected", vad_metrics_wrapper)

    async def on_llm_metrics_collected(self, metrics):
        logger.info(f"LLM Metrics: {{" +
            f"'type': {metrics.type}, " +
            f"'label': {metrics.label}, " +
            f"'request_id': {metrics.request_id}, " +
            f"'timestamp': {metrics.timestamp if isinstance(metrics.timestamp, (int, float)) else metrics.timestamp.isoformat() if metrics.timestamp else None}, " +
            f"'duration': {metrics.duration}, " +
            f"'ttft': {getattr(metrics, 'ttft', None)}, " +
            f"'cancelled': {getattr(metrics, 'cancelled', None)}, " +
            f"'completion_tokens': {getattr(metrics, 'completion_tokens', None)}, " +
            f"'prompt_tokens': {getattr(metrics, 'prompt_tokens', None)}, " +
            f"'total_tokens': {getattr(metrics, 'total_tokens', None)}, " +
            f"'tokens_per_second': {getattr(metrics, 'tokens_per_second', None)}" +
            "}")

    async def on_stt_metrics_collected(self, metrics):
        logger.info(f"STT Metrics: {{" +
            f"'type': {metrics.type}, " +
            f"'label': {metrics.label}, " +
            f"'request_id': {metrics.request_id}, " +
            f"'timestamp': {metrics.timestamp if isinstance(metrics.timestamp, (int, float)) else metrics.timestamp.isoformat() if metrics.timestamp else None}, " +
            f"'duration': {metrics.duration}, " +
            f"'speech_id': {getattr(metrics, 'speech_id', None)}, " +
            f"'error': {str(getattr(metrics, 'error', None)) if getattr(metrics, 'error', None) else None}, " +
            f"'streamed': {getattr(metrics, 'streamed', None)}, " +
            f"'audio_duration': {getattr(metrics, 'audio_duration', None)}" +
            "}")

    async def on_eou_metrics_collected(self, metrics):
        logger.info(f"EOU Metrics: {{" +
            f"'type': {metrics.type}, " +
            f"'label': {metrics.label}, " +
            f"'timestamp': {metrics.timestamp if isinstance(metrics.timestamp, (int, float)) else metrics.timestamp.isoformat() if metrics.timestamp else None}, " +
            f"'end_of_utterance_delay': {getattr(metrics, 'end_of_utterance_delay', None)}, " +
            f"'transcription_delay': {getattr(metrics, 'transcription_delay', None)}, " +
            f"'speech_id': {getattr(metrics, 'speech_id', None)}, " +
            f"'error': {str(getattr(metrics, 'error', None)) if getattr(metrics, 'error', None) else None}" +
            "}")

    async def on_tts_metrics_collected(self, metrics):
        logger.info(f"TTS Metrics: {{" +
            f"'type': {metrics.type}, " +
            f"'label': {metrics.label}, " +
            f"'request_id': {metrics.request_id}, " +
            f"'timestamp': {metrics.timestamp if isinstance(metrics.timestamp, (int, float)) else metrics.timestamp.isoformat() if metrics.timestamp else None}, " +
            f"'ttfb': {getattr(metrics, 'ttfb', None)}, " +
            f"'duration': {metrics.duration}, " +
            f"'audio_duration': {getattr(metrics, 'audio_duration', None)}, " +
            f"'cancelled': {getattr(metrics, 'cancelled', None)}, " +
            f"'characters_count': {getattr(metrics, 'characters_count', None)}, " +
            f"'streamed': {getattr(metrics, 'streamed', None)}, " +
            f"'speech_id': {getattr(metrics, 'speech_id', None)}, " +
            f"'error': {str(getattr(metrics, 'error', None)) if getattr(metrics, 'error', None) else None}" +
            "}")

    async def on_vad_event(self, event):
        None

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession()

    await session.start(
        agent=SimpleAgent(),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))