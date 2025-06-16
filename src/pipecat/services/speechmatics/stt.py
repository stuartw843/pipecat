#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Dict, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    EndOfUtteranceFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import speechmatics
    from speechmatics.models import ConnectionSettings, TranscriptionConfig, AudioSettings, ConversationConfig
    from speechmatics.client import WebsocketClient
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Speechmatics, you need to `pip install speechmatics-python`.")
    raise Exception(f"Missing module: {e}")


class AudioProcessor:
    """Audio processor for handling audio data stream to Speechmatics with async-optimized design."""
    
    def __init__(self):
        self.wave_data = bytearray()
        self.read_offset = 0
        self._lock = asyncio.Lock()
        self._closed = False

    async def read(self, chunk_size):
        """Read audio data for streaming to Speechmatics (async version with minimal latency)."""
        # High-efficiency async approach with minimal sleep time
        while not self._closed and self.read_offset + chunk_size > len(self.wave_data):
            await asyncio.sleep(0.001)  # Much shorter sleep for lower latency
            
        if self._closed:
            return b''
            
        async with self._lock:
            if self.read_offset + chunk_size > len(self.wave_data):
                return b''
                
            new_offset = self.read_offset + chunk_size
            data = self.wave_data[self.read_offset:new_offset]
            self.read_offset = new_offset
            
            # Clean up old data more aggressively to prevent memory growth
            if self.read_offset > 4096:  # Clean up after 4KB for better memory management
                self.wave_data = self.wave_data[self.read_offset:]
                self.read_offset = 0
                
            return bytes(data)

    async def write_audio(self, data):
        """Write audio data to buffer (async version)."""
        if not self._closed:
            async with self._lock:
                self.wave_data.extend(data)

    def write_audio_sync(self, data):
        """Synchronous write for compatibility."""
        if not self._closed:
            # Use asyncio.create_task if event loop is running
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.write_audio(data))
            except RuntimeError:
                # Fallback for when no event loop is running
                self.wave_data.extend(data)

    async def close(self):
        """Close the audio processor."""
        async with self._lock:
            self._closed = True

    async def clear(self):
        """Clear the audio buffer."""
        async with self._lock:
            self.wave_data.clear()
            self.read_offset = 0
            self._closed = False


class AsyncAudioProcessorWrapper:
    """Wrapper to provide sync interface for speechmatics client while using async processor internally."""
    
    def __init__(self, async_processor):
        self.async_processor = async_processor
        self._loop = None

    def read(self, chunk_size):
        """Synchronous read that bridges to async processor."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # Create a new task and run it
            task = loop.create_task(self.async_processor.read(chunk_size))
            # This is a bit tricky - we need to block until the task completes
            # We'll use a simple approach with a small timeout loop
            import time
            start_time = time.time()
            while not task.done() and time.time() - start_time < 0.1:
                time.sleep(0.001)
            
            if task.done():
                return task.result()
            else:
                return b''
        except RuntimeError:
            # No event loop running, fall back to blocking approach
            return asyncio.run(self.async_processor.read(chunk_size))

    def write_audio(self, data):
        """Synchronous write that bridges to async processor."""
        self.async_processor.write_audio_sync(data)

    def close(self):
        """Synchronous close."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.async_processor.close())
        except RuntimeError:
            asyncio.run(self.async_processor.close())

    def clear(self):
        """Synchronous clear."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.async_processor.clear())
        except RuntimeError:
            asyncio.run(self.async_processor.clear())


class SpeechmaticsSTTService(STTService):
    """Speechmatics STT service implementation.
    
    This service provides real-time speech-to-text transcription using the Speechmatics API.
    It supports partial and final transcriptions, multiple languages, and various audio formats.
    
    Args:
        api_key: Speechmatics API key for authentication.
        language: Language code for transcription (default: Language.EN).
        base_url: Base URL for Speechmatics API (default: eu2.rt.speechmatics.com).
        enable_partials: Enable partial transcription results (default: True).
        max_delay: Maximum delay for transcription in seconds (default: 5).
        sample_rate: Audio sample rate in Hz (default: None, inferred from pipeline).
        chunk_size: Audio chunk size for streaming (default: 256).
        audio_encoding: Audio encoding format (default: "pcm_f32le").
        end_of_utterance_silence_trigger: Silence duration in seconds to trigger end of utterance detection (default: None, disabled).
        operating_point: Operating point for transcription accuracy vs. latency tradeoff (default: "enhanced").
        transcription_config: Custom transcription configuration.
        **kwargs: Additional arguments passed to STTService.
    """

    def __init__(
        self,
        *,
        api_key: str,
        language: Language = Language.EN,
        base_url: str = "preview.rt.speechmatics.com",
        enable_partials: bool = True,
        max_delay: float = 1.5,
        sample_rate: Optional[int] = None,
        chunk_size: int = 256,
        audio_encoding: str = "pcm_s16le",
        end_of_utterance_silence_trigger: Optional[float] = None,
        operating_point: str = "enhanced",
        transcription_config: Optional[TranscriptionConfig] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._language = language
        self._base_url = base_url
        self._enable_partials = enable_partials
        self._max_delay = max_delay
        self._chunk_size = chunk_size
        self._audio_encoding = audio_encoding
        self._end_of_utterance_silence_trigger = end_of_utterance_silence_trigger
        self._operating_point = operating_point
        self._custom_config = transcription_config

        # Connection management
        self._connection_settings = None
        self._websocket_client = None
        self._audio_processor = None
        self._connected = False
        self._connection_task = None
        
        # Dedicated thread executor for reduced overhead
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speechmatics-stt")

        # Set model name for metrics
        self.set_model_name("speechmatics")

    @property
    def connection_url(self) -> str:
        """Get the WebSocket connection URL."""
        return f"wss://{self._base_url}/v2/{self._language.value}"

    def can_generate_metrics(self) -> bool:
        return True

    async def set_language(self, language: Language):
        """Set the transcription language.
        
        Args:
            language: The language to use for transcription.
        """
        logger.info(f"Switching STT language to: [{language}]")
        self._language = language
        await self._disconnect()
        if self._connected:
            await self._connect()

    async def start(self, frame: StartFrame):
        """Start the STT service.
        
        Args:
            frame: The start frame containing audio configuration.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service.
        
        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service.
        
        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data for speech-to-text conversion.
        
        Args:
            audio: Raw audio bytes to process.
            
        Yields:
            Frame objects as they become available.
        """
        if not self._connected:
            await self._connect()
            
        if self._connected and self._audio_processor:
            self._audio_processor.write_audio(audio)
        
        # This method needs to yield, but actual frames are pushed via event handlers
        # We yield nothing here as frames are handled asynchronously
        return
        yield  # This line will never be reached, but satisfies the AsyncGenerator type

    async def _connect(self):
        """Establish connection to Speechmatics WebSocket API."""
        if self._connected:
            return

        try:
            logger.debug("Connecting to Speechmatics")

            # Set up connection settings
            self._connection_settings = ConnectionSettings(
                url=self.connection_url,
                auth_token=self._api_key,
            )

            # Create WebSocket client
            self._websocket_client = WebsocketClient(self._connection_settings)

            # Set up async audio processor with sync wrapper for speechmatics client
            async_processor = AudioProcessor()
            self._audio_processor = AsyncAudioProcessorWrapper(async_processor)

            # Configure transcription settings
            if self._custom_config:
                transcription_config = self._custom_config
            else:
                # Configure conversation config for end of utterance detection
                conversation_config = None
                if self._end_of_utterance_silence_trigger is not None:
                    conversation_config = ConversationConfig(
                        end_of_utterance_silence_trigger=self._end_of_utterance_silence_trigger
                    )
                
                transcription_config = TranscriptionConfig(
                    language=self._language.value,
                    enable_partials=self._enable_partials,
                    max_delay=self._max_delay,
                    operating_point=self._operating_point,
                    conversation_config=conversation_config,
                )

            # Configure audio settings
            audio_settings = AudioSettings()
            audio_settings.encoding = self._audio_encoding
            audio_settings.sample_rate = self.sample_rate
            audio_settings.chunk_size = self._chunk_size

            # Register event handlers
            self._websocket_client.add_event_handler(
                event_name=speechmatics.models.ServerMessageType.AddPartialTranscript,
                event_handler=self._on_partial_transcript,
            )

            self._websocket_client.add_event_handler(
                event_name=speechmatics.models.ServerMessageType.AddTranscript,
                event_handler=self._on_final_transcript,
            )

            # Register end of utterance event handler
            self._websocket_client.add_event_handler(
                event_name=speechmatics.models.ServerMessageType.EndOfUtterance,
                event_handler=self._on_end_of_utterance,
            )

            # Start connection in background task
            self._connection_task = self.create_task(
                self._run_connection(transcription_config, audio_settings)
            )
            
            self._connected = True
            logger.debug("Connected to Speechmatics")

        except Exception as e:
            logger.error(f"Failed to connect to Speechmatics: {e}")
            self._connected = False
            raise

    async def _disconnect(self):
        """Disconnect from Speechmatics WebSocket API."""
        if not self._connected:
            return

        try:
            logger.debug("Disconnecting from Speechmatics")
            
            self._connected = False

            if self._audio_processor:
                self._audio_processor.close()
                self._audio_processor = None

            if self._connection_task:
                await self.cancel_task(self._connection_task)
                self._connection_task = None

            self._websocket_client = None
            self._connection_settings = None

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    async def cleanup(self):
        """Cleanup resources including the thread executor."""
        await self._disconnect()
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=True)

    async def _run_connection(self, transcription_config: TranscriptionConfig, audio_settings: AudioSettings):
        """Run the WebSocket connection with Speechmatics.
        
        Args:
            transcription_config: Configuration for transcription.
            audio_settings: Configuration for audio processing.
        """
        try:
            # Run the synchronous speechmatics client in a thread executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._websocket_client.run_synchronously,
                self._audio_processor,
                transcription_config,
                audio_settings
            )
        except Exception as e:
            logger.error(f"Connection error: {e}")
            # Attempt reconnection
            if self._connected:
                await self._reconnect()

    async def _reconnect(self):
        """Reconnect to Speechmatics after connection failure."""
        logger.info("Attempting to reconnect to Speechmatics")
        await self._disconnect()
        await asyncio.sleep(1)  # Brief delay before reconnection
        await self._connect()

    def _on_partial_transcript(self, message: Dict):
        """Handle partial transcript events.
        
        Args:
            message: Partial transcript message from Speechmatics.
        """
        try:
            transcript = message.get("metadata", {}).get("transcript", "")
            if transcript:
                # Schedule the async operations using the event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self._handle_partial_transcript(transcript, message))
                except RuntimeError:
                    # Fallback if no event loop is running
                    logger.warning("No event loop available for partial transcript")
        except Exception as e:
            logger.error(f"Error processing partial transcript: {e}")

    def _on_final_transcript(self, message: Dict):
        """Handle final transcript events.
        
        Args:
            message: Final transcript message from Speechmatics.
        """
        try:
            transcript = message.get("metadata", {}).get("transcript", "")
            if transcript:
                # Schedule the async operations using the event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self._handle_final_transcript(transcript, message))
                except RuntimeError:
                    # Fallback if no event loop is running
                    logger.warning("No event loop available for final transcript")
        except Exception as e:
            logger.error(f"Error processing final transcript: {e}")

    def _on_end_of_utterance(self, message: Dict):
        """Handle end of utterance events.
        
        Args:
            message: End of utterance message from Speechmatics.
        """
        try:
            # Schedule the async operations using the event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._handle_end_of_utterance(message))
            except RuntimeError:
                # Fallback if no event loop is running
                logger.warning("No event loop available for end of utterance")
        except Exception as e:
            logger.error(f"Error processing end of utterance: {e}")

    async def _handle_partial_transcript(self, transcript: str, message: Dict):
        """Handle partial transcript asynchronously."""
        try:
            await self.stop_ttfb_metrics()
            await self.push_frame(
                InterimTranscriptionFrame(
                    transcript,
                    "",
                    time_now_iso8601(),
                    self._language,
                    result=message,
                )
            )
        except Exception as e:
            logger.error(f"Error handling partial transcript: {e}")

    async def _handle_final_transcript(self, transcript: str, message: Dict):
        """Handle final transcript asynchronously."""
        try:
            await self.stop_ttfb_metrics()
            await self.push_frame(
                TranscriptionFrame(
                    transcript,
                    "",
                    time_now_iso8601(),
                    self._language,
                    result=message,
                )
            )
            await self._handle_transcription(transcript, True, self._language)
            await self.stop_processing_metrics()
        except Exception as e:
            logger.error(f"Error handling final transcript: {e}")

    async def _handle_end_of_utterance(self, message: Dict):
        """Handle end of utterance asynchronously."""
        try:
            metadata = message.get("metadata", {})
            start_time = metadata.get("start_time")
            end_time = metadata.get("end_time")
            
            await self.push_frame(
                EndOfUtteranceFrame(
                    user_id="",
                    timestamp=time_now_iso8601(),
                    start_time=start_time,
                    end_time=end_time,
                    result=message,
                )
            )
        except Exception as e:
            logger.error(f"Error handling end of utterance: {e}")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing.
        
        Args:
            transcript: The transcribed text.
            is_final: Whether this is a final or interim result.
            language: The detected language.
        """
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames.
        
        Args:
            frame: The frame to process.
            direction: The direction of the frame.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            # Start metrics when user begins speaking
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Could implement end-of-speech logic here if needed
            pass

    async def start_ttfb_metrics(self):
        """Start time-to-first-byte metrics."""
        await super().start_ttfb_metrics()

    async def start_processing_metrics(self):
        """Start processing metrics."""
        await super().start_processing_metrics()
