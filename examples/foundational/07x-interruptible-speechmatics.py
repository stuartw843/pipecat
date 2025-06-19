#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import os
import sys
from pathlib import Path

# Add src to path for local development
root_dir = Path(__file__).parent.parent.parent
src_path = root_dir / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndOfUtteranceFrame, TranscriptionFrame, InterimTranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.cartesia.tts import CartesiaHttpTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


class EndOfUtteranceProcessor(FrameProcessor):
    """Processor to handle end of utterance events from Speechmatics STT."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, EndOfUtteranceFrame):
            logger.info(f"🔚 End of utterance detected - ready for AI response! Start: {frame.start_time}s, End: {frame.end_time}s")
            # This is where your voice AI would process the complete utterance
            # and generate a response

        await self.push_frame(frame, direction)


class SpeakerDiarizationProcessor(FrameProcessor):
    """Processor to handle speaker diarization and modify transcription content for LLM context."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            # Check if speaker information is available
            if frame.user_id:
                # Speechmatics uses "S1", "S2", etc. or custom speaker names
                speaker_label = frame.user_id
                
                # If it's the standard S1, S2 format, convert to more readable format
                if speaker_label.startswith("S") and speaker_label[1:].isdigit():
                    speaker_num = speaker_label[1:]  # Extract number from "S1" -> "1"
                    speaker_label = f"Speaker {speaker_num}"
                
                # Modify the text to include speaker label for LLM context
                original_text = frame.text
                modified_text = f"{speaker_label}: {original_text}"
                
                logger.info(f"🎤 {speaker_label} said: {original_text}")
                
                # Create a new frame with the modified text
                if isinstance(frame, TranscriptionFrame):
                    new_frame = TranscriptionFrame(
                        text=modified_text,
                        user_id=frame.user_id,
                        timestamp=frame.timestamp,
                        language=frame.language,
                        result=frame.result
                    )
                else:  # InterimTranscriptionFrame
                    new_frame = InterimTranscriptionFrame(
                        text=modified_text,
                        user_id=frame.user_id,
                        timestamp=frame.timestamp,
                        language=frame.language,
                        result=frame.result
                    )
                
                # Copy metadata and other properties
                new_frame.pts = frame.pts
                new_frame.metadata = frame.metadata
                new_frame.transport_source = frame.transport_source
                new_frame.transport_destination = frame.transport_destination
                
                await self.push_frame(new_frame, direction)
                return

        await self.push_frame(frame, direction)


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot with Speechmatics STT")

    # Optimized Speechmatics STT with reduced latency settings, end of utterance detection, and speaker diarization
    stt = SpeechmaticsSTTService(
        api_key=os.getenv("SPEECHMATICS_API_KEY"),
        chunk_size=1024,  # Reduced from default 1024 for lower latency
        enable_partials=True,  # Ensure partial results are enabled
        end_of_utterance_silence_trigger=0.5,  # Enable end of utterance detection with 0.5s silence
        enable_speaker_diarization=True,  # Enable speaker diarization
        max_speakers=4,  # Maximum number of speakers to detect (optional)
    )


    tts = CartesiaHttpTTSService(
    api_key=os.getenv("CARTESIA_API_KEY"),
    voice_id="bf0a246a-8642-498a-9950-80c35e9276b5",
    model="sonic",
    params=CartesiaHttpTTSService.InputParams(
        language="en",
    )
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"),model="gpt-4o-mini")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Create processors
    eou_processor = EndOfUtteranceProcessor()
    speaker_processor = SpeakerDiarizationProcessor()

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # Speechmatics STT with speaker diarization
            speaker_processor,  # Speaker diarization processor (adds speaker labels to transcription)
            eou_processor,  # End of utterance processor
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Just say hi."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_example, transport_params=transport_params)
