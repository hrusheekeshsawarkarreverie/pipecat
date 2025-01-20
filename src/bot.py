import os
import sys

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from datetime import datetime
from pipecat.services.azure import AzureTTSService
from pipecat.services.azure import AzureSTTService
from pipecat.serializers.twilio import TwilioFrameSerializer
import aiohttp
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.reverie_tts import ReverieTTSService
from pipecat.services.reverie_stt import ReverieSTTService

from loguru import logger

from dotenv import load_dotenv

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Definet the global websocket
STREAM_SID_WS = {}
STREAM_SID_CONVERSATION = {}


async def run_bot(websocket_client, stream_sid,call_sid,bot_details):

    # function to save the bot context
    def save_bot_context(messages):
        bot_context_messages = []

        for message in messages:
            timestamped_message = message.copy()
            timestamped_message['timestamp'] = datetime.utcnow().isoformat() + 'Z'  #UTC timestamp
            bot_context_messages.append(timestamped_message)

        # log the bot context messages
        # logger.info(f"Bot Context Messages: {bot_context_messages}")
        logger.debug(f"Bot Context Messages: ... {bot_context_messages[-1]}")
        STREAM_SID_CONVERSATION[stream_sid] = bot_context_messages


    async with aiohttp.ClientSession() as session:
        logger.debug("started")
        transport = FastAPIWebsocketTransport(
            websocket=websocket_client,
            params=FastAPIWebsocketParams(
                audio_out_enabled=True,
                audio_out_sample_rate=16000,
                add_wav_header=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(sample_rate=16000),
                vad_audio_passthrough=True,
                serializer=TwilioFrameSerializer(stream_sid),
            ),
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"), 
            model="gpt-4o",
            save_bot_context=save_bot_context,
            # set_first_message_received=set_first_message_received,
            tgt_lan="bot_lang",
            nmt_flag="nmt_flag",
            nmt_provider="nmt_provider",
            conversation_id="conversation_id",
            ner_list="agentPrompt"
            
            )

        # stt_services = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        stt_services = AzureSTTService(
                                api_key=os.getenv("AZURE_API_KEY"),
                                region=os.getenv("AZURE_REGION"),
                                language="en-US",
                            )
        # stt_services = ReverieSTTService(
        #                         api_key="84148cc0e57e75c7d1b1331bb99a2e94aa588d48",
        #                         src_lang="en",
        #                         domain="generic",
        #                     )
        # tts_services = ElevenLabsTTSService(
        #                         aiohttp_session=session,
        #                         api_key=os.getenv("ELEVENLABS_API_KEY"),
        #                         voice_id="JNaMjd7t4u3EhgkVknn3",
        #                         call_provider="twilio",
        #                         model="eleven_multilingual_v1"
        #        
        #                 ),

        # tts_services = AzureTTSService(
        #                     api_key=os.getenv("AZURE_API_KEY"),
        #                     region=os.getenv("AZURE_REGION"),
        #                     voice="En-IN-PrabhatNeural",
        #                 )
        tts_services = ReverieTTSService(
                            aiohttp_session=session,
                            api_key=os.getenv("REVERIE_API_KEY"),
                            # speaker=f"{lang}_female",  # or another dynamic value if provided
                            speaker="hi_female",
                            format="wav",
                            speed=1.2,
                            pitch=1,
                        )
        # tts_services = ElevenLabsTTSService(
        #                     # aiohttp_session=session,
        #                     api_key=os.getenv("ELEVENLABS_API_KEY"),
        #                     voice_id="JNaMjd7t4u3EhgkVknn3",
        #                     # call_provider=call_provider,
        #                     model="eleven_multilingual_v1",
        #                 )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in an audio call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Websocket input from client
                stt_services,  # Speech-To-Text
                context_aggregator.user(),
                llm,  # LLM
                tts_services,  # Text-To-Speech
                transport.output(),  # Websocket output to client
                context_aggregator.assistant(),
            ]
        )

        logger.debug("transport done")
        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            # Kick off the conversation.
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMMessagesFrame(messages)])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            await task.queue_frames([EndFrame()])

        runner = PipelineRunner(handle_sigint=False)

        await runner.run(task)
