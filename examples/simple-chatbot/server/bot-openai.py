#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Bot Implementation.

This module implements a chatbot using OpenAI's GPT-4 model for natural language
processing. It includes:
- Real-time audio/video interaction through Daily
- Animated robot avatar
- Text-to-speech using ElevenLabs
- Support for both English and Spanish

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow.
"""

import asyncio
import os
import sys
import hashlib

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from runner import configure
import json
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    LLMMessagesFrame,
    OutputImageRawFrame,
    SpriteFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import (
    RTVIBotTranscriptionProcessor,
    RTVIConfig,
    RTVIMetricsProcessor,
    RTVIProcessor,
    RTVISpeakingProcessor,
    RTVIUserTranscriptionProcessor,
)
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
import redis

from pipecat_old.services.dify import (
    DifyLLMService,
)



load_dotenv(override=True)
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sprites = []
script_dir = os.path.dirname(__file__)
        # Connect to Redis and fetch bot details
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)
# Load sequential animation frames
for i in range(1, 26):
    # Build the full path to the image file
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

# Create a smooth animation by adding reversed frames
flipped = sprites[::-1]
sprites.extend(flipped)

# Define static and animated states
quiet_frame = sprites[0]  # Static frame for when bot is listening
talking_frame = SpriteFrame(images=sprites)  # Animation sequence for when bot is talking


class TalkingAnimation(FrameProcessor):
    """Manages the bot's visual animation states.

    Switches between static (listening) and animated (talking) states based on
    the bot's current speaking status.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and update animation state.

        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        await super().process_frame(frame, direction)

        # Switch to talking animation when bot starts speaking
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        # Return to static frame when bot stops speaking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame, direction)


async def main():
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport
    - Speech-to-text and text-to-speech services
    - Language model integration
    - Animation processing
    - RTVI event handling
    """

    async with aiohttp.ClientSession() as session:
        (room_url, token,conversation_id) = await configure(session)
        print(f"room_url: {room_url}, token: {token}, conversation_id: {conversation_id}")



        # Fetch bot details using conversation_id
        bot_details = redis_client.get(f"bot_details:{conversation_id}")
        if bot_details:
            logger.info(f"Retrieved bot details for conversation_id {conversation_id}")
            try:
                # Check if bot_details is already a dictionary
                if isinstance(bot_details, dict):
                    logger.info("Bot details is already a dictionary")
                    bot_details_dict = bot_details
                else:
                    # Try to parse as JSON string
                    bot_details_dict = json.loads(bot_details)
                logger.info(f"Parsed bot details: {bot_details_dict}")
                bot_details = bot_details_dict
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse bot details JSON: {e}")
                logger.error(f"Raw bot details: {bot_details}")
                bot_details = {}
        else:
            logger.warning(f"No bot details found for conversation_id {conversation_id}")
            bot_details = {}

        api_details = bot_details.get("api_details", {})
        logger.info(f"api_details {api_details}")
        available_languages = api_details.get("available_languages", [])
        logger.info(f"available language {available_languages}")

        selected_stt_providers = bot_details.get("providerData",{}).get("selected", {}).get("ivr", {}).get("stt", {})
        logger.info(f"selected_stt_providers: {selected_stt_providers}")
        serializer_stt_provider = selected_stt_providers.get(available_languages[0],"")
        logger.info(f"serializer_stt_provider: {serializer_stt_provider}")

        ivrDetails = bot_details.get("ivrDetails", {})
        # logger.info(f"IVR Details: {ivrDetails}")

        providerData = bot_details.get("providerData", {})
        # logger.info(f"PROVIDER DATA: {providerData}")
        botType = ivrDetails.get("botType")
        nmt_flag = ivrDetails.get("nmt")
        nmt_provider = ivrDetails.get("nmtProvider")
        logger.debug(f"NMT: {nmt_flag}, NMT Provider: {nmt_provider }")
        user_details = bot_details.get("user_details", {})

        # callProvider= ivrDetails.get("callProvider")

        # logger.info(f"Call Provider: {callProvider}")
        logger.info(f"Bot Type: {botType}")

        agentSettings = bot_details.get("agentSettings", {})
        # logger.info(f"AGENT SETTINGS: {agentSettings}")
        callProvider = agentSettings.get("call", {}).get("callProvider", "")
        llmProvider = agentSettings.get("llm", {}).get("llmProvider", "")
        llmModel = "gpt-4o"  # Default model
                # call_provider = "exotel"
        call_provider = callProvider or "twilio"
        logger.debug(f"call_provider: {call_provider}")
        # call_provider = "twilio" # removing the callProvider since we know for sure if the call is coming from /media the call is coming from exotel
        # and we can take that info form the kwargs dict

        # conv_id = None #conv id is global used inside 2 api calls

        if llmProvider == "openai":
            llmModel = agentSettings.get("llm", {}).get("llmModel", llmModel)

        logger.info(f"Selected LLM Model: {llmProvider,llmModel}")

        # logger.info(f"Logger Info: {stt_pipeline}")



        global current_language
        current_language = ""
        # current_language = "hi"

        # initialize messages
        messages = []

        async def english_language_filter(frame) -> bool:
            # log that the current language is being checked
            # logger.debug(f"Checking the current language: {current_language}")
            return current_language == "en"

        async def hindi_language_filter(frame) -> bool:
            # log that the current language is being checked
            # logger.debug(f"Checking the current language: {current_language}")
            return current_language == "hi"

        async def hindi_tts_language_filter(frame) -> bool:
            # log that the current language is being checked
            # logger.debug(f"Checking the current language: {current_language}")
            return current_language == "hi" or current_language == "choice"

        # function to set language choice filter
        async def choice_language_filter(frame) -> bool:
            return current_language == "choice"

        # bengali language filter
        async def bengali_language_filter(frame) -> bool:
            return current_language == "bn"

        # assamese language filter
        async def assamese_language_filter(frame) -> bool:
            return current_language == "as"

        # kannada language filter
        async def kannada_language_filter(frame) -> bool:
            return current_language == "kn"

        # malayalam language filter
        async def malayalam_language_filter(frame) -> bool:
            return current_language == "ml"

        # marathi language filter
        async def marathi_language_filter(frame) -> bool:
            return current_language == "mr"

        # odia language filter
        async def odia_language_filter(frame) -> bool:
            return current_language == "or"

        # tamil language filter
        async def tamil_language_filter(frame) -> bool:
            return current_language == "ta"

        # telugu language filter
        async def telugu_language_filter(frame) -> bool:
            return current_language == "te"

        # punjabi language filter
        async def punjabi_language_filter(frame) -> bool:
            return current_language == "pa"

        # gujarati language filter
        async def gujarati_language_filter(frame) -> bool:
            return current_language == "gu"

        # arabic language filter
        async def arabic_language_filter(frame) -> bool:
            return current_language == "ar"

        # write a function which will change the global variable current_language, and this function will be called from the llm service
        def set_current_language(language):
            # log that the current language is being changed
            logger.debug(f"Changing the current language to: {language}")

            global current_language
            current_language = language

        # add global variable to check if we have received first message from llm
        first_message_received = False

        # add parallel pipeline filter to check if we have received first message from llm
        async def first_message_filter(frame) -> bool:
            return first_message_received

        # function which will be called from llm service to set first message received
        def set_first_message_received():
            global first_message_received
            first_message_received = True
            logger.debug(f"First message received: {first_message_received}")



        bot_context_messages = []

        # function to save the bot context
        # def save_bot_context(messages):
        #     """
        #     Appends new messages to STREAM_SID_CONVERSATION without duplication.
        #     """
        #     timestamped_messages = []
        #     for message in messages:
        #         if 'content' in message:
        #             timestamped_message = message.copy()
        #             timestamped_message['timestamp'] = datetime.utcnow().isoformat() + 'Z'  # UTC timestamp
        #             timestamped_messages.append(timestamped_message)
            
        #     async def append_messages():
        #         async with conversation_lock:
        #             # Initialize conversation and last_saved_index if not present
        #             if stream_sid not in STREAM_SID_CONVERSATION:
        #                 STREAM_SID_CONVERSATION[stream_sid] = []
        #                 last_saved_index[stream_sid] = 0
                    
        #             # Retrieve the last saved index
        #             last_index = last_saved_index.get(stream_sid, 0)
                    
        #             # Determine new messages to append
        #             new_messages = timestamped_messages[last_index:]
                    
        #             if new_messages:
        #                 # Append new messages
        #                 STREAM_SID_CONVERSATION[stream_sid].extend(new_messages)
                        
        #                 # Update the last saved index
        #                 last_saved_index[stream_sid] = len(timestamped_messages)
                        
        #                 # Log the latest message for verification
        #                 logger.debug(f"Bot Context Messages: ... {STREAM_SID_CONVERSATION[stream_sid][-1]}")
            
        #     # Schedule the append operation without blocking
        #     asyncio.create_task(append_messages())


        async def save_conversation(bot_context_messages, bot_details, call_sid, provider):

            # Generate conv_id
            # conv_id = uuid.uuid4().hex[:16]
            conv_id = hashlib.md5(call_sid.encode()).hexdigest()[:16]

            # Extract template_id from bot_details
            template_id = bot_details["api_details"].get("TEMPLATE", "default_template")
            project_id = bot_details["api_details"].get("PROJECT", "default_project")

            url = "http://172.18.0.04:8003/save_dify_conversation"

            # Prepare the data payload
            data = {
                "conv_id": conv_id,
                "template_id": project_id,
                "conversation": bot_context_messages,
                "call_sid" : call_sid,
                "provider" : provider
            }

            # log the data
            logger.info(f"Data: {data}")

            # Make the API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(data),
                ) as response:
                    response_data = await response.text()
                    logger.info(f"API Response: {response_data}")

            return response_data

        async def call_summarize_api(bot_context_messages, call_sid, provider):
            # conv_id = uuid.uuid4().hex[:16]
            conv_id = hashlib.md5(call_sid.encode()).hexdigest()[:16]
            try:
                url = "http://172.18.0.4:8005/chat_summarizer"

                # Prepare the payload
                data = {
                    "conv_json": {"conversation": bot_context_messages},
                    "conv_id": conv_id,
                    "call_sid": call_sid,
                    "response_type": "both",
                    "provider": provider
                }
                
                logger.info(f"Summary api payload: {data}")

                # Make the API call
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        data=json.dumps(data),
                    ) as response:
                        response_data = await response.text()
                        logger.info(f"Summarize API Response: {response_data}")

                        return response_data

            except Exception as e:
                logger.error(f"Failed to call Summarize API: {str(e)}")
                return None

        agent_welcome_message = ""
        # Set up Daily transport with video/audio parameters
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=576,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,

            ),
        )
        rev_pipecat_llm = None
        context = None
        user_context = None
        assistant_context = None

        if botType == "dify-element":

            # get the dify api key
            difyApiKey = ivrDetails.get("difyApiKey")
            logger.info(f"Dify API Key: {difyApiKey}")
            agent_welcome_message = agentSettings.get("agent", {}).get("message")
            agentPrompt = agentSettings.get("agent", {}).get("prompt")
            agentPing = agentSettings.get("agent", {}).get("ping","")
            logger.info(f"Agent Welcome Message: {agent_welcome_message}")
            bot_lang = available_languages[0]

            # set the current language
            set_current_language(bot_lang)

            # get current language dify token
            language_dify_token = difyApiKey.get(bot_lang)
            logger.info(f"Language Dify Token: {language_dify_token}")
            logger.debug(f"user_details dify: {user_details}")
            session_id=user_details.get("session_id","")
            phone_number=user_details.get("recipient_phone_number","")
            print(f"phone_number:{phone_number}")
            # intelligent HR hindi assistant
            dify_llm = DifyLLMService(
                aiohttp_session=session,
                api_key=language_dify_token,
                save_bot_context= save_bot_context,
                tgt_lan=bot_lang,
                nmt_flag=nmt_flag,
                nmt_provider=nmt_provider,
                session_id=session_id,
                phone_number=phone_number
            )

            rev_pipecat_llm = dify_llm

            # based on the language set the message in messages
            if bot_lang == "hi":
                messages = [{"role": "system", "content": agentPing}]
            elif bot_lang == "en":
                messages = [{"role": "system", "content": agentPing}]
            elif bot_lang == "bn":
                messages = [{"role": "system", "content": "নমস্কার।"}]
            elif bot_lang == "as":
                messages = [{"role": "system", "content": "নমস্কাৰ।"}]
            elif bot_lang == "kn":
                messages = [{"role": "system", "content": "ಹಲೋ।"}]
            elif bot_lang == "ml":
                messages = [{"role": "system", "content": "ഹലോ।"}]
            elif bot_lang == "mr":
                messages = [{"role": "system", "content": "नमस्कार।"}]
            elif bot_lang == "or":
                messages = [{"role": "system", "content": "ନମସ୍କାର।"}]
            elif bot_lang == "ta":
                messages = [{"role": "system", "content": "வணக்கம்।"}]
            elif bot_lang == "te":
                messages = [{"role": "system", "content": "హలో।"}]
            elif bot_lang == "pa":
                messages = [{"role": "system", "content": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ।"}]
            elif bot_lang == "gu":
                messages = [{"role": "system", "content": "નમસ્તે।"}]

            tools = [
                ChatCompletionToolParam(
                    type="function",
                    function={
                        "name": "conversation_end",
                        "description": "Funnction to end the call when conversation ends or user wants to end the call or user is busy",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "call_sid": {
                                    "type": "string",
                                    "description": "The call_sid that is being passed to the function.",
                                    "default": call_sid,
                                }
                            },
                            "required": ["call_sid"],
                        },
                    },
                ),
            ]

            context = OpenAILLMContext(messages, tools)
            user_context = LLMUserContextAggregator(context)
            assistant_context = LLMAssistantContextAggregator(context)

        elif botType == "rev-chatter":
            rev_chatter_llm = ReverieChatterLLMService(
                bot_details=bot_details,
                set_current_language=set_current_language,
            )

            rev_pipecat_llm = rev_chatter_llm

            user_context = LLMUserResponseAggregator(messages)
            assistant_context = LLMAssistantResponseAggregator(messages)

        elif botType == "reverie-llm":
            bot_lang = available_languages[0]

            # set the current language
            set_current_language(bot_lang)

            agent_welcome_message = agentSettings.get("agent", {}).get("message")
            agentPrompt = agentSettings.get("agent", {}).get("prompt")
            agentPing = agentSettings.get("agent", {}).get("ping")
            
            logger.info(f"Agent Welcome Message: {agent_welcome_message}")
            
            # get name and constituency from user details
            name = user_details.get("name", "")
            conversation_id = user_details.get("conversation_id") or user_details.get("pin", "") or pin
            # constituency = user_details.get("constituency", "")
            
            # log the name and constituency
            logger.info(f"Name: {name}")
            logger.debug(f"Agent Prompt: {agentPrompt}")
            if name:
                agentPrompt = agentPrompt.replace("{name}", name)
            logger.debug(f"Agent Prompt2: {agentPrompt}")
            # if constituency:
            #     agentPrompt = agentPrompt.replace("Karnal", constituency)

            # log the agent prompt
            # logger.info(f"Agent Prompt: {agentPrompt}")

            # reverie openai llm service
            llm = OpenAILLMService(
                api_key=os.getenv("OPENAI_API_KEY"),
                # model="gpt-3.5-turbo",
                model="gpt-4o",
                # model=llmModel,
                save_bot_context=save_bot_context,
                # set_first_message_received=set_first_message_received,
                tgt_lan=bot_lang,
                nmt_flag=nmt_flag,
                nmt_provider=nmt_provider,
                conversation_id=conversation_id,
                ner_list=agentPrompt
            )
            
            
            # tools calling test
            messages = [
                {
                    "role": "system",
                    "content": agentPrompt,
                },
                {
                    "role":"assistant",
                    "content": agent_welcome_message
                },
                # {
                #     "role": "system",
                #     "content": "Please introduce yourself and provide your name.",
                # },
                {
                    "role": "system",
                    "content": agentPing,
                },
            ]

            tools = [
                ChatCompletionToolParam(
                    type="function",
                    function={
                        "name": "conversation_end",
                        "description": "Funnction to end the call when conversation ends or user wants to end the call or user is busy",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "call_sid": {
                                    "type": "string",
                                    "description": "The call_sid that is being passed to the function.",
                                    "default": call_sid,
                                },
                                "stream_sid": {
                                    "type": "string",
                                    "description": "The stream_sid that is being passed to the function.",
                                    "default": stream_sid,
                                },
                                "call_provider": {
                                    "type": "string",
                                    "description": "The call_provider that is being passed to the function.",
                                    "default": call_provider,
                                },
                            },
                            "required": ["call_sid", "stream_sid","call_provider"],
                        },
                    },
                ),
            ]

            # register conversation_end function
            llm.register_function("conversation_end", conversation_end)

            rev_pipecat_llm = llm
            context = OpenAILLMContext(messages, tools)
            user_context = LLMUserContextAggregator(context)
            assistant_context = LLMAssistantContextAggregator(context)
        
        elif botType == "reverie-azure-llm":
            bot_lang = available_languages[0]

            # set the current language
            set_current_language(bot_lang)

            agent_welcome_message = agentSettings.get("agent", {}).get("message")
            agentPrompt = agentSettings.get("agent", {}).get("prompt")
            agentPing = agentSettings.get("agent", {}).get("ping")
            
            logger.info(f"Agent Welcome Message: {agent_welcome_message}")
            
            # get name and constituency from user details
            name = user_details.get("name", "")
            constituency = user_details.get("constituency", "")
            
            # log the name and constituency
            logger.info(f"Name: {name}, Constituency: {constituency}")
            
            if name:
                agentPrompt = agentPrompt.replace("Rakesh", name)
            if constituency:
                agentPrompt = agentPrompt.replace("Karnal", constituency)

            # log the agent prompt
            # logger.info(f"Agent Prompt: {agentPrompt}")

            # reverie openai llm service
            # llm = ReverieOpenAILLMService(
            #     api_key=os.getenv("OPENAI_API_KEY"),
            #     # model="gpt-3.5-turbo",
            #     model="gpt-4o",
            #     # model=llmModel,
            #     save_bot_context=save_bot_context,
            #     # set_first_message_received=set_first_message_received,
            #     tgt_lan=bot_lang,
            #     nmt_flag=nmt_flag,
            #     nmt_provider=nmt_provider
            # )
            
            # llm = BaseOpenAILLMServiceWithCache(
            #     api_key=os.getenv("OPENAI_API_KEY"),
            #     model="gpt-4o",
            #     save_bot_context=save_bot_context,
            #     # set_first_message_received=set_first_message_received,
            #     tgt_lan=bot_lang,
            #     nmt_flag=nmt_flag,
            #     nmt_provider=nmt_provider
            # )
            
            llm = AzureOpenAILLMService(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                model="gpt-4o-mini",
                api_version="2024-02-15-preview",
                save_bot_context=save_bot_context,
                tgt_lan=bot_lang,
                nmt_flag=nmt_flag,
                nmt_provider=nmt_provider
            )
            
            # tools calling test
            messages = [
                {
                    "role": "system",
                    "content": agentPrompt,
                },
                {
                    "role":"assistant",
                    "content": agent_welcome_message
                },
                # {
                #     "role": "system",
                #     "content": "Please introduce yourself and provide your name.",
                # },
                {
                    "role": "system",
                    "content": agentPing,
                },
            ]

            tools = [
                ChatCompletionToolParam(
                    type="function",
                    function={
                        "name": "conversation_end",
                        "description": "Funnction to end the call when conversation ends or user wants to end the call or user is busy",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "call_sid": {
                                    "type": "string",
                                    "description": "The call_sid that is being passed to the function.",
                                    "default": call_sid,
                                },
                                "stream_sid": {
                                    "type": "string",
                                    "description": "The stream_sid that is being passed to the function.",
                                    "default": stream_sid,
                                },
                            },
                            "required": ["call_sid", "stream_sid"],
                        },
                    },
                ),
            ]

            # register conversation_end function
            llm.register_function("conversation_end", conversation_end)

            rev_pipecat_llm = llm
            context = OpenAILLMContext(messages, tools)
            user_context = LLMUserContextAggregator(context)
            assistant_context = LLMAssistantContextAggregator(context)
        
        elif botType == "knowledge-base":

            agent_welcome_message = agentSettings.get("agent", {}).get("message")
            agentPrompt = agentSettings.get("agent", {}).get("prompt")
            
            logger.info(f"Agent Welcome Message: {agent_welcome_message}")

            logger.debug(f"available_languages[0]: {available_languages[0]}")
            bot_lang = available_languages[0]
            # set the current language
            set_current_language(bot_lang)

            collection_name = '53f0f281-ab6a-4af1-96bd-a39734964960'

            logger.debug(f"in reverie-kb")
            llm = ReverieKnowledgeBase(
                api_key=os.getenv("OPENAI_API_KEY"),
                # model="gpt-3.5-turbo",
                model="gpt-4o",
                namespace=collection_name,
                # model=llmModel,
                save_bot_context=save_bot_context,
                # set_first_message_received=set_first_message_received,
                tgt_lan=bot_lang,
                nmt_flag=nmt_flag,
                nmt_provider=nmt_provider
            )


            # tools calling test
            messages = [
                {
                    "role": "system",
                    "content": agentPrompt,
                },
                {
                    "role":"assistant",
                    "content": agent_welcome_message
                },
                {
                    "role": "system",
                    "content": "Please introduce yourself and provide your name.",
                },
            ]

            tools = [
                ChatCompletionToolParam(
                    type="function",
                    function={
                        "name": "conversation_end",
                        "description": "Funnction to end the call when conversation ends or user wants to end the call or user is busy",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "call_sid": {
                                    "type": "string",
                                    "description": "The call_sid that is being passed to the function.",
                                    "default": call_sid,
                                },
                                "stream_sid": {
                                    "type": "string",
                                    "description": "The stream_sid that is being passed to the function.",
                                    "default": stream_sid,
                                },
                            },
                            "required": ["call_sid", "stream_sid"],
                        },
                    },
                ),
            ]

            # register conversation_end function
            llm.register_function("conversation_end", conversation_end)
            rev_pipecat_llm = llm
            context = OpenAILLMContext(messages, tools)
            user_context = LLMUserContextAggregator(context)
            assistant_context = LLMAssistantContextAggregator(context)


        else:
            rev_chatter_llm = ReverieChatterLLMService(
                bot_details=bot_details,
                set_current_language=set_current_language,
            )

            rev_pipecat_llm = rev_chatter_llm
            user_context = LLMUserResponseAggregator(messages)
            assistant_context = LLMAssistantResponseAggregator(messages)

        # Initialize text-to-speech service
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            #
            # English
            #
            voice_id="pNInz6obpgDQGcFmaJgB",
            #
            # Spanish
            #
            # model="eleven_multilingual_v2",
            # voice_id="gD1IexrzCvsXPHUuT0s3",
        )

        # Initialize LLM service
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        messages = [
            {
                "role": "system",
                #
                # English
                #
                "content": "You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself.",
                #
                # Spanish
                #
                # "content": "Eres Chatbot, un amigable y útil robot. Tu objetivo es demostrar tus capacidades de una manera breve. Tus respuestas se convertiran a audio así que nunca no debes incluir caracteres especiales. Contesta a lo que el usuario pregunte de una manera creativa, útil y breve. Empieza por presentarte a ti mismo.",
            },
        ]

        # Set up conversation context and management
        # The context_aggregator will automatically collect conversation context
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        ta = TalkingAnimation()

        #
        # RTVI events for Pipecat client UI
        #

        # This will send `user-*-speaking` and `bot-*-speaking` messages.
        rtvi_speaking = RTVISpeakingProcessor()

        # This will emit UserTranscript events.
        rtvi_user_transcription = RTVIUserTranscriptionProcessor()

        # This will emit BotTranscript events.
        rtvi_bot_transcription = RTVIBotTranscriptionProcessor()

        # This will send `metrics` messages.
        rtvi_metrics = RTVIMetricsProcessor()

        # Handles RTVI messages from the client
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                rtvi_speaking,
                rtvi_user_transcription,
                context_aggregator.user(),
                llm,
                rtvi_bot_transcription,
                tts,
                ta,
                rtvi_metrics,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )
        await task.queue_frame(quiet_frame)

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([LLMMessagesFrame(messages)])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
