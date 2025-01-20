#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVI Bot Server Implementation.

This FastAPI server manages RTVI bot instances and provides endpoints for both
direct browser access and RTVI client connections. It handles:
- Creating Daily rooms
- Managing bot processes
- Providing connection credentials
- Monitoring bot status

Requirements:
- Daily API key (set in .env file)
- Python 3.10+
- FastAPI
- Running bot implementation
"""

import argparse
import os
import subprocess
from contextlib import asynccontextmanager
from typing import Any, Dict
from loguru import logger
import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import redis
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams
import json
# Load environment variables from .env file
load_dotenv(override=True)

# Maximum number of bot instances allowed per room
MAX_BOTS_PER_ROOM = 1

# Dictionary to track bot processes: {pid: (process, room_url)}
bot_procs = {}

# Store Daily API helpers
daily_helpers = {}


def cleanup():
    """Cleanup function to terminate all bot processes.

    Called during server shutdown.
    """
    for entry in bot_procs.values():
        proc = entry[0]
        proc.terminate()
        proc.wait()


def get_bot_file():
    bot_implementation = os.getenv("BOT_IMPLEMENTATION", "openai").lower().strip()
    # If blank or None, default to openai
    if not bot_implementation:
        bot_implementation = "openai"
    if bot_implementation not in ["openai", "gemini","simple"]:
        raise ValueError(
            f"Invalid BOT_IMPLEMENTATION: {bot_implementation}. Must be 'openai' or 'gemini'"
        )
    return f"bot-{bot_implementation}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager that handles startup and shutdown tasks.

    - Creates aiohttp session
    - Initializes Daily API helper
    - Cleans up resources on shutdown
    """
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    cleanup()


# Initialize FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
redis_host = os.getenv("REDIS_HOST", "redis")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

def store_bot_details(conversation_id, bot_details):
    try:
        # If bot_details is already a string, store it directly
        if isinstance(bot_details, str):
            redis_client.set(f"bot_details:{conversation_id}", bot_details)
        else:
            # If it's a dictionary or any other type, convert to JSON string
            redis_client.set(f"bot_details:{conversation_id}", json.dumps(bot_details))
        logger.info(f"Successfully stored bot details for conversation_id: {conversation_id}")
    except Exception as e:
        logger.error(f"An error occurred while storing bot details in Redis: {str(e)}")


def get_bot_details_from_memory(conversation_id):
    try:
        # log that we are getting bot details from memory
        logger.info(
            f"Getting bot details from memory for conversation id: {conversation_id}"
        )

        # get the bot details from Redis cache
        bot_details = redis_client.get(conversation_id)

        # log the bot_details
        logger.debug(f"Bot Details: {bot_details}")

        if bot_details:
            return json.loads(bot_details)

        logger.warning(
            f"Bot details not found in memory for conversation id: {conversation_id}"
        )
        return None
    except Exception as e:
        logger.error(
            f"An error occurred while getting bot details from memory: {str(e)}"
        )
        return None

def format_stt_variables(variables):
    formatted_variables = dict()
    for variable in variables:
        formatted_variables[variable["name"]] = variable["sttConfig"]["domain"]

    # logger.debug(f"formatted variables : {formatted_variables}")
    return formatted_variables

async def get_bot_details_by_conversation_id(conversation_id):
    try:
        # check if the bot details are already stored in memory
        bot_details = get_bot_details_from_memory(conversation_id)
        if bot_details:
            return bot_details

        url = "https://sansadhak-dev.reverieinc.com/api/bot/deploy/details"
        payload = json.dumps({"conversationId": int(conversation_id)})
        headers = {
            "Content-Type": "application/json",
            "Origin": "https://sansadhak-dev.reverieinc.com",
        }

        async with aiohttp.ClientSession() as session:
            print(f"url: {url}, headers: {headers}, payload: {payload}")
            async with session.post(url, headers=headers, data=payload) as response:
                resp_obj = await response.json()
                logger.info(f"Got context for {conversation_id} => {resp_obj}")

                if "data" not in resp_obj:
                    response = {
                        "api_details": {
                            "REV-APP-ID": "com.domain",
                            "REV-APPNAME": "nlu",
                            "REV-API-KEY": "732407ffce16f9362f9f0eeb2b5aa5758cd09039",
                            "PROJECT": "Eastman Auto",
                            "MODEL": "eastman_model",
                            "SUPPORT_PROJECT": "Eastman Auto",
                            "SUPPORT_MODEL": "eastman_model",
                            "TEMPLATE": "Eastman Auto_1720609506.0128822",
                            "available_languages": ["en", "hi"],
                        },
                        "stt_variables": {},
                        "tts_variables": {},
                        "selectLanguage": False,
                    }
                    # Store as JSON string
                    store_bot_details(conversation_id, json.dumps(response))
                    return response

                languages = resp_obj["data"]["testDetails"].get("languages", [])
                PROJECT = resp_obj["data"]["testDetails"].get("projectName", "")
                MODEL = resp_obj["data"]["testDetails"].get("modelName", "")
                TEMPLATE = resp_obj["data"]["testDetails"].get("templateName", "")

                stt_variables = format_stt_variables(
                    resp_obj["data"].get("sttVariablesInfo", [])
                )

                tts_variables = {}
                if "ttsProvider" in resp_obj["data"]["testDetails"]:
                    pass

                if "ttsSettings" in resp_obj["data"]["testDetails"]:
                    pass

                # log some message here
                logger.info(f"Languages: {languages}")
                selectLanguage = False
                # try:
                #     selectLanguage = (
                #         resp_obj.get("data", {})
                #         .get("botStyle", {})
                #         .get("style", {})
                #         .get("selectLanguage", True)
                #     )
                # except Exception as e:
                #     logger.error(f"An error occurred while fetching selectLanguage: {str(e)}")
                #     selectLanguage = True  # Default value if there's an error

                # log some message here
                logger.info(f"Select Language: {selectLanguage}")

                ivrDetails = resp_obj.get("data", {}).get("ivrDetails", {})
                providerData = resp_obj.get("data", {}).get("providerData", {})

                # log the ivr details
                logger.info(f"IVR details: {ivrDetails}")

                logger.info(f"PROVIDER DATA: {providerData}")

                agentSettings = resp_obj.get("data", {}).get("agentSettings", {})
                logger.info(f"Agent settings: {agentSettings}")

                response = {
                    "api_details": {
                        "REV-APP-ID": "com.domain",
                        "REV-APPNAME": "nlu",
                        "REV-API-KEY": "732407ffce16f9362f9f0eeb2b5aa5758cd09039",
                        "PROJECT": PROJECT,
                        "MODEL": MODEL,
                        "SUPPORT_PROJECT": PROJECT,
                        "SUPPORT_MODEL": MODEL,
                        "TEMPLATE": TEMPLATE,
                        "available_languages": languages,
                    },
                    "stt_variables": stt_variables,
                    "tts_variables": tts_variables,
                    "selectLanguage": selectLanguage,
                    "ivrDetails": ivrDetails,
                    "providerData": providerData,
                    "agentSettings": agentSettings,
                }

                # Store as JSON string
                store_bot_details(conversation_id, json.dumps(response))
                return response

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return None

async def create_room_and_token() -> tuple[str, str]:
    """Helper function to create a Daily room and generate an access token.

    Returns:
        tuple[str, str]: A tuple containing (room_url, token)

    Raises:
        HTTPException: If room creation or token generation fails
    """
    room = await daily_helpers["rest"].create_room(DailyRoomParams())
    if not room.url:
        raise HTTPException(status_code=500, detail="Failed to create room")

    token = await daily_helpers["rest"].get_token(room.url)
    if not token:
        raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room.url}")

    return room.url, token


@app.get("/")
async def start_agent(request: Request):
    """Endpoint for direct browser access to the bot.

    Creates a room, starts a bot instance, and redirects to the Daily room URL.

    Returns:
        RedirectResponse: Redirects to the Daily room URL

    Raises:
        HTTPException: If room creation, token generation, or bot startup fails
    """
    print("Creating room")
    room_url, token = await create_room_and_token()
    print(f"Room URL: {room_url}")

    # Check if there is already an existing process running in this room
    num_bots_in_room = sum(
        1 for proc in bot_procs.values() if proc[1] == room_url and proc[0].poll() is None
    )
    if num_bots_in_room >= MAX_BOTS_PER_ROOM:
        raise HTTPException(status_code=500, detail=f"Max bot limit reached for room: {room_url}")

    # Spawn a new bot process
    try:
        bot_file = get_bot_file()
        proc = subprocess.Popen(
            [f"python3 -m {bot_file} -u {room_url} -t {token}"],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        bot_procs[proc.pid] = (proc, room_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    return RedirectResponse(room_url)


@app.post("/connect/{conversation_id}")
async def rtvi_connect(conversation_id: int) -> Dict[Any, Any]:
    print("Creating room for RTVI connection")
    room_url, token = await create_room_and_token()
    print(f"Room URL: {room_url}")
    if await get_bot_details_by_conversation_id(conversation_id):
        
        print(f"bot details: True")


    # Start the bot process
    try:
        bot_file = get_bot_file()
        proc = subprocess.Popen(
            [f"python3 -m {bot_file} -u {room_url} -t {token} -b {conversation_id}"],
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        bot_procs[proc.pid] = (proc, room_url)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    # Return the authentication bundle in format expected by DailyTransport
    return {"room_url": room_url, "token": token}


@app.get("/status/{pid}")
def get_status(pid: int):
    """Get the status of a specific bot process.

    Args:
        pid (int): Process ID of the bot

    Returns:
        JSONResponse: Status information for the bot

    Raises:
        HTTPException: If the specified bot process is not found
    """
    # Look up the subprocess
    proc = bot_procs.get(pid)

    # If the subprocess doesn't exist, return an error
    if not proc:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {pid} not found")

    # Check the status of the subprocess
    status = "running" if proc[0].poll() is None else "finished"
    return JSONResponse({"bot_id": pid, "status": status})


if __name__ == "__main__":
    import uvicorn

    # Parse command line arguments for server configuration
    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "7860"))

    parser = argparse.ArgumentParser(description="Daily Storyteller FastAPI server")
    parser.add_argument("--host", type=str, default=default_host, help="Host address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Reload code on change")

    config = parser.parse_args()

    # Start the FastAPI server
    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )
