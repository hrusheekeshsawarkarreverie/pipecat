import json
import aiohttp
import uvicorn
from bot import run_bot
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CALL_DETAILS = dict()

# Fetch Twilio API Details
twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")

# Initialize Twilio client
twilio_client = Client(twilio_account_sid, twilio_auth_token)

def format_stt_variables(variables):
    formatted_variables = dict()
    for variable in variables:
        formatted_variables[variable["name"]] = variable["sttConfig"]["domain"]

    # logger.debug(f"formatted variables : {formatted_variables}")
    return formatted_variables


async def get_bot_details_by_conversation_id(conversation_id):
    try:

        # if the conversation id is uuid type then it is not a valid conversation id
        if len(str(conversation_id)) == 16:
            # log that conversation id is not valid
            logger.warning(f"Conversation id is not valid: {conversation_id}")
            return None

        # log that control is comng in the else block
        logger.info(f"Getting bot details for conversation id: {conversation_id}")

        # check if the bot details are already stored in memory
        bot_details = get_bot_details_from_memory(conversation_id)
        if bot_details:
            # logger.debug(f"if Bot Detail: {bot_details}")
            return bot_details

        url = "https://sansadhak-dev.reverieinc.com/api/bot/deploy/details"
        payload = json.dumps({"conversationId": int(conversation_id)})
        headers = {
            "Content-Type": "application/json",
            "Origin": "https://sansadhak-dev.reverieinc.com",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload) as response:
                resp_obj = await response.json()
                logger.info(f"Got context for {conversation_id} => {resp_obj}")

                if "data" not in resp_obj:
                    return {
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

                # store the bot details in memory
                store_bot_details(conversation_id, response)

                return response
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        # Handle the error here
        return None


# write a sample code to store json data in memory based on the conversation id
def store_bot_details(conversation_id, bot_details):
    try:
        # store the bot details in Redis with conversation_id as the key
        redis_client.set(conversation_id, json.dumps(bot_details))
    except Exception as e:
        logger.error(f"An error occurred while storing bot details in Redis: {str(e)}")


# function to get bot details from memory based on the conversation id
# Initialize Redis client
redis_host = os.getenv("REDIS_HOST", "redis")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0)


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


# function to store user details based on unique id
def store_user_details(user_id, user_details):
    try:
        # store the user details in Redis with user_id as the key
        redis_client.set(user_id, json.dumps(user_details))
    except redis.ConnectionError as e:
        logger.error(f"Redis connection error: {str(e)}")
        # Retry logic or alternative handling can be added here
    except Exception as e:
        logger.error(f"An error occurred while storing user details in Redis: {str(e)}")


# function to get user details from memory based on the unique id
def get_user_details_from_memory(user_id):
    try:
        # log that we are getting user details from memory
        logger.info(f"Getting user details from memory for user id: {user_id}")

        # get the user details from Redis cache
        user_details = redis_client.get(user_id)
        if user_details:
            return json.loads(user_details)

        logger.warning(f"User details not found in memory for user id: {user_id}")
        return None
    except Exception as e:
        logger.error(
            f"An error occurred while getting user details from memory: {str(e)}"
        )
        return None


@app.post("/")
async def start_call():
    print("POST TwiML")
    return HTMLResponse(content=open("templates/streams.xml").read(), media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    start_data = websocket.iter_text()
    await start_data.__anext__()
    call_data = json.loads(await start_data.__anext__())
    print(call_data, flush=True)
    stream_sid = call_data["start"]["streamSid"]
    print("WebSocket connection accepted")
    await run_bot(websocket, stream_sid)


####Twilio implementaion
# api to make outbound call
@app.post("/bulk_call")
async def make_bulk_call(request: Request):
    try:
        call_details_list = await request.json()

        for call_details in call_details_list:

            # log the call details
            logger.info(f"Call details: {call_details}")

            conversation_id = call_details["conversation_id"]
            recipient_phone_number = call_details["recipient_phone_number"]
            # name = call_details["name"]
            # constituency = call_details["constituency"]
            name = call_details.get("name")
            logger.debug(f"NAME: {name}")

            constituency = call_details.get("constituency")  # Returns None if missing
            session_id = call_details.get("session_id")
            # generate unique id for the user with uuid
            user_id_pin = str(uuid.uuid4())[:16]

            # save the user details in memory
            user_details = {
                "name": name,
                "constituency": constituency,
                "recipient_phone_number": recipient_phone_number,
                "conversation_id": conversation_id,
                "session_id": session_id,
            }

            # call the function to store the user details
            store_user_details(user_id_pin, user_details)

            # call the function to make the call
            make_call(user_id_pin, recipient_phone_number)

        return PlainTextResponse("done", status_code=200)

    except Exception as e:
        logger.info(f"Exception occurred in make_call: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.websocket("/ws/{pin}")
async def websocket_endpoint(websocket: WebSocket, pin: str):
    try:
        await websocket.accept()

        user_details = None
        bot_details = None

        bot_details_start_time = time.time()

        # log that we are getting data based on the conversation id
        logger.info(f"Getting bot details for conversation id: {pin}")
        bot_details = await get_bot_details_by_conversation_id(pin)

        # if bot_details is not found, then get the user details based on the pin
        if not bot_details:
            user_details = get_user_details_from_memory(pin)
            logger.debug(f"User details: {user_details}")

            # if user_details is found, get the conversation_id from user_details
            if user_details:
                conversation_id = user_details.get("conversation_id", "")

                # log conversation_id from user_details
                logger.debug(f"Conversation id from user details: {conversation_id}")

                get_bot_details_start_time = time.time()
                bot_details = await get_bot_details_by_conversation_id(conversation_id)
                logger.info(
                    "get bot details Time consuming: {:.4f}s".format(
                        time.time() - get_bot_details_start_time
                    )
                )
                # logger.debug(f"Bot details: {bot_details}")

                # add the user details to the bot details
                name = user_details.get("name", "")
                constituency = user_details.get("constituency", "")
                recipient_phone_number = user_details.get("recipient_phone_number", "")
                session_id = user_details.get("session_id", "")
                bot_details["user_details"] = {
                    "name": name,
                    "constituency": constituency,
                    "recipient_phone_number": recipient_phone_number,
                    "conversation_id": conversation_id,
                    "session_id": session_id,
                }

        # log the bot details
        logger.debug(f"Bot details after modification: {bot_details}")

        logger.info(
            "bot details Time consuming: {:.4f}s".format(
                time.time() - bot_details_start_time
            )
        )

        start_data = websocket.iter_text()
        await start_data.__anext__()
        call_data = json.loads(await start_data.__anext__())
        logger.info(call_data)
        stream_sid = call_data["start"]["streamSid"]
        # get callSid
        call_sid = call_data["start"]["callSid"]
        logger.info("WebSocket connection accepted")
        await run_bot(websocket, stream_sid, call_sid, bot_details)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        # Handle the error here
    
def make_call(user_id_pin: str, recipient_phone_number: str):
    if not user_id_pin:
        raise HTTPException(status_code=404, detail="Pin not provided")

    if not recipient_phone_number:
        raise HTTPException(
            status_code=404, detail="Recipient phone number not provided"
        )

    stream_url = os.getenv("APPLICATION_BASE_URL")
    xml_content = open("templates/streams_with_pin.xml").read()
    xml_content_with_url = xml_content.replace("{{ base_url }}", stream_url)
    xml_content_with_url_with_pin = xml_content_with_url.replace(
        "{{ pin }}", user_id_pin
    )

    try:
        call = twilio_client.calls.create(
            to=recipient_phone_number,
            from_=twilio_phone_number,
            twiml=xml_content_with_url_with_pin,
            method="POST",
            record=True,
        )
    except Exception as e:
        logger.info(f"make_call exception: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
