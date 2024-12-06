#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import io
import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional
import redis

import aiohttp
import httpx
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMUpdateSettingsFrame,
    StartInterruptionFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    URLImageRawFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import ImageGenService, LLMService, TTSService
import os
try:
    from openai import (
        NOT_GIVEN,
        AsyncOpenAI,
        AsyncStream,
        BadRequestError,
        DefaultAsyncHttpxClient,
    )
    from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenAI, you need to `pip install pipecat-ai[openai]`. Also, set `OPENAI_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")
from qdrant_client import QdrantClient,models
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import http.client
import re
import time
from pipecat.services.nmt import NMTService


_embedder = SentenceTransformer("l3cube-pune/indic-sentence-bert-nli")

ValidVoice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

VALID_VOICES: Dict[str, ValidVoice] = {
    "alloy": "alloy",
    "echo": "echo",
    "fable": "fable",
    "onyx": "onyx",
    "nova": "nova",
    "shimmer": "shimmer",
}

# set the qdrant client
if os.getenv("VECTOR_STORE", "redis").lower() == "qdrant":
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        _qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        logger.debug(
            f"Qdrant client initialized with host: {qdrant_host}, port: {qdrant_port}"
        )

        # Optional: Verify connection
        _qdrant_client.get_collections()
        logger.debug("Qdrant client connection verified.")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant client: {e}", exc_info=True)
        _qdrant_client = None
else:
    _qdrant_client = None

# semantic text prefix for caching
semantic_response_prefix = "semantic-response"
semantic_vector_prefix = "semantic-vector"

class OpenAIUnhandledFunctionException(Exception):
    pass


class BaseOpenAILLMService(LLMService):
    """This is the base for all services that use the AsyncOpenAI client.

    This service consumes OpenAILLMContextFrame frames, which contain a reference
    to an OpenAILLMContext frame. The OpenAILLMContext object defines the context
    sent to the LLM for a completion. This includes user, assistant and system messages
    as well as tool choices and the tool, which is used if requesting function
    calls from the LLM.
    """

    class InputParams(BaseModel):
        frequency_penalty: Optional[float] = Field(
            default_factory=lambda: NOT_GIVEN, ge=-2.0, le=2.0
        )
        presence_penalty: Optional[float] = Field(
            default_factory=lambda: NOT_GIVEN, ge=-2.0, le=2.0
        )
        seed: Optional[int] = Field(default_factory=lambda: NOT_GIVEN, ge=0)
        temperature: Optional[float] = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=2.0)
        # Note: top_k is currently not supported by the OpenAI client library,
        # so top_k is ignore right now.
        top_k: Optional[int] = Field(default=None, ge=0)
        top_p: Optional[float] = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=1.0)
        max_tokens: Optional[int] = Field(default_factory=lambda: NOT_GIVEN, ge=1)
        max_completion_tokens: Optional[int] = Field(default_factory=lambda: NOT_GIVEN, ge=1)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        model: str,
        api_key=None,
        base_url=None,
        tgt_lan: str,
        nmt_flag: str,
        nmt_provider: str,
        conversation_id: str,
        ner_list: str,
        redis_host: str = os.getenv('REDIS_HOST', 'redis'),
        redis_port: int = int(os.getenv('REDIS_PORT', 6379)),
        redis_db: int = 0,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._settings = {
            "frequency_penalty": params.frequency_penalty,
            "presence_penalty": params.presence_penalty,
            "seed": params.seed,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "max_completion_tokens": params.max_completion_tokens,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
        self.set_model_name(model)
        self._client = self.create_client(api_key=api_key, base_url=base_url, **kwargs)
        self._save_bot_context = kwargs.get("save_bot_context")
        self._tgt_lan = tgt_lan
        self._src_lan = "en"
        self._nmt_flag = nmt_flag
        self._nmt_provider = nmt_provider
        self._is_start_msg_sent = False
        self._frame = ""
        self._processed_text = ""
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis_db = redis_db
        self._cache_type = os.getenv("cache_type", "").lower()
        self._is_semantic_caching_enabled = self._cache_type == "semantic"
        self._is_text_caching_enabled = self._cache_type == "text"
        if self._is_semantic_caching_enabled:
            self._embedder = _embedder
        self._redis_client = self.create_redis_client()
        self._qdrant_client = _qdrant_client
        self._collection_name = conversation_id
        self._vector_store = os.getenv("VECTOR_STORE", "redis").lower()
        self._ner_check = os.getenv("NER_CHECK", "false").lower()
        self._ner_list=[]
        # Ensure the collection exists
        if self._vector_store == "qdrant":

            try:
                logger.debug(f"collections: {self._qdrant_client.get_collections()}")
                self._qdrant_client.get_collection(self._collection_name)
            except Exception as e:
                logger.info(f"Collection {self._collection_name} not found. Creating a new one.")
                self._qdrant_client.create_collection(
                collection_name=self._collection_name,
        
                vectors_config=models.VectorParams(
                    # size=384,
                    size=768,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
                )
            
            ner_key = str(conversation_id)+"_ner_cache"
            retrieved_str = self._redis_client.get(ner_key)
            if retrieved_str != None:
                retrieved_str = retrieved_str.decode("utf-8")
                self._ner_list = retrieved_str.split(",")
                logger.debug(f"initialised self._ner_list from redis: {self._ner_list}")
            else:
                try:
                    if self._ner_check=="true":
                        self._ner_list = self.get_ner(ner_list)
                        logger.debug(f"initialised self._ner_list: {self._ner_list}")
                        if self._ner_list:
                            self.get_transliteration()
                            logger.debug(f"self._ner_list after transliteration: {self._ner_list}")
                            list = ",".join(self._ner_list)
                            self._redis_client.set(ner_key, list)
                    else:
                        logger.debug(f"empty ner list")
                        self._ner_list = []

                except Exception as e:
                    logger.debug(f"Error in transliteration or saving list to redis: {e}")
                    self._ner_list = []



    def create_client(self, api_key=None, base_url=None, **kwargs):
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100, max_connections=1000, keepalive_expiry=None
                )
            ),
        )

    def can_generate_metrics(self) -> bool:
        return True
    
    def create_redis_client(self):
        return redis.StrictRedis(host=self._redis_host, port=self._redis_port, db=self._redis_db)

    def can_generate_metrics(self) -> bool:
        return True

    # Utility function to compute cosine similarity between two vectors
    def compute_cosine_similarity(self, vec1, vec2):
        return cosine_similarity([vec1], [vec2])[0][0]
    
    
    def store_response_in_redis(self, conversation_context: str, response: str):
        hash_key = self.generate_hash_key(conversation_context)
        response_key = f"{semantic_response_prefix}:{hash_key}.txt"
        if not self._redis_client.exists(response_key):
            self._redis_client.set(response_key, response)
    
    # Store embeddings in Redis under a specific prefix
    def store_embedding_in_redis(self, conversation_context: str, response: str):
        hash_key = self.generate_hash_key(conversation_context)
        key = f"{semantic_vector_prefix}:{hash_key}"
        # Check if the embedding is already stored in Redis
        if not self._redis_client.exists(key):
            # Get the embedding for the conversation context
            embedding = self.generate_embedding(conversation_context)
            self._redis_client.set(key, embedding.tobytes())
        
        # store the response in redis
        self.store_response_in_redis(conversation_context, response)

        
        
    def store_embedding_in_qdrant(self, conversation_context: str, response: str, user_message: str):
        hash_key_string = conversation_context + user_message
        hash_key = self.generate_hash_key(hash_key_string)
        
        # Log the conversation context and response
        logger.debug(f"Storing embedding in Qdrant for context: {conversation_context}")

        # Check if the data is already present based on the hash_key
        existing_data = self._qdrant_client.retrieve(
            collection_name=self._collection_name,
            ids=[hash_key]
        )

        if not existing_data:
            logger.debug(f"conversation_context: {conversation_context}, conversation_context_type: {type(conversation_context)},user_message: {user_message}, type: {type(user_message)} ")
            # Store the response in Qdrant
            self._qdrant_client.upsert(
            collection_name=self._collection_name,

            points=[
                models.PointStruct(
                    id=hash_key,
                    vector=[self.generate_embedding(conversation_context).tolist(),self.generate_embedding(user_message).tolist()],
                    payload= {
                        'conversation_context': conversation_context,
                        'user_message':user_message,
                        'response': response
                    }
                )
            ],
            )
            
            # store the response in redis so that later we can retrieve it based on the hash key
            # logger.debug(f"storing in Redis: {conversation_context}, response: {response}")
            # self.store_response_in_redis(conversation_context, response)
        
        else:
            logger.debug(f"Data already present for hash_key: {hash_key}")
            

    # Generate embeddings for the redis db
    def generate_embedding_for_redis(self, text: str) -> np.ndarray:
        try:
            # new_embedding = model.encode([text])[0]
            return self._embedder.encode([text])[0]
        except Exception as e:
            logger.error(f"Error generating embedding for Redis: {e}")
            return np.array([], dtype=np.float32)
        
    def generate_embedding_for_qdrant(self, text: str) -> np.ndarray:
        try:
            embedding = self._embedder.encode([text])[0]
            embedding = np.array(embedding, dtype=np.float32)  # Ensure embedding is a numpy array with float32 type
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.array([], dtype=np.float32)

    def generate_embedding(self, text: str) -> np.ndarray:
        try:
            if self._vector_store == "redis":
                embedding = self.generate_embedding_for_redis(text)
            elif self._vector_store == "qdrant":
                embedding = self.generate_embedding_for_qdrant(text)
            else:
                logger.error(f"Unknown vector store: {self._vector_store}")
                return np.array([], dtype=np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.array([], dtype=np.float32)
        
    # Get the response from Redis based on the key hash
    def get_cached_response_from_redis(self, key_hash: str) -> Optional[str]:
        
        key_hash = key_hash.replace("-", "")
        
        response_key = f"{semantic_response_prefix}:{key_hash}.txt"
        cached_response = self._redis_client.get(response_key)
        
        # log the cached response
        logger.debug(f"Cached response: {cached_response}")
        
        if cached_response:
            return cached_response.decode('utf-8')
        return None
        
    # Retrieve embeddings from Redis prefix and check cosine similarity
    def fetch_redis_cached_response(self, sentence: str):
        try:
            prefix_pattern = f"{semantic_vector_prefix}:*"
            stored_keys = self._redis_client.keys(prefix_pattern)  # Get all stored keys with the prefix
            new_embedding = self.generate_embedding(sentence)  # Generate embedding for new sentence

            if new_embedding.size == 0:
                logger.error("Generated embedding is empty.")
                return None

            for key in stored_keys:
                stored_embedding = np.frombuffer(self._redis_client.get(key), dtype=np.float32)

                if stored_embedding.size == 0:
                    logger.error(f"Stored embedding for key {key} is empty.")
                    continue

                # Compute cosine similarity with stored embeddings
                similarity = self.compute_cosine_similarity(stored_embedding, new_embedding)

                if similarity > 0.9:  # Define a threshold for similarity (0.9 here)
                    key_hash = key.decode('utf-8').split(":", 1)[1]  # Extract the sentence part from the key
                    
                    logger.debug(f"Key hash: {key_hash}")
                    
                    # Return the cached response based on the key hash
                    return self.get_cached_response_from_redis(key_hash)
                    
        except Exception as e:
            logger.error(f"Error in fetch_redis_cached_response: {e}")
        return None
        
    def get_semantic_cached_response(self, context: OpenAILLMContext) -> Optional[str]:
        try:
            messages = context.get_messages()
            if len(messages) < 1:
                return None    
            
            if messages[-1]['role'] == "system":
                conversation_context, user_message = messages[0]["content"] , "nulll"
            else:
                conversation_context, user_message = self.get_conversation_context(messages)
            
            # log the vectore store
            logger.debug(f"Vector store: {self._vector_store}")
            
            if self._vector_store == "redis":
                # get cached response based on embedding from Redis
                cached_response = self.fetch_redis_cached_response(conversation_context,user_message)
            elif self._vector_store == "qdrant":
                # get cached response based on embedding from Qdrant
                cached_response = self.fetch_qdrant_cached_response(conversation_context,user_message)
            else:
                logger.error(f"Unknown vector store: {self._vector_store}")
                cached_response = None

            logger.debug(f"cached response {cached_response}")
            
            return cached_response

        except Exception as e:
            logger.error(f"Error in get_semantic_cached_response: {e}")
        return None


    def get_conversation_context(self, messages: List[dict]) -> str:
            # """Get the conversation context for caching."""

        try: 
            # Start from the last `user` message, going backwards
            last_user_index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]['role'] == 'user'), None)
            
            # Collect `assistant` messages immediately before the last `user` message
            assistant_messages = []
            for i in range(last_user_index - 1, -1, -1):
                if messages[i]['role'] == 'assistant':
                    assistant_messages.append(messages[i]['content'])
                else:
                    break  # Stop if we encounter a non-assistant message

            # Reverse the collected messages to maintain original order
            assistant_messages.reverse()
            return ' '.join(assistant_messages), messages[last_user_index]['content']
        
        except Exception as e:
            logger.debug(f"l message: {messages}")
            return messages[-1]["content"], "nulll"

    def generate_hash_key(self, conversation_context: str) -> str:
        return hashlib.md5(conversation_context.encode()).hexdigest()
    
    def fetch_redis_text_cached_response(self, redis_client, response_key):
        return redis_client.get(response_key)
    
    def fetch_qdrant_cached_response(self, conversation_context: str, user_response: str) -> Optional[str]:
        try:
            # Log the conversation context
            logger.debug(f"fetch_qdrant_cached_response:: Conversation context: {conversation_context}, user_response: {user_response}")
            
            # Search for the cached response in Qdrant
            search_result = self._qdrant_client.query_points(
                collection_name=self._collection_name,
                query=[
                    self.generate_embedding(conversation_context).tolist(),
                    self.generate_embedding(user_response).tolist()
                ],
                limit=1  # We only need the top result
            )
            
            logger.debug(f"Search result qdrant baseopenai: {search_result.points}")

            # Check if the similarity score is more than 90 percent
            if search_result and len(search_result.points) > 0:
                top_result = search_result.points[0]
                similarity_score = top_result.score  # Access the score attribute
                if similarity_score >= 1.80:
                    # get the id of the record
                    key_hash = top_result.id
                    
                    # get the response based on the id from the redis
                    logger.debug(f"Key hash: {key_hash}")
                    
                    # Return the cached response based on the key hash
                    # search_result = self.get_cached_response_from_redis(key_hash)
                    search_result = top_result.payload["response"]
                    
                    # Log the search result
                    logger.debug(f"Search result: {search_result}")
                    
                    return search_result
        except Exception as e:
            logger.error(f"Error in fetch_qdrant_cached_response: {e}")
        return None

    def get_text_cached_response(self, context: OpenAILLMContext) -> Optional[str]:
        """
        Retrieve a cached text response based on the provided context.
        
        Args:
            context (OpenAILLMContext): The context for which the cached response is to be retrieved.
        
        Returns:
            Optional[str]: The cached response if found, otherwise None.
        """
        try:
            # Get the list of messages from the context
            messages = context.get_messages()
            
            # If there are no messages, return None
            if len(messages) < 1:
                return None

            # Use the get_conversation_context function to get the conversation context
            if messages[-1]['role'] == "system":
                conversation_context, user_message = messages[0]["content"] , "nulll"
            else:
                conversation_context, user_message = self.get_conversation_context(messages)
                
            # Log the conversation context for debugging purposes
            logger.debug(f"Conversation context: {conversation_context}")
                            
            # Generate a hash key for the conversation context
            hash_key = self.generate_hash_key(conversation_context)
            
            # Create the response key using the hash key
            response_key = f"{hash_key}.txt"
            
            # Log the response key for debugging purposes
            logger.debug(f"Response key: {response_key}")

            # Fetch the cached response from Redis using the response key
            cached_response = self.fetch_redis_text_cached_response(self._redis_client, response_key)
            
            # If a cached response is found, log it and return the decoded response
            if cached_response:
                logger.debug("Serving response from cache")
                return cached_response.decode('utf-8')
        except Exception as e:
            # Log any exceptions that occur during the process
            logger.error(f"Error in get_text_cached_response: {e}")
        
        # Return None if no cached response is found or an error occurs
        return None

    def get_cached_response(self, context: OpenAILLMContext) -> Optional[str]:
        """
        Retrieve a cached response based on the provided context.
        This method checks if semantic caching is enabled and attempts to 
        retrieve a cached response using semantic caching first. If a 
        semantic cached response is not found, it falls back to text 
        caching. If semantic caching is not enabled, it directly attempts 
        to retrieve a cached response using text caching.
        Args:
            context (OpenAILLMContext): The context for which the cached 
            response is to be retrieved.
        Returns:
            Optional[str]: The cached response if found, otherwise None.
        """
        # Implementation here
        if self._is_semantic_caching_enabled:
            text_response = self.get_semantic_cached_response(context)
            if text_response:
                return text_response
            
            return self.get_semantic_cached_response(context)
        elif self._is_text_caching_enabled:
            return self.get_text_cached_response(context)
        return None
    
    def store_embedding(self, conversation_context: str, response: str,user_message: str):
        """
        Store the embedding of the conversation context and the response in the appropriate vector store.
        
        Args:
            conversation_context (str): The context of the conversation.
            response (str): The response to be stored.
        """
        if self._vector_store == "redis":
            # Store the embedding and response in Redis
            self.store_embedding_in_redis(conversation_context, response, user_message)
        elif self._vector_store == "qdrant":
            # Store the embedding and response in Qdrant
            self.store_embedding_in_qdrant(conversation_context, response, user_message)
        else:
            # Log an error if the vector store is unknown
            logger.error(f"Unknown vector store: {self._vector_store}")
    

    def cache_response(self, context: OpenAILLMContext, response: str):
        if not self._is_semantic_caching_enabled and not self._is_text_caching_enabled:
            return

        messages = context.get_messages()
        if len(messages) < 1:
            return
        if messages[-1]['role'] == "system":
            conversation_context, user_message = messages[0]["content"] , "nulll"
        else:
            conversation_context, user_message = self.get_conversation_context(messages)
        
        # log that the response is being cached along with the conversation context and response
        logger.debug(f"Caching response for context: {conversation_context}")
        logger.debug(f"user message: {user_message}")
        logger.debug(f"Response: {response}")
        
        if self._is_semantic_caching_enabled:
            logger.debug(f"Storing new embedding for sentence: {conversation_context} with prefix: llm")

            # If no similar sentence is found, store the new embedding and return None
            self.store_embedding(conversation_context, response,user_message)
                
        elif self._is_text_caching_enabled:
            hash_key = self.generate_hash_key(conversation_context)
            self._redis_client.set(f"{hash_key}.txt", response)

    def get_transliteration(self):

        
        conn = http.client.HTTPSConnection("revapi.reverieinc.com")
        payload = json.dumps({
        "data": self._ner_list,
        "isBulk": False,
        "ignoreTaggedEntities": False,
        "convertNumber": "false"
        })
        headers = {
        'Content-Type': 'application/json',
        'REV-API-KEY': '172c5bb5af18516905473091fd58d30afe740b3f',
        'REV-APP-ID': 'rev.transliteration',
        'REV-APPNAME': 'transliteration',
        'src_lang': 'en',
        'tgt_lang': self._tgt_lan,
        'domain': '1'
        }
        conn.request("POST", "/", payload, headers)
        res = conn.getresponse()
        data = res.read()
        response = data.decode("utf-8")
        response=json.loads(response)
        response = response["responseList"]
        logger.debug(f"response: {response}, restype: {type(response)}")
        for res in response:
            transliterated_text = res["outString"][0]

            logger.debug(f"transliterated_text: {transliterated_text}")

            self._ner_list.append(transliterated_text)
        

    def get_ner(self, text: str):
        """
        Check if the full response  contains entities, cache only those resposnse which do not have entities

        Args:
            full_response: (str): llm response

        """

        text = re.sub(r'[\n"\']', '', text)
        logger.debug(f"Text before NER: {text}")

        conn = http.client.HTTPConnection("revapi.reverieinc.com")
        payload = json.dumps({
            "text": text,
            "language": self._tgt_lan,
            "entity_recognition": {
                "entity_types": [
                    "name"
                ]
            }
        })
        headers = {
            'Content-Type': 'application/json',
            'REV-API-KEY': '78a6f3fd7d809177b817e2aa4b7dbc0494c2d89d',
            'REV-APP-ID': 'com.nlpapi',
            'REV-APPNAME': 'text-analysis'
        }
        conn.request("POST", "/api/v2/text-analyse?detect_entities=true", payload, headers)
        res = conn.getresponse()
        data = res.read()
        # data.decode("utf-8")
        # print(data)
        decoded_data = data.decode("utf-8")
        parsed_data = json.loads(decoded_data)
        print(decoded_data)
        # Pretty print the JSON data for readability
        # print(json.dumps(parsed_data, ensure_ascii=False, indent=4))
        ner_values= parsed_data.get("results").get("entity_recognition").get("entities").get("name")

        logger.debug(f"Name NER: {ner_values}")

        return ner_values

    def ner_check(self, full_response: str):
        ner_present=self.get_ner(full_response)
        logger.debug(f"ner_check: {ner_present}")
        logger.debug(f"self._ner_list: {self._ner_list}")

        if ner_present !=None:
            if ner_present[0] in self._ner_list:
                return False
            else:
                logger.debug(f"PII present do not cache")
                return True
        return False

    async def get_chat_completions(
        self, context: OpenAILLMContext, messages: List[ChatCompletionMessageParam]
    ) -> AsyncStream[ChatCompletionChunk]:
        # logger.debug(f"get_chat_completions context: {context}, messages: {messages}")
        if self._vector_store:
            cache_response_time=time.time()
            cached_response = self.get_cached_response(context)
            logger.info("Cache response time: {:.4f}s".format(time.time() - cache_response_time))
            if cached_response:            
                # log the cached response
                logger.debug(f"Cached response: {cached_response}")
                
                async def cached_response_stream():
                    yield ChatCompletionChunk(
                        id="cached_response",
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=self._model,
                        choices=[{"index": 0, "delta": {"content": cached_response}}]
                    )
                return cached_response_stream()
            
        params = {
            "model": self.model_name,
            "stream": True,
            "messages": messages,
            "tools": context.tools,
            "tool_choice": context.tool_choice,
            "stream_options": {"include_usage": True},
            "frequency_penalty": self._settings["frequency_penalty"],
            "presence_penalty": self._settings["presence_penalty"],
            "seed": self._settings["seed"],
            "temperature": self._settings["temperature"],
            "top_p": self._settings["top_p"],
            "max_tokens": self._settings["max_tokens"],
            "max_completion_tokens": self._settings["max_completion_tokens"],
        }

        params.update(self._settings["extra"])

        chunks = await self._client.chat.completions.create(**params)
        return chunks

    async def _stream_chat_completions(
        self, context: OpenAILLMContext
    ) -> AsyncStream[ChatCompletionChunk]:
        logger.debug(f"Generating chat: {context.get_messages_for_logging()}")

        messages: List[ChatCompletionMessageParam] = context.get_messages()

        # save the bot context messages to the context
        self._save_bot_context(messages)

        # base64 encode any images
        for message in messages:
            if message.get("mime_type") == "image/jpeg":
                encoded_image = base64.b64encode(message["data"].getvalue()).decode("utf-8")
                text = message["content"]
                message["content"] = [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
                del message["data"]
                del message["mime_type"]

        chunks = await self.get_chat_completions(context, messages)

        return chunks

    async def _process_context(self, context: OpenAILLMContext):
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        chunk_stream: AsyncStream[ChatCompletionChunk] = await self._stream_chat_completions(
            context
        )

        full_response = ""  # Initialize full_response to accumulate the response

        async for chunk in chunk_stream:
            if chunk.usage:
                tokens = LLMTokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
                await self.start_llm_usage_metrics(tokens)

            if len(chunk.choices) == 0:
                continue

            await self.stop_ttfb_metrics()

            if not chunk.choices[0].delta:
                continue

            if chunk.choices[0].delta.tool_calls:
                # We're streaming the LLM response to enable the fastest response times.
                # For text, we just yield each chunk as we receive it and count on consumers
                # to do whatever coalescing they need (eg. to pass full sentences to TTS)
                #
                # If the LLM is a function call, we'll do some coalescing here.
                # If the response contains a function name, we'll yield a frame to tell consumers
                # that they can start preparing to call the function with that name.
                # We accumulate all the arguments for the rest of the streamed response, then when
                # the response is done, we package up all the arguments and the function name and
                # yield a frame containing the function name and the arguments.

                tool_call = chunk.choices[0].delta.tool_calls[0]
                if tool_call.index != func_idx:
                    functions_list.append(function_name)
                    arguments_list.append(arguments)
                    tool_id_list.append(tool_call_id)
                    function_name = ""
                    arguments = ""
                    tool_call_id = ""
                    func_idx += 1
                if tool_call.function and tool_call.function.name:
                    function_name += tool_call.function.name
                    tool_call_id = tool_call.id
                    await self.call_start_function(context, function_name)
                if tool_call.function and tool_call.function.arguments:
                    # Keep iterating through the response to collect all the argument fragments
                    arguments += tool_call.function.arguments
            # elif chunk.choices[0].delta.content:
            #     await self.push_frame(TextFrame(chunk.choices[0].delta.content))

            if self._nmt_flag == True:
                if chunk.choices[0].delta.content is not None:
                    self._frame += chunk.choices[0].delta.content
                
                if self._frame.strip().endswith(
                    (".", "?", "!", "|", "।")) and not self._frame.strip().endswith(
                    ("Mr,", "Mrs.", "Ms.", "Dr.")):
                    text = self._frame
                    text = text.replace("*", "")
                    logger.debug(f"consolidated: {text}")
                    translator = NMTService(text, self._tgt_lan, self._nmt_provider)
                    processed_text = await translator.translate()
                    self._processed_text = processed_text
                    logger.debug(f"processed_text: {self._processed_text}")
                    self._frame = ""
            else:
                if chunk.choices[0].delta.content is not None:
                    cleaned_text = chunk.choices[0].delta.content.replace("*", "")
                    self._processed_text = cleaned_text
                else:
                    self._processed_text = chunk.choices[0].delta.content

            if self._processed_text:
                # await self.push_frame(LLMResponseStartFrame())
                await self.push_frame(TextFrame(self._processed_text))
                # await self.push_frame(LLMResponseEndFrame())
                full_response += self._processed_text  # Accumulate the response
                self._processed_text = ""

        # if we got a function name and arguments, check to see if it's a function with
        # a registered handler. If so, run the registered callback, save the result to
        # the context, and re-prompt to get a chat answer. If we don't have a registered
        # handler, raise an exception.
        if function_name and arguments:
            # added to the list as last function name and arguments not added to the list
            functions_list.append(function_name)
            arguments_list.append(arguments)
            tool_id_list.append(tool_call_id)

            for index, (function_name, arguments, tool_id) in enumerate(
                zip(functions_list, arguments_list, tool_id_list), start=1
            ):
                if self.has_function(function_name):
                    run_llm = False
                    arguments = json.loads(arguments)
                    await self.call_function(
                        context=context,
                        function_name=function_name,
                        arguments=arguments,
                        tool_call_id=tool_id,
                        run_llm=run_llm,
                    )
                else:
                    raise OpenAIUnhandledFunctionException(
                        f"The LLM tried to call a function named '{function_name}', but there isn't a callback registered for that function."
                    )
        # Cache the full response
        if self._vector_store == "qdrant":
            if self._ner_check=="true":
                if self.ner_check(full_response) == False:
                    logger.debug(f"going to cache")
                    self.cache_response(context, full_response)
            else:
                logger.debug(f"going to cache")
                self.cache_response(context, full_response)
        elif self._vector_store == "redis":
            self.cache_response(context, full_response)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = OpenAILLMContext.from_image_frame(frame)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self._process_context(context)
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())


@dataclass
class OpenAIContextAggregatorPair:
    _user: "OpenAIUserContextAggregator"
    _assistant: "OpenAIAssistantContextAggregator"

    def user(self) -> "OpenAIUserContextAggregator":
        return self._user

    def assistant(self) -> "OpenAIAssistantContextAggregator":
        return self._assistant


class OpenAILLMService(BaseOpenAILLMService):
    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        params: BaseOpenAILLMService.InputParams = BaseOpenAILLMService.InputParams(),
        **kwargs,
    ):
        super().__init__(model=model, params=params, **kwargs)

    @staticmethod
    def create_context_aggregator(
        context: OpenAILLMContext, *, assistant_expect_stripped_words: bool = True
    ) -> OpenAIContextAggregatorPair:
        user = OpenAIUserContextAggregator(context)
        assistant = OpenAIAssistantContextAggregator(
            user, expect_stripped_words=assistant_expect_stripped_words
        )
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)


class OpenAIImageGenService(ImageGenService):
    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        image_size: Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
        model: str = "dall-e-3",
    ):
        super().__init__()
        self.set_model_name(model)
        self._image_size = image_size
        self._client = AsyncOpenAI(api_key=api_key)
        self._aiohttp_session = aiohttp_session

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating image from prompt: {prompt}")

        image = await self._client.images.generate(
            prompt=prompt, model=self.model_name, n=1, size=self._image_size
        )

        image_url = image.data[0].url

        if not image_url:
            logger.error(f"{self} No image provided in response: {image}")
            yield ErrorFrame("Image generation failed")
            return

        # Load the image from the url
        async with self._aiohttp_session.get(image_url) as response:
            image_stream = io.BytesIO(await response.content.read())
            image = Image.open(image_stream)
            frame = URLImageRawFrame(image_url, image.tobytes(), image.size, image.format)
            yield frame


class OpenAITTSService(TTSService):
    """This service uses the OpenAI TTS API to generate audio from text.
    The returned audio is PCM encoded at 24kHz. When using the DailyTransport, set the sample rate in the DailyParams accordingly:
    ```
    DailyParams(
        audio_out_enabled=True,
        audio_out_sample_rate=24_000,
    )
    ```
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice: str = "alloy",
        model: Literal["tts-1", "tts-1-hd"] = "tts-1",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._settings = {
            "sample_rate": sample_rate,
        }
        self.set_model_name(model)
        self.set_voice(voice)

        self._client = AsyncOpenAI(api_key=api_key)

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        logger.info(f"Switching TTS model to: [{model}]")
        self.set_model_name(model)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")
        try:
            await self.start_ttfb_metrics()

            async with self._client.audio.speech.with_streaming_response.create(
                input=text or " ",  # Text must contain at least one character
                model=self.model_name,
                voice=VALID_VOICES[self._voice_id],
                response_format="pcm",
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(
                        f"{self} error getting audio (status: {r.status_code}, error: {error})"
                    )
                    yield ErrorFrame(
                        f"Error getting audio (status: {r.status_code}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame()
                async for chunk in r.iter_bytes(8192):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = TTSAudioRawFrame(chunk, self._settings["sample_rate"], 1)
                        yield frame
                yield TTSStoppedFrame()
        except BadRequestError as e:
            logger.exception(f"{self} error generating TTS: {e}")


# internal use only -- todo: refactor
@dataclass
class OpenAIImageMessageFrame(Frame):
    user_image_raw_frame: UserImageRawFrame
    text: Optional[str] = None


class OpenAIUserContextAggregator(LLMUserContextAggregator):
    def __init__(self, context: OpenAILLMContext):
        super().__init__(context=context)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # Our parent method has already called push_frame(). So we can't interrupt the
        # flow here and we don't need to call push_frame() ourselves.
        try:
            if isinstance(frame, UserImageRequestFrame):
                # The LLM sends a UserImageRequestFrame upstream. Cache any context provided with
                # that frame so we can use it when we assemble the image message in the assistant
                # context aggregator.
                if frame.context:
                    if isinstance(frame.context, str):
                        self._context._user_image_request_context[frame.user_id] = frame.context
                    else:
                        logger.error(
                            f"Unexpected UserImageRequestFrame context type: {type(frame.context)}"
                        )
                        del self._context._user_image_request_context[frame.user_id]
                else:
                    if frame.user_id in self._context._user_image_request_context:
                        del self._context._user_image_request_context[frame.user_id]
            elif isinstance(frame, UserImageRawFrame):
                # Push a new OpenAIImageMessageFrame with the text context we cached
                # downstream to be handled by our assistant context aggregator. This is
                # necessary so that we add the message to the context in the right order.
                text = self._context._user_image_request_context.get(frame.user_id) or ""
                if text:
                    del self._context._user_image_request_context[frame.user_id]
                frame = OpenAIImageMessageFrame(user_image_raw_frame=frame, text=text)
                await self.push_frame(frame)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")


class OpenAIAssistantContextAggregator(LLMAssistantContextAggregator):
    def __init__(self, user_context_aggregator: OpenAIUserContextAggregator, **kwargs):
        super().__init__(context=user_context_aggregator._context, **kwargs)
        self._user_context_aggregator = user_context_aggregator
        self._function_calls_in_progress = {}
        self._function_call_result = None
        self._pending_image_frame_message = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # See note above about not calling push_frame() here.
        if isinstance(frame, StartInterruptionFrame):
            self._function_calls_in_progress.clear()
            self._function_call_finished = None
        elif isinstance(frame, FunctionCallInProgressFrame):
            logger.debug(f"FunctionCallInProgressFrame: {frame}")
            self._function_calls_in_progress[frame.tool_call_id] = frame
        elif isinstance(frame, FunctionCallResultFrame):
            logger.debug(f"FunctionCallResultFrame: {frame}")
            if frame.tool_call_id in self._function_calls_in_progress:
                del self._function_calls_in_progress[frame.tool_call_id]
                self._function_call_result = frame
                # TODO-CB: Kwin wants us to refactor this out of here but I REFUSE
                await self._push_aggregation()
            else:
                logger.warning(
                    "FunctionCallResultFrame tool_call_id does not match any function call in progress"
                )
                self._function_call_result = None
        elif isinstance(frame, OpenAIImageMessageFrame):
            self._pending_image_frame_message = frame
            await self._push_aggregation()

    async def _push_aggregation(self):
        if not (
            self._aggregation or self._function_call_result or self._pending_image_frame_message
        ):
            return

        run_llm = False

        aggregation = self._aggregation
        self._reset()

        try:
            if self._function_call_result:
                frame = self._function_call_result
                self._function_call_result = None
                if frame.result:
                    self._context.add_message(
                        {
                            "role": "assistant",
                            "content": "",  # content field required for Grok function calling
                            "tool_calls": [
                                {
                                    "id": frame.tool_call_id,
                                    "function": {
                                        "name": frame.function_name,
                                        "arguments": json.dumps(frame.arguments),
                                    },
                                    "type": "function",
                                }
                            ],
                        }
                    )
                    self._context.add_message(
                        {
                            "role": "tool",
                            "content": json.dumps(frame.result),
                            "tool_call_id": frame.tool_call_id,
                        }
                    )
                    # Only run the LLM if there are no more function calls in progress.
                    run_llm = not bool(self._function_calls_in_progress)
            else:
                self._context.add_message({"role": "assistant", "content": aggregation})

            if self._pending_image_frame_message:
                frame = self._pending_image_frame_message
                self._pending_image_frame_message = None
                self._context.add_image_frame_message(
                    format=frame.user_image_raw_frame.format,
                    size=frame.user_image_raw_frame.size,
                    image=frame.user_image_raw_frame.image,
                    text=frame.text,
                )
                run_llm = True

            if run_llm:
                await self._user_context_aggregator.push_context_frame()

            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
