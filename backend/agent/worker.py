
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Annotated, Optional, List
import json
import httpx
import asyncio
import importlib.metadata

from dotenv import load_dotenv

# Add backend directory to sys.path FIRST so app.config can be imported!
BACKEND_DIR = Path(__file__).parent.parent
sys.path.append(str(BACKEND_DIR))

# Configuration flags
ANAM_ENABLED = False
DITTO_ENABLED = True  # Ditto TalkingHead avatar integration
DEBUG = False  # Set to True for debug logs
VERBOSE_LOGGING = False  # Set to True for very verbose logs

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    UserInputTranscribedEvent,
    ConversationItemAddedEvent,
    function_tool,
)
from livekit.agents.voice.room_io import RoomInputOptions

from livekit.plugins import (
    speechmatics,
    cartesia,
    silero,
    anam,
)
from livekit.agents import llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.llm import ChatContext, Tool, ToolChoice
from livekit.plugins.speechmatics import STT, SpeakerIdentifier, TurnDetectionMode

# Handle both cases: run directly from agent dir or imported as agent.worker
try:
    from prompts import SYSTEM_PROMPT
except ImportError:
    from agent.prompts import SYSTEM_PROMPT

from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("voice_agent")
logger.setLevel(logging.INFO)


# Create dedicated ANAM logger
anam_logger = logging.getLogger("anam_debug")
anam_logger.setLevel(logging.DEBUG)

# Create dedicated LLM logger
llm_logger = logging.getLogger("llm_debug")
llm_logger.setLevel(logging.DEBUG)

# Ensure loggers have handlers
if not anam_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    anam_logger.addHandler(handler)

if not llm_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    llm_logger.addHandler(handler)


env_loaded = load_dotenv()

# ==================== GLOBAL VARIABLES (YouTube-style architecture)
SPEAKERS_FILE = Path(__file__).parent.parent / "speakers.json"
known_speakers: List[SpeakerIdentifier] = []
stt: Optional[STT] = None

# Import speaker_utils functions
try:
    from .speaker_utils import load_known_speakers, save_speakers, normalize_speaker_ids
except ImportError:
    from speaker_utils import load_known_speakers, save_speakers, normalize_speaker_ids


from livekit import rtc
from livekit.agents.voice.avatar import DataStreamAudioOutput

# Import Ditto Avatar Worker
DITTO_AVATAR_AVAILABLE = False
DittoAvatarWorker = None
ditto_import_error = None

if DITTO_ENABLED:
    try:
        DITTO_DIR = BACKEND_DIR / "avatar" / "ditto"
        sys.path.insert(0, str(DITTO_DIR))
        from avatar.ditto.ditto_avatar_worker import DittoAvatarWorker as _DittoAvatarWorker
        DittoAvatarWorker = _DittoAvatarWorker
        DITTO_AVATAR_AVAILABLE = True
        logger.info("[DITTO] DittoAvatarWorker imported successfully.")
    except Exception as e:
        ditto_import_error = e
        logger.exception(f"[DITTO ERROR] Failed to import DittoAvatarWorker: {e}")

# Import ANAM modules with logging
if VERBOSE_LOGGING:
    anam_logger.info("[ANAM] Importing ANAM modules...")
try:
    import time
    import traceback
    if VERBOSE_LOGGING:
        anam_logger.info("[ANAM] ANAM modules imported successfully.")
except Exception as e:
    anam_logger.exception("[ANAM ERROR] Failed to import ANAM modules.")
    raise


# Custom Hugging Face LLM implementation using huggingface_hub.InferenceClient
class HuggingFaceLLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str = settings.HF_LLM_MODEL,
        api_key: str | None = None,
        provider: str = "auto",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        super().__init__()
        self._model = model
        self._api_key = api_key or os.getenv("HF_TOKEN")
        self._provider = provider
        self._temperature = temperature
        self._max_tokens = max_tokens
        
        # Initialize InferenceClient
        from huggingface_hub import InferenceClient
        self._client = InferenceClient(
            api_key=self._api_key,
            provider=self._provider,
        )
        if VERBOSE_LOGGING:
            llm_logger.info(f"[LLM INIT] Using provider: {self._provider}")

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "huggingface"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        return HuggingFaceLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            client=self._client,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            tool_choice=tool_choice,
        )

    async def aclose(self) -> None:
        pass


class HuggingFaceLLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: HuggingFaceLLM,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
        client,
        model: str,
        temperature: float,
        max_tokens: int,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
    ):
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._tool_choice = tool_choice

    async def _generate_once(self, chat_ctx_list, openai_tools, hf_tool_choice):
        import time
        import json
        start_time = time.time()
        full_response = ""
        num_chunks = 0
        first_chunk_latency = None
        # Dictionary to accumulate partial tool calls (key is index)
        partial_tool_calls = {}
        # Set to track emitted call_ids to avoid duplicates
        emitted_call_ids = set()

        def is_valid_json(s: str) -> bool:
            try:
                json.loads(s)
                return True
            except Exception:
                return False

        async def flush_pending_tool_calls():
            """Emit any valid, completed tool calls from partial_tool_calls"""
            delta_tool_calls = []
            for tc_index in list(partial_tool_calls.keys()):
                ptc = partial_tool_calls[tc_index]
                call_id = ptc.get("call_id")
                name = ptc.get("name")
                arguments = ptc.get("arguments", "")

                if not call_id or not name:
                    continue
                if call_id in emitted_call_ids:
                    llm_logger.info(f"[TOOL CALL] Duplicate call_id ignored: {call_id}")
                    continue
                if not is_valid_json(arguments):
                    llm_logger.info(f"[TOOL CALL] JSON not valid yet for {call_id}: {repr(arguments)}")
                    continue

                # Valid, add to delta_tool_calls
                llm_logger.info(f"[TOOL CALL] JSON validated for {call_id}: {repr(arguments)}")
                fn_tool_call = llm.FunctionToolCall(
                    type="function",
                    name=name,
                    arguments=arguments,
                    call_id=call_id,
                    extra=None
                )
                delta_tool_calls.append(fn_tool_call)
                emitted_call_ids.add(call_id)
                llm_logger.info(f"[TOOL CALL] Emitting: {call_id} ({name})")

                # Remove from partial to avoid re-emitting
                del partial_tool_calls[tc_index]

            if delta_tool_calls:
                chat_chunk = llm.ChatChunk(
                    id=f"flush-{num_chunks}",
                    delta=llm.ChoiceDelta(
                        content=None,
                        role="assistant",
                        tool_calls=delta_tool_calls,
                    ),
                )
                self._event_ch.send_nowait(chat_chunk)

        try:
            if VERBOSE_LOGGING:
                llm_logger.info("========== HF REQUEST ==========")
                llm_logger.info(f"Model: {self._model}")
                llm_logger.info(f"Messages: {chat_ctx_list}")
                llm_logger.info(f"Tools: {openai_tools}")
                llm_logger.info(f"Tool Choice: {hf_tool_choice}")
                llm_logger.info(f"Stream: True")
                llm_logger.info("================================")
                
                request_body = {
                    "model": self._model,
                    "messages": chat_ctx_list,
                    "tools": openai_tools,
                    "tool_choice": hf_tool_choice,
                    "stream": True,
                    "max_tokens": self._max_tokens,
                    "temperature": self._temperature
                }
                llm_logger.info(f"REQUEST BODY:\n{json.dumps(request_body, indent=2, default=str)}")
            
            # Build request args
            request_args = {
                "model": self._model,
                "messages": chat_ctx_list,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "stream": True,
            }
            if openai_tools:
                request_args["tools"] = openai_tools
            if hf_tool_choice:
                request_args["tool_choice"] = hf_tool_choice
            
            # Call Hugging Face chat completions in a separate thread
            response = await asyncio.to_thread(
                self._client.chat.completions.create,
                **request_args
            )

            if VERBOSE_LOGGING:
                llm_logger.info(f"type(response) = {type(response)}")

            for chunk in response:
                await asyncio.sleep(0)
                num_chunks += 1

                if first_chunk_latency is None:
                    first_chunk_latency = time.time() - start_time

                if VERBOSE_LOGGING:
                    llm_logger.info("========== RAW CHUNK ==========")
                    llm_logger.info(f"RAW CHUNK: {chunk}")
                    if hasattr(chunk, 'choices') and chunk.choices:
                        llm_logger.info(f"CHOICES: {chunk.choices}")
                        choice = chunk.choices[0]
                        if hasattr(choice, 'delta'):
                            delta = choice.delta
                            llm_logger.info(f"DELTA: {delta}")
                            if hasattr(delta, 'content'):
                                llm_logger.info(f"CONTENT: {delta.content}")
                            if hasattr(delta, 'tool_calls'):
                                llm_logger.info(f"TOOL_CALLS: {delta.tool_calls}")
                        if hasattr(choice, 'finish_reason'):
                            llm_logger.info(f"FINISH_REASON: {choice.finish_reason}")
                    llm_logger.info("================================")

                try:
                    delta_content = None
                    finish_reason = None
                    
                    if hasattr(chunk, 'choices') and chunk.choices:
                        choice = chunk.choices[0]
                        if hasattr(choice, 'delta'):
                            delta = choice.delta
                            if hasattr(delta, 'content'):
                                delta_content = delta.content
                            
                            # Detect tool calls and accumulate them
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                for tc in delta.tool_calls:
                                    tc_index = getattr(tc, 'index', 0)
                                    
                                    # Initialize or update partial tool call
                                    if tc_index not in partial_tool_calls:
                                        partial_tool_calls[tc_index] = {
                                            "call_id": None,
                                            "name": None,
                                            "arguments": ""
                                        }
                                        llm_logger.info(f"[TOOL CALL] Initialized partial tool call at index {tc_index}")
                                    
                                    # Update call_id if present
                                    if hasattr(tc, 'id') and tc.id is not None:
                                        partial_tool_calls[tc_index]["call_id"] = tc.id
                                        llm_logger.info(f"[TOOL CALL] Updated call_id at index {tc_index}: {tc.id}")
                                    
                                    # Update function name and arguments if present
                                    if hasattr(tc, 'function'):
                                        func = tc.function
                                        if hasattr(func, 'name') and func.name is not None:
                                            partial_tool_calls[tc_index]["name"] = func.name
                                            llm_logger.info(f"[TOOL CALL] Updated name at index {tc_index}: {func.name}")
                                        if hasattr(func, 'arguments') and func.arguments is not None:
                                            partial_tool_calls[tc_index]["arguments"] += func.arguments
                                            llm_logger.info(f"[TOOL CALL] Accumulated arguments at index {tc_index}: {repr(partial_tool_calls[tc_index]['arguments'])}")
                        elif hasattr(choice, 'text'):
                            delta_content = choice.text
                        
                        if hasattr(choice, 'finish_reason'):
                            finish_reason = choice.finish_reason
                            # llm_logger.info(f"[TOOL CALL] Finish reason received: {finish_reason}")
                    elif hasattr(chunk, 'delta'):
                        delta = chunk.delta
                        if hasattr(delta, 'content'):
                            delta_content = delta.content

                    if delta_content:
                        full_response += delta_content
                        chat_chunk = llm.ChatChunk(
                            id=getattr(chunk, 'id', f"chunk-{num_chunks}"),
                            delta=llm.ChoiceDelta(
                                content=delta_content,
                                role="assistant",
                                tool_calls=[],
                            ),
                        )
                        self._event_ch.send_nowait(chat_chunk)
                    
                    if finish_reason in ["tool_calls", "stop"]:
                        llm_logger.info("[TOOL CALL] Flushing pending tool calls due to finish reason")
                        await flush_pending_tool_calls()

                except Exception as parse_e:
                    llm_logger.warning(f"[LLM WARNING] Failed to parse chunk: {parse_e}")
                    if VERBOSE_LOGGING:
                        llm_logger.warning(f"[LLM WARNING] Parse error traceback:\n{traceback.format_exc()}")
                    continue

            # Flush remaining pending tool calls at end of stream
            llm_logger.info("[TOOL CALL] End of stream, flushing pending tool calls")
            await flush_pending_tool_calls()

            total_latency = time.time() - start_time
            if VERBOSE_LOGGING:
                llm_logger.info(f"[LLM RESPONSE] Generation successful")
            return True, full_response

        except Exception as e:
            llm_logger.error(f"[LLM ERROR] Error type: {type(e).__name__}")
            llm_logger.error(f"[LLM ERROR] Error message: {str(e)}")
            llm_logger.error(f"[LLM ERROR] Traceback:\n{traceback.format_exc()}")
            return False, str(e)

    async def _run(self) -> None:
        import time
        start_time = time.time()
        
        try:
            chat_ctx_list, _ = self._chat_ctx.to_provider_format(format="openai")
            if VERBOSE_LOGGING:
                llm_logger.info("[LLM REQUEST]")
                llm_logger.info(f"[LLM MESSAGE COUNT] {len(chat_ctx_list)}")
            
            openai_tools = []
            if self._tools:
                from livekit.agents import llm
                for tool in self._tools:
                    if isinstance(tool, llm.FunctionTool):
                        schema = llm.utils.build_strict_openai_schema(tool)
                        openai_tools.append(schema)
                if VERBOSE_LOGGING:
                    llm_logger.info(f"[LLM TOOLS] Converted {len(openai_tools)} tools to OpenAI format")
            
            hf_tool_choice = None
            if VERBOSE_LOGGING:
                llm_logger.info(f"[TOOL CHOICE DEBUG] type={type(self._tool_choice)} value={self._tool_choice}")
            if self._tool_choice is not NOT_GIVEN:
                if isinstance(self._tool_choice, str):
                    hf_tool_choice = self._tool_choice
                elif isinstance(self._tool_choice, dict):
                    hf_tool_choice = self._tool_choice
            elif openai_tools:
                hf_tool_choice = "auto"
            
            last_user_msg = None
            for msg in reversed(chat_ctx_list):
                if msg.get("role") == "user":
                    last_user_msg = msg.get("content", "")
                    break
            if VERBOSE_LOGGING:
                llm_logger.info(f"[LLM REQUEST] Last user message: {last_user_msg}")
                llm_logger.info("[LLM REQUEST] Sending request to Hugging Face endpoint")
                llm_logger.info(f"[LLM MODEL] {self._model}")
                llm_logger.info(f"[LLM PROVIDER] {self._llm.provider}")

            success, result = await self._generate_once(chat_ctx_list, openai_tools, hf_tool_choice)
            if success:
                return

            max_retries = 3
            for attempt in range(1, max_retries + 1):
                if VERBOSE_LOGGING:
                    llm_logger.info(f"[LLM RETRY] Attempt: {attempt}")
                success, result = await self._generate_once(chat_ctx_list, openai_tools, hf_tool_choice)
                if success:
                    return
                else:
                    llm_logger.error(f"[LLM ERROR] Retry {attempt} failed: {result}")

            llm_logger.error("[LLM FAILURE] All retries exhausted.")
            llm_logger.error("[LLM FAILURE] Returning fallback response.")
            fallback = "I'm sorry, I couldn't generate a response right now. Please try again."
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id="fallback",
                    delta=llm.ChoiceDelta(
                        content=fallback,
                        role="assistant",
                    ),
                )
            )

        finally:
            try:
                if hasattr(self._event_ch, 'aclose'):
                    await self._event_ch.aclose()
                    if VERBOSE_LOGGING:
                        llm_logger.info("[LLM STREAM] Event channel closed (aclose)")
                elif hasattr(self._event_ch, 'close'):
                    self._event_ch.close()
                    if VERBOSE_LOGGING:
                        llm_logger.info("[LLM STREAM] Event channel closed (close)")
            except Exception as close_e:
                llm_logger.warning(f"[LLM WARNING] Failed to close event channel: {close_e}")


# # Define function tools
# @function_tool
# async def assign_name_2_speaker_ids(
#     name: Annotated[
#         str,
#         "The name of the current speaker to remember",
#     ],
# ) -> str:
#     """Assigns a human-readable name to the current speaker; call this ONLY when the user says their own name (e.g., "My name is X", "I am X", "Call me X") and NOT when they mention someone else's name, ask questions, or discuss general topics."""
#     logger.info(f"[TOOL] assign_name_2_speaker_ids() called with name: {name}")
    
#     global stt, known_speakers
#     if not stt:
#         logger.error("[TOOL] STT not initialized")
#         raise RuntimeError("STT not initialized")
    
#     logger.info("[STT] Calling get_speaker_ids()")
#     raw_speaker_ids = await stt.get_speaker_ids()
#     speaker_ids = normalize_speaker_ids(raw_speaker_ids)

#     logger.info(f"[STT] Normalized speaker ids count: {len(speaker_ids)}")
    
#     updated_speakers = list(known_speakers)
#     found = False
    
#     for i, speaker in enumerate(updated_speakers):
#         if speaker.label.lower() == name.lower():
#             current_ids = set(speaker.speaker_identifiers)
#             current_ids.update(speaker_ids)
            
#             updated_speakers[i] = SpeakerIdentifier(
#                 label=speaker.label,
#                 speaker_identifiers=list(current_ids),
#             )
#             found = True
#             logger.info(f"[SPEAKER] Updated {name}")
#             logger.info(f"[SPEAKER] Identifiers: {updated_speakers[i].speaker_identifiers}")
#             break
    
#     if not found:
#         new_speaker = SpeakerIdentifier(
#             label=name,
#             speaker_identifiers=speaker_ids,
#         )
#         updated_speakers.append(new_speaker)
#         logger.info(f"[SPEAKER] Created new {name}")
#         logger.info(f"[SPEAKER] Identifiers: {new_speaker.speaker_identifiers}")
    
#     known_speakers = updated_speakers
#     save_speakers(SPEAKERS_FILE, updated_speakers)
    
#     return f"Got it! I'll remember you as {name}."

# Define function tools
@function_tool
async def assign_name_2_speaker_ids(
    name: Annotated[
        str,
        "The name of the current speaker to remember",
    ],
) -> str:
    """
    Assigns a human-readable name to the current speaker.

    Call ONLY when the user is introducing themselves:
    - My name is John
    - I am John
    - Call me John

    Never call for names of other people.
    """

    logger.info(f"[TOOL] assign_name_2_speaker_ids() called with raw name: {name}")

    global stt, known_speakers

    if not stt:
        logger.error("[TOOL] STT not initialized")
        raise RuntimeError("STT not initialized")

    # ------------------------------------------------------------------
    # Normalize name
    # ------------------------------------------------------------------
    name = (name or "").strip()

    if not name:
        logger.warning("[TOOL] Empty name received.")
        return "Sorry, I didn't catch your name."

    # Remove surrounding punctuation
    name = name.strip(".,!?\"' ")

    # Remove duplicated whitespace
    name = " ".join(name.split())

    # Reject clearly invalid names
    INVALID_NAMES = {
        "my",
        "name",
        "is",
        "was",
        "i",
        "me",
        "call",
        "unknown",
        "speaker",
        "user",
        "person",
    }

    if name.lower() in INVALID_NAMES:
        logger.warning(f"[TOOL] Ignoring invalid extracted name: {name}")
        return "Sorry, I couldn't determine your name."

    logger.info(f"[TOOL] Normalized name: {name}")

    # ------------------------------------------------------------------
    # Fetch current Speechmatics speaker identifiers
    # ------------------------------------------------------------------
    try:
        logger.info("[STT] Calling get_speaker_ids()")
        raw_speaker_ids = await stt.get_speaker_ids()
        speaker_ids = normalize_speaker_ids(raw_speaker_ids)

    except Exception:
        logger.exception("[STT] Failed to obtain speaker identifiers.")
        return (
            f"I heard your name is {name}, "
            "but I couldn't save it because speaker identification failed."
        )

    if not speaker_ids:
        logger.warning("[STT] No speaker identifiers returned.")
        return (
            f"I heard your name is {name}, "
            "but I couldn't identify your voice yet."
        )

    logger.info(f"[STT] Normalized speaker IDs: {speaker_ids}")

    # ------------------------------------------------------------------
    # Remove duplicate IDs
    # ------------------------------------------------------------------
    speaker_ids = list(dict.fromkeys(speaker_ids))

    updated_speakers = list(known_speakers)

    # ------------------------------------------------------------------
    # Check whether these identifiers already belong to another user
    # ------------------------------------------------------------------
    existing_owner = None

    for speaker in updated_speakers:
        overlap = set(speaker.speaker_identifiers).intersection(speaker_ids)

        if overlap:
            existing_owner = speaker
            logger.info(
                f"[SPEAKER] Matching identifiers already belong to "
                f"{speaker.label}: {list(overlap)}"
            )
            break

    # ------------------------------------------------------------------
    # Same speaker introducing themselves again
    # ------------------------------------------------------------------
    if existing_owner:

        if existing_owner.label.lower() == name.lower():

            merged_ids = sorted(
                set(existing_owner.speaker_identifiers).union(speaker_ids)
            )

            existing_owner.speaker_identifiers = merged_ids

            logger.info(
                f"[SPEAKER] Refreshed identifiers for existing speaker '{name}'."
            )

        else:
            logger.warning(
                f"[SPEAKER] Speaker identifiers already belong to "
                f"'{existing_owner.label}', not '{name}'."
            )

            return (
                f"I already recognize this voice as "
                f"{existing_owner.label}. "
                "If you really want to rename this speaker, "
                "please use a dedicated rename flow."
            )

    # ------------------------------------------------------------------
    # Otherwise search by label
    # ------------------------------------------------------------------
    else:

        label_match = None

        for speaker in updated_speakers:
            if speaker.label.lower() == name.lower():
                label_match = speaker
                break

        if label_match:

            merged_ids = sorted(
                set(label_match.speaker_identifiers).union(speaker_ids)
            )

            label_match.speaker_identifiers = merged_ids

            logger.info(
                f"[SPEAKER] Added new identifiers to existing label '{name}'."
            )

        else:

            new_speaker = SpeakerIdentifier(
                label=name,
                speaker_identifiers=sorted(speaker_ids),
            )

            updated_speakers.append(new_speaker)

            logger.info(
                f"[SPEAKER] Created new speaker '{name}' "
                f"with identifiers: {speaker_ids}"
            )

    # ------------------------------------------------------------------
    # Persist to disk
    # ------------------------------------------------------------------
    try:
        known_speakers = updated_speakers
        save_speakers(SPEAKERS_FILE, updated_speakers)

        logger.info(
            f"[SPEAKER] Successfully saved {len(updated_speakers)} speakers."
        )

    except Exception:
        logger.exception("[SPEAKER] Failed saving speakers.json")
        return (
            f"I recognized your name as {name}, "
            "but couldn't save it due to an internal error."
        )

    return f"Got it! I'll remember you as {name}."


# Global variable for participant metadata
participant_metadata = {
    "user_id": "default_user",
    "document_id": "default_doc",
}


@function_tool
async def search_company_documents(
    question: Annotated[
        str,
        "Question regarding uploaded company documents, policies, manuals, procedures, SOPs, internal knowledge base or enterprise documents",
    ],
) -> str:
    """Searches the company's document knowledge base and returns the answer"""
    logger.info("[RAG TOOL] Tool invoked")
    logger.info(f"[RAG TOOL] Question received: {question}")
    logger.info(
        f"[RAG TOOL] Metadata - user_id={participant_metadata['user_id']}, document_id={participant_metadata['document_id']}"
    )

    try:
        logger.info(f"[RAG TOOL] Calling FastAPI at {settings.RAG_API_URL}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                settings.RAG_API_URL,
                json={
                    "question": question,
                    "user_id": participant_metadata["user_id"],
                    "document_id": participant_metadata["document_id"],
                },
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "")
            logger.info(f"[RAG TOOL] Response received")
            logger.info(f"[RAG TOOL] Answer length: {len(answer)} characters")
            logger.info(f"[RAG TOOL] Answer: {answer}")
            return answer
    except httpx.TimeoutException:
        logger.error("[RAG TOOL] Error: Request to FastAPI timed out")
        return "I'm sorry, the document search service took too long to respond. Please try again later."
    except httpx.HTTPStatusError as e:
        logger.error(
            f"[RAG TOOL] Error: FastAPI returned status {e.response.status_code}: {e}"
        )
        return "I'm sorry, there was an error searching the documents. Please try again later."
    except httpx.RequestError as e:
        logger.error(
            f"[RAG TOOL] Error: Network error when calling FastAPI: {e}"
        )
        return "I'm sorry, I couldn't connect to the document search service. Please check your connection and try again."
    except Exception as e:
        logger.exception(f"[RAG TOOL] Error: Unexpected error: {e}")
        return "I'm sorry, something went wrong while searching the documents. Please try again later."


class DebugSynthesizeStream:
    def __init__(self, stream, ditto_consumer=None):
        self._stream = stream
        self._ditto_consumer = ditto_consumer
        if ditto_consumer is not None:
            logger.info("[DITTO] DebugSynthesizeStream wired with Ditto consumer")

    def __aiter__(self):
        return self

    async def __anext__(self):
        ev = await self._stream.__anext__()

        frame = ev.frame

        logger.info("\n" + "=" * 60)
        logger.info("[CARTESIA → SYNTHESIZED AUDIO]")
        logger.info("=" * 60)
        logger.info(f"request_id:           {ev.request_id}")
        logger.info(f"segment_id:           {ev.segment_id}")
        logger.info(f"is_final:             {ev.is_final}")
        logger.info(f"sample_rate:          {frame.sample_rate}")
        logger.info(f"num_channels:         {frame.num_channels}")
        logger.info(f"samples_per_channel:  {frame.samples_per_channel}")
        logger.info(f"duration:              {frame.duration}")
        logger.info(f"data type:             {type(frame.data)}")
        logger.info(f"data length:           {len(frame.data)}")
        logger.info("=" * 60)

        if self._ditto_consumer is not None:
            try:
                self._ditto_consumer.on_synthesized_audio(ev)
            except Exception as ditto_e:
                logger.error(
                    f"[DITTO ERROR] Ditto consumer failed (audio playback unaffected): {ditto_e}"
                )
                if VERBOSE_LOGGING:
                    logger.exception(ditto_e)

        return ev

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, exc_tb):
        return await self._stream.__aexit__(
            exc_type,
            exc,
            exc_tb,
        )

    @property
    def done(self):
        return self._stream.done

    @property
    def exception(self):
        return self._stream.exception

    async def aclose(self):
        await self._stream.aclose()

    def push_text(self, token):
        return self._stream.push_text(token)

    def flush(self):
        return self._stream.flush()

    def end_input(self):
        return self._stream.end_input()


class DebugCartesiaTTS(cartesia.TTS):
    def __init__(self, *args, ditto_consumer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ditto_consumer = ditto_consumer
        if ditto_consumer is not None:
            logger.info("[DITTO] DebugCartesiaTTS wired with Ditto consumer")

    def stream(
        self,
        *,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
    ):
        stream = super().stream(
            conn_options=conn_options
        )

        return DebugSynthesizeStream(stream, ditto_consumer=self._ditto_consumer)


async def entrypoint(ctx: JobContext):
    global stt, known_speakers
    logger.info("[WORKER] ================ ENTRYPOINT STARTED ================")
    logger.info(f"[WORKER] JobContext ctx attributes: {dir(ctx)}")
    logger.info(f"[WORKER] Job ID: {ctx.job.id}")
    logger.info(f"[WORKER] Room name: {ctx.room.name}")
    # Try to get room sid from ctx.job or ctx.room
    room_sid = getattr(ctx.job, "room_sid", None) or getattr(ctx.room, "sid", None) or getattr(ctx.room, "room_sid", None)
    logger.info(f"[WORKER] Room SID (room_id): {room_sid}")
    logger.info(f"[WORKER] Room object: {ctx.room}")
    logger.info(f"[WORKER] ctx has delete_room method: {hasattr(ctx, 'delete_room')}")
    logger.info("[WORKER] Connecting to room...")
    
    # Track if we already tried to delete room
    room_deleted = False
    ditto_worker = None

    # Event to gate entrypoint lifetime: set ONLY when LiveKit fires the end-of-job
    # shutdown signal.  Without this, `await session.start()` returns ~0.3s after
    # starting the session loop and the entrypoint function exits prematurely.
    # (The underlying ipc/job_proc_lazy_main keeps the job alive anyway, but our
    #  cleanup logic attached to `on_shutdown` must remain registered and our
    #  DittoAdapter thread must remain alive for the FULL job duration.)
    shutdown_done = asyncio.Event()

    async def check_and_delete_room():
        """Check remaining participants and delete if only agent left"""
        nonlocal room_deleted
        if room_deleted:
            return
        
        logger.info("================ ROOM CLEANUP ================")
        logger.info("Participant disconnected - checking remaining participants...")
        # Get all remote participants (excluding the agent itself)
        remote_participants = list(ctx.room.remote_participants.values())
        human_participants = [
            p for p in remote_participants 
            if not (p.identity.startswith("agent-") or p.identity == "voice-agent-rag")
        ]
        logger.info(f"Remaining human participants: {len(human_participants)}")
        if len(human_participants) == 0:
            logger.info(f"No human participants left. Deleting room: '{ctx.room.name}'")
            room_deleted = True
            try:
                await ctx.delete_room()
                logger.info(f"✅ Room deleted successfully: name='{ctx.room.name}'")
            except Exception as e:
                logger.exception(f"❌ Failed to delete room: name='{ctx.room.name}'")
        logger.info("==============================================")
    
    # Add shutdown callback using LiveKit's official ctx.add_shutdown_callback
    # This callback fires ONLY when the job truly ends (triggered by
    # job_proc_lazy_main.py after `_shutdown_fut` resolves and session.aclose()
    # has been awaited). It is the canonical end-of-job cleanup hook.
    async def on_shutdown(shutdown_reason: str = ""):
        logger.info(
            f"[WORKER] LiveKit END-OF-JOB shutdown callback triggered! "
            f"Shutdown reason: '{shutdown_reason}'. Cleaning up..."
        )

        if ditto_worker is not None:
            try:
                logger.info(
                    f"[DITTO] Stopping Ditto avatar worker via end-of-job on_shutdown callback "
                    f"(reason='{shutdown_reason}')..."
                )
                await ditto_worker.aclose()
                logger.info(
                    "[DITTO] Ditto avatar worker stopped successfully (via end-of-job on_shutdown)"
                )
            except Exception as ditto_e:
                logger.exception(f"[DITTO ERROR] Error stopping Ditto avatar worker: {ditto_e}")

        if not room_deleted:
            await check_and_delete_room()

        shutdown_done.set()
        logger.info(
            f"[WORKER] End-of-job on_shutdown complete | shutdown_done event set | "
            f"shutdown_reason='{shutdown_reason}'"
        )
    ctx.add_shutdown_callback(on_shutdown)

    await ctx.connect(
        auto_subscribe=AutoSubscribe.AUDIO_ONLY,
    )
    logger.info("[WORKER] Successfully connected to room!")

    # Helper function to send agent status to frontend via data channel
    async def send_agent_status(status: str, tool_name: str | None = None):
        payload = {
            "type": "agent_status",
            "status": status
        }
        if tool_name:
            payload["tool_name"] = tool_name
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps(payload).encode("utf-8"),
                reliable=True
            )
            logger.info(f"[AGENT STATUS] Sent: {payload}")
        except Exception as e:
            logger.error(f"[AGENT STATUS] Failed to send: {e}")

    # Create wrapper for function tools that sends status events
    def tool_wrapper(func, tool_name):
        async def wrapped(*args, **kwargs):
            await send_agent_status("calling_tool", tool_name)
            try:
                result = await func(*args, **kwargs)
                await send_agent_status("generating_response")
                return result
            except Exception as e:
                await send_agent_status("tool_failed", tool_name)
                raise
        return wrapped

    # Register all room event listeners
    @ctx.room.on("participant_connected")
    def on_participant_connected(participant):
        logger.info(
            f"[ROOM EVENT] Participant connected: identity={participant.identity}, sid={participant.sid}"
        )
        current_room_sid = getattr(ctx.job, "room_sid", None) or getattr(ctx.room, "sid", None) or getattr(ctx.room, "room_sid", None)
        logger.info(f"[ROOM EVENT] Current room name: {ctx.room.name}, room sid: {current_room_sid}")
        if VERBOSE_LOGGING:
            anam_logger.info(f"[ANAM] PARTICIPANT CONNECTED: {participant.identity}")

        # Extract metadata from participant
        if participant.metadata:
            try:
                metadata = json.loads(participant.metadata)
                participant_metadata["user_id"] = metadata.get("user_id", "default_user")
                participant_metadata["document_id"] = metadata.get(
                    "document_id", "default_doc"
                )
                logger.info(
                    f"[PARTICIPANT METADATA] user_id={participant_metadata['user_id']}, document_id={participant_metadata['document_id']}"
                )
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse participant metadata as JSON: {participant.metadata}"
                )

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        logger.info(
            f"[ROOM EVENT] Participant disconnected: identity={participant.identity}, sid={participant.sid}"
        )
        current_room_sid = getattr(ctx.job, "room_sid", None) or getattr(ctx.room, "sid", None) or getattr(ctx.room, "room_sid", None)
        logger.info(f"[ROOM EVENT] Current room name: {ctx.room.name}, room sid: {current_room_sid}")
        if VERBOSE_LOGGING:
            anam_logger.info(f"[ANAM] PARTICIPANT DISCONNECTED: {participant.identity}")
        # Schedule check for deleting room
        asyncio.create_task(check_and_delete_room())

    @ctx.room.on("track_published")
    def on_track_published(publication, participant):
        logger.info(
            f"[ROOM EVENT] Track published: kind={publication.kind}, participant={participant.identity}"
        )
        current_room_sid = getattr(ctx.job, "room_sid", None) or getattr(ctx.room, "sid", None) or getattr(ctx.room, "room_sid", None)
        logger.info(f"[ROOM EVENT] Current room name: {ctx.room.name}, room sid: {current_room_sid}")
        if VERBOSE_LOGGING:
            anam_logger.info(
                f"[ANAM] TRACK PUBLISHED: {participant.identity} {publication.kind}"
            )

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        logger.info(
            f"[ROOM EVENT] Track subscribed: kind={track.kind}, participant={participant.identity}"
        )
        current_room_sid = getattr(ctx.job, "room_sid", None) or getattr(ctx.room, "sid", None) or getattr(ctx.room, "room_sid", None)
        logger.info(f"[ROOM EVENT] Current room name: {ctx.room.name}, room sid: {current_room_sid}")
        if VERBOSE_LOGGING:
            anam_logger.info(
                f"[ANAM] TRACK SUBSCRIBED: {participant.identity} {track.kind}"
            )

    logger.info("[WORKER] Waiting for participant to join...")
    participant = await ctx.wait_for_participant()
    logger.info(f"[WORKER] Participant joined: {participant.identity}")

    # Extract metadata from joined participant
    if participant.metadata:
        try:
            metadata = json.loads(participant.metadata)
            participant_metadata["user_id"] = metadata.get("user_id", "default_user")
            participant_metadata["document_id"] = metadata.get(
                "document_id", "default_doc"
            )
            logger.info(
                f"[PARTICIPANT METADATA] user_id={participant_metadata['user_id']}, document_id={participant_metadata['document_id']}"
            )
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse participant metadata as JSON: {participant.metadata}"
            )

    # Initialize known speakers and STT (global)
    logger.info("[SPEAKER] Initializing STT...")
    known_speakers = load_known_speakers(SPEAKERS_FILE)
    
    stt = STT(
        turn_detection_mode=TurnDetectionMode.SMART_TURN,
        enable_diarization=True,
        speaker_active_format="&lt;{speaker_id}&gt;{text}&lt;/{speaker_id}&gt;",
        speaker_passive_format="&lt;PASSIVE&gt;&lt;{speaker_id}&gt;{text}&lt;/{speaker_id}&gt;&lt;/PASSIVE&gt;",
        known_speakers=known_speakers,
        api_key=settings.SPEECHMATICS_API_KEY,
    )
    
    logger.info("[SPEAKER] STT initialized successfully")

    # Wrap the tools with status sending
    wrapped_assign_name = tool_wrapper(assign_name_2_speaker_ids, "assign_name_2_speaker_ids")
    wrapped_search_docs = tool_wrapper(search_company_documents, "search_company_documents")

    # Create wrapped function tools
    from livekit.agents import function_tool
    @function_tool
    async def wrapped_assign_name_2_speaker_ids(
        name: Annotated[
            str,
            "The name of the current speaker to remember",
        ],
    ) -> str:
        return await wrapped_assign_name(name)

    @function_tool
    async def wrapped_search_company_documents(
        question: Annotated[
            str,
            "Question regarding uploaded company documents, policies, manuals, procedures, SOPs, internal knowledge base or enterprise documents",
        ],
    ) -> str:
        return await wrapped_search_docs(question)

    # Initialize the agent with wrapped tools
    agent = Agent(
        instructions=SYSTEM_PROMPT,
        tools=[wrapped_assign_name_2_speaker_ids, wrapped_search_company_documents],
    )

    # LLM startup checks and initialization
    if VERBOSE_LOGGING:
        llm_logger.info("[LLM CHECK] Checking environment variables...")
    hf_token = os.getenv("HF_TOKEN")
    if VERBOSE_LOGGING:
        llm_logger.info(f"[LLM CHECK] HF_TOKEN exists: {bool(hf_token)}")

    llm_logger.info("[LLM INIT] Loading Hugging Face LLM")
    llm_logger.info(f"[LLM INIT] Model: {settings.HF_LLM_MODEL}")
    
    if not hf_token:
        llm_logger.error("[LLM ERROR] HF_TOKEN environment variable not found.")
        raise ValueError("HF_TOKEN is required for Hugging Face Inference API")

    # Initialize custom HuggingFace LLM
    llm_instance = HuggingFaceLLM(
        model=settings.HF_LLM_MODEL,
        api_key=hf_token,
        provider="auto",
        temperature=0.7,
        max_tokens=1024,
    )
    llm_logger.info("LLM instance created successfully")

    # Initialize session first
    session = AgentSession(
        stt=stt,
        llm=llm_instance,
        tts=cartesia.TTS(
            voice=os.getenv("CARTESIA_VOICE_ID"),
            sample_rate=16000,
        ),
        vad=silero.VAD.load(),
        preemptive_generation=False,
    )

    # ===== DITTO AVATAR WORKER INITIALIZATION =====
    if DITTO_ENABLED and DITTO_AVATAR_AVAILABLE and DittoAvatarWorker is not None:
        try:
            logger.info("[DITTO] ===== ENTERED DITTO AVATAR WORKER INITIALIZATION =====")
            runtime_agent_identity = ctx.room.local_participant.identity
            logger.info(
                f"[DITTO] Starting DittoAvatarWorker for room '{ctx.room.name}', agent identity='{runtime_agent_identity}'..."
            )
            ditto_worker = DittoAvatarWorker(
                identity=settings.DITTO_AVATAR_IDENTITY,
                debug_display=settings.DITTO_DEBUG_DISPLAY,
            )
            await ditto_worker.start(
                room_name=ctx.room.name,
                runtime_agent_identity=runtime_agent_identity,
            )
            logger.info(
                f"[DITTO] DittoAvatarWorker started successfully (identity='{settings.DITTO_AVATAR_IDENTITY}')."
            )

            # Route Cartesia TTS audio to Ditto Avatar Worker via LiveKit DataStream
            session.output.audio = DataStreamAudioOutput(
                room=ctx.room,
                destination_identity=settings.DITTO_AVATAR_IDENTITY,
                sample_rate=16000,
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
                wait_playback_start=True,
            )
            logger.info(
                f"[DITTO] session.output.audio configured with DataStreamAudioOutput(destination='{settings.DITTO_AVATAR_IDENTITY}')"
            )
            logger.info("[DITTO] ===== DITTO INITIALIZATION COMPLETE =====")
        except Exception as e:
            logger.exception(f"[DITTO ERROR] Ditto avatar worker failed to initialize: {e}")
            logger.warning("[DITTO] Ditto integration disabled (voice agent continues normally with direct audio)")
            ditto_worker = None
    else:
        if not DITTO_ENABLED:
            logger.info("[DITTO] Ditto integration disabled via DITTO_ENABLED=False")
        elif not DITTO_AVATAR_AVAILABLE:
            logger.warning(f"[DITTO] DittoAvatarWorker not available (import error: {ditto_import_error})")
        else:
            logger.warning("[DITTO] DittoAvatarWorker class not loaded")

    # Keep track of state for playback synchronization and logging
    current_agent_response = [""]

    # Add USER INPUT TRANSCRIBED event listener
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent):
        speaker_id = event.speaker_id or "UNKNOWN"
        transcript = event.transcript.strip()
        logger.info(f"\n[STT FINAL TRANSCRIPT]")
        logger.info(f"Speaker ID: {speaker_id}")
        logger.info(f"Transcript: '{transcript}'")
        if DEBUG and not event.is_final:
            logger.info(f"[STT PARTIAL] '{transcript}'")

    @session.on("conversation_item_added")
    def on_conversation_item_added(event: ConversationItemAddedEvent):
        if not hasattr(event.item, "text_content"):
            return

        if event.item.role == "assistant":
            current_agent_response[0] = event.item.text_content or ""
            logger.info(f"\n[TTS START]")
            logger.info(f"Text: '{current_agent_response[0]}'")

    # Add AGENT STATE CHANGED event listener
    @session.on("agent_state_changed")
    def on_agent_state_changed(event):
        old_state = getattr(event, "old_state", None)
        new_state = getattr(event, "new_state", None)

        logger.info(f"\n[CONVERSATION STATE]")
        logger.info(f"Previous: {old_state}")
        logger.info(f"Current: {new_state}")

        # Send status to frontend
        if new_state in ["listening", "speaking", "thinking"]:
            asyncio.create_task(send_agent_status(new_state))

        if new_state == "listening" and old_state == "speaking":
            logger.info(f"\n[TTS END]")
            logger.info(f"Text: '{current_agent_response[0]}'")

    # ===== ANAM INITIALIZATION =====
    if VERBOSE_LOGGING:
        anam_logger.info(f"[ANAM] Enabled: {ANAM_ENABLED}")
    if ANAM_ENABLED:
        if VERBOSE_LOGGING:
            anam_logger.info("[ANAM] ===== ENTERED ANAM INITIALIZATION =====")
            anam_logger.info(f"[ANAM] Room name: {ctx.room.name}")
            anam_logger.info(
                f"[ANAM] Local participant identity: {ctx.room.local_participant.identity}"
            )
            anam_logger.info(
                f"[ANAM] Local participant metadata: {ctx.room.local_participant.metadata}"
            )

        avatar = None
        try:
            if VERBOSE_LOGGING:
                anam_logger.info("[ANAM] Checking environment variables...")
            api_key = os.getenv("ANAM_API_KEY")
            avatar_id = os.getenv("ANAM_AVATAR_ID")
            anam_api_url = os.getenv("ANAM_API_URL")
            anam_session_url = os.getenv("ANAM_SESSION_URL")

            if VERBOSE_LOGGING:
                anam_logger.info(f"[ANAM] ANAM_API_KEY exists: {bool(api_key)}")
                anam_logger.info(f"[ANAM] ANAM_AVATAR_ID exists: {bool(avatar_id)}")
                anam_logger.info(f"[ANAM] ANAM_API_URL exists: {bool(anam_api_url)}")
                anam_logger.info(f"[ANAM] ANAM_SESSION_URL exists: {bool(anam_session_url)}")

            if not api_key or not avatar_id:
                anam_logger.warning(
                    "[ANAM] ANAM_API_KEY or ANAM_AVATAR_ID missing - skipping initialization"
                )
            else:
                if VERBOSE_LOGGING:
                    anam_logger.info("[ANAM] STEP 1 - Creating avatar session")
                avatar = anam.AvatarSession(
                    persona_config=anam.PersonaConfig(
                        name="Assistant Avatar",
                        avatarId=avatar_id,
                    ),
                )
                if VERBOSE_LOGGING:
                    anam_logger.info("[ANAM] STEP 2 - Starting avatar session")
                await avatar.start(session, room=ctx.room)
                if VERBOSE_LOGGING:
                    anam_logger.info("[ANAM] Avatar successfully initialized and joined room.")

        except Exception as e:
            anam_logger.exception(f"[ANAM ERROR] Avatar failed to initialize: {e}")
            raise
    else:
        if VERBOSE_LOGGING:
            anam_logger.info("[ANAM] Avatar integration temporarily disabled for debugging.")

    logger.info("[WORKER] Starting AgentSession...")
    try:
        # session.start() STARTS the session background loop and returns
        # almost immediately (~0.3s). The job continues to live for as long
        # as the caller (ipc/job_proc_lazy_main.py) does not signal shutdown.
        await session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(close_on_disconnect=False),
        )
        logger.info(
            "[WORKER] AgentSession session.start() returned (session loop now running in background). "
            "Waiting for actual end-of-job shutdown signal via shutdown_done.wait()..."
        )

        # THIS is the real lifetime wait: block until LiveKit's end-of-job
        # on_shutdown callback sets `shutdown_done` (see callback above).
        await shutdown_done.wait()

        logger.info(
            "[WORKER] shutdown_done received — end-of-job shutdown completed normally via on_shutdown callback."
        )
    except Exception as session_e:
        logger.exception(
            f"[WORKER ERROR] Exception during session lifetime (session.start + shutdown_done wait): {session_e}"
        )
        raise
    finally:
        logger.info("[WORKER] entrypoint finally-block: safety cleanup")

        # Safety-net only: if we arrived here WITHOUT shutdown_done ever being
        # set (e.g. unhandled exception propagated from session.start() or
        # from shutdown_done.wait() itself before on_shutdown ran), ensure
        # DittoAdapter still gets torn down. If on_shutdown already ran then
        # stop() is a no-op thanks to DittoAdapter's idempotency guard.
        if ditto_worker is not None and not shutdown_done.is_set():
            try:
                logger.warning(
                    "[DITTO] FINALLY SAFETY-NET: shutdown_done never set. "
                    "Stopping Ditto avatar worker from finally-block to avoid resource leaks. "
                    "(This path should only trigger on catastrophic exception, NOT on normal exit.)"
                )
                await ditto_worker.aclose()
                logger.info(
                    "[DITTO] Finally-block safety-net stop completed successfully."
                )
            except Exception as ditto_e:
                logger.exception(
                    f"[DITTO ERROR] Error stopping Ditto avatar worker in finally safety-net: {ditto_e}"
                )
        elif ditto_worker is not None and shutdown_done.is_set():
            logger.info(
                "[DITTO] No need to stop in finally-block; on_shutdown already handled cleanup "
                "(confirmed by shutdown_done.is_set()==True)."
            )



if __name__ == "__main__":
    logger.info("[WORKER] Starting LiveKit agent worker is starting up")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )

