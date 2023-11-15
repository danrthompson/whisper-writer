import json
import logging
import os
import tempfile
import wave
from abc import ABC, abstractmethod
from enum import Enum, auto
from time import time
from typing import Any, Callable, Optional, Type

import assemblyai as aai
import numpy as np
import openai
import sounddevice as sd
import webrtcvad
from assemblyai import LanguageCode, TranscriptionConfig
from deepgram import Deepgram
from deepgram.transcription import PrerecordedOptions
from deepgram._types import BufferSource
from numpy.typing import NDArray

# Constants and Environment Variables
DEBUG = False
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")

SAMPLE_RATE = 16000

if not (OPENAI_API_KEY and DEEPGRAM_API_KEY and ASSEMBLYAI_API_KEY):
    missing_keys = [
        key
        for key in ["OPENAI_API_KEY", "DEEPGRAM_API_KEY", "ASSEMBLYAI_API_KEY"]
        if not os.getenv(key)
    ]
    raise RuntimeError(f"API keys not found. Missing keys: {', '.join(missing_keys)}")

# Initialize logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    NOVA = "nova"
    OPENAI_WHISPER = "openai_whisper"
    DEEPGRAM_WHISPER = "deepgram_whisper"
    ASSEMBLYAI = "assemblyai"


class AbstractTranscriptionModel(ABC):
    _registry: dict[ModelType, Type["AbstractTranscriptionModel"]] = {}

    @classmethod
    def register_model(cls, enum_value: ModelType):
        def decorator(subclass: Type["AbstractTranscriptionModel"]):
            cls._registry[enum_value] = subclass
            return subclass

        return decorator

    @classmethod
    def transcribe_for_model(
        cls, model_type: ModelType, audio_file: str, config: Optional[dict] = None
    ) -> str:
        if not config:
            config = {}
        model_class = cls._registry.get(model_type)
        if not model_class:
            raise ValueError(f"Unsupported model type: {model_type}")
        model_instance = model_class()
        return model_instance.transcribe(audio_file, config)

    @abstractmethod
    def transcribe(self, audio_file: str, config: dict) -> str:
        """Transcribe an audio file using the specific model."""
        raise NotImplementedError()


@AbstractTranscriptionModel.register_model(ModelType.OPENAI_WHISPER)
class OpenAIWhisperTranscription(AbstractTranscriptionModel):
    def __init__(self):
        openai.api_key = OPENAI_API_KEY

    def transcribe(self, audio_file: str, config: dict) -> str:
        logger.info("Transcribing with OpenAI...")
        # If configured, transcribe the temporary audio file using the OpenAI API
        # if config["use_api"]:
        api_options = config["api_options"]
        prompt = api_options.get("prompt_override", "") or api_options.get(
            "initial_prompt", ""
        )

        with open(audio_file, "rb") as f:
            response: Any = openai.Audio.transcribe(
                model=api_options["model"],
                file=f,
                language=api_options["language"],
                prompt=prompt,
                temperature=api_options["temperature"],
            )

        result: str = response.get("text", "")

        return result


@AbstractTranscriptionModel.register_model(ModelType.ASSEMBLYAI)
class AssemblyAITranscription(AbstractTranscriptionModel):
    def __init__(self):
        aai.settings.api_key = ASSEMBLYAI_API_KEY

    def transcribe(self, audio_file: str, config: dict) -> str:
        logger.info("Transcribing with AssemblyAI...")

        # replace with your API token

        # create a Transcriber object
        transcriber = aai.Transcriber()
        aai_config = TranscriptionConfig(
            language_code=LanguageCode.en_us,
            punctuate=True,
            format_text=True,
            dual_channel=False,
            speaker_labels=False,
            content_safety=False,
            iab_categories=False,
            language_detection=False,
            disfluencies=False,
            sentiment_analysis=False,
            auto_chapters=False,
            summarization=False,
            entity_detection=False,
        )

        transcript = transcriber.transcribe(audio_file, aai_config)

        return transcript.text or ""


class DeepgramTranscriber(AbstractTranscriptionModel):
    def __init__(self):
        self.deepgram = Deepgram(DEEPGRAM_API_KEY)

    def transcribe(self, audio_file: str, config: dict) -> str:
        logger.info(f"Transcribing with {self.name}...")

        deepgram_options = self.deepgram_options

        with open(audio_file, "rb") as f:
            source = BufferSource(buffer=f.read(), mimetype="audio/wav")
            response = self.deepgram.transcription.sync_prerecorded(
                source, deepgram_options
            )
            if DEBUG:
                logger.debug(json.dumps(response, indent=4))

        try:
            result = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        except Exception:
            logger.exception("Error parsing response from Deepgram")
            return ""
        return result

    @property
    @abstractmethod
    def deepgram_options(self) -> PrerecordedOptions:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


@AbstractTranscriptionModel.register_model(ModelType.NOVA)
class NovaTranscription(DeepgramTranscriber):
    @property
    def name(self) -> str:
        return "Nova"

    @property
    def deepgram_options(self) -> PrerecordedOptions:
        return PrerecordedOptions(
            model="general",
            tier="nova",
            language="en-US",
            punctuate=True,
            smart_format=True,
            summarize=False,
            measurements=True,
        )


@AbstractTranscriptionModel.register_model(ModelType.DEEPGRAM_WHISPER)
class DeepgramWhisperTranscription(DeepgramTranscriber):
    @property
    def name(self) -> str:
        return "Deepgram Whisper"

    @property
    def deepgram_options(self) -> PrerecordedOptions:
        return PrerecordedOptions(
            model="whisper-large",
            language="en-US",
            punctuate=True,
            smart_format=True,
            summarize=False,
            measurements=True,
        )


def process_transcription(transcription: str, config=None) -> str:
    if config:
        if config["remove_trailing_period"] and transcription.endswith("."):
            transcription = transcription[:-1]
        if config["add_trailing_space"]:
            transcription += " "
        if config["remove_capitalization"]:
            transcription = transcription.lower()

    return transcription


def record_audio(
    status_queue,
    cancel_flag: Callable[[], bool],
    stop_flag: Callable[[], bool],
    config=None,
) -> Optional[NDArray[np.int16]]:
    frame_duration = 30  # 30ms, supported values: 10, 20, 30
    silence_duration = config["silence_duration"] if config else 900  # 900ms

    vad = webrtcvad.Vad(3)  # Aggressiveness mode: 3 (highest)
    buffer = []
    recording = []
    num_silent_frames = 0
    num_silence_frames = silence_duration // frame_duration
    logger.info("Recording...")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=SAMPLE_RATE * frame_duration // 1000,
        callback=lambda indata, frames, time, status: buffer.extend(indata[:, 0]),
    ):
        while not cancel_flag() and not stop_flag():
            if len(buffer) < SAMPLE_RATE * frame_duration // 1000:
                continue

            frame = buffer[: SAMPLE_RATE * frame_duration // 1000]
            buffer = buffer[SAMPLE_RATE * frame_duration // 1000 :]

            is_speech = vad.is_speech(np.array(frame).tobytes(), SAMPLE_RATE)
            if is_speech:
                recording.extend(frame)
                num_silent_frames = 0
            else:
                if len(recording) > 0:
                    num_silent_frames += 1

                if num_silent_frames >= num_silence_frames:
                    pass

    if cancel_flag():
        status_queue.put(("cancel", ""))
        return None

    audio_data = np.array(recording, dtype=np.int16)
    return audio_data


def check_audio_duration(
    audio_data: NDArray[np.int16], status_queue
) -> tuple[bool, float]:
    audio_data_size = audio_data.size
    total_samples = audio_data_size // 2
    recording_duration_seconds = total_samples / SAMPLE_RATE
    logger.info(
        "Recording finished. Size: %s bytes. Duration: %s seconds",
        audio_data_size,
        recording_duration_seconds,
    )

    if recording_duration_seconds > 900:
        logger.error(
            "Recording too long. Duration: %s mins is greater than 15 minutes",
            recording_duration_seconds / 60,
        )
        status_queue.put(("error", "Recording too long"))
        return (False, recording_duration_seconds)

    return (True, recording_duration_seconds)


def save_audio_to_temp_file(audio_data: NDArray[np.int16]) -> str:
    # Save the recorded audio as a temporary WAV file on disk
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        with wave.open(temp_audio_file.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes (16 bits) per sample
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

    return temp_audio_file.name


def get_model_type(config):
    model_type: Optional[ModelType] = None
    model_str: str = config.get("model")
    if model_str:
        try:
            model_type = ModelType(model_str)
        except ValueError:
            logger.exception("Invalid model type: %s", config.get("model"))

    if not model_type:
        logger.warning("No model type specified. Using default model.")
        model_type = ModelType.NOVA

    return model_type


def get_transcription(
    model_type: ModelType, audio_file_name: str, status_queue, config: dict
) -> str:
    status_queue.put(("transcribing", "Transcribing..."))

    result = AbstractTranscriptionModel.transcribe_for_model(
        model_type, audio_file_name
    )

    # Remove the temporary audio file
    os.remove(audio_file_name)

    return result


def record_and_transcribe(
    status_queue,
    cancel_flag: Callable[[], bool],
    stop_flag: Callable[[], bool],
    config=None,
    prompt_override: Optional[str] = None,
) -> str:
    """
    Record audio from the microphone and transcribe it using the OpenAI API.
    Recording stops when the user stops speaking.
    """

    if not config:
        config = {}
    config["prompt_override"] = prompt_override

    # Record audio
    audio_data = record_audio(status_queue, cancel_flag, stop_flag, config)
    if audio_data is None:
        logger.warning("No audio data recorded. Possible error")
        return ""

    # returns false if it took too long
    acceptable_duration, recording_duration = check_audio_duration(
        audio_data, status_queue
    )
    if not acceptable_duration:
        return ""

    audio_file_name = save_audio_to_temp_file(audio_data)

    model_type = get_model_type(config)

    start_time = time()
    result = get_transcription(model_type, audio_file_name, status_queue, config)
    end_time = time()

    duration = end_time - start_time

    if cancel_flag():
        status_queue.put(("cancel", ""))
        return ""

    logger.info(
        "Transcription finished. Recording duration: %.1f. Transcription duration: %.1f. Model: %s Result: %s",
        recording_duration,
        duration,
        model_type.name,
        result,
    )

    status_queue.put(("idle", ""))

    return process_transcription(result.strip(), config) if result else ""
