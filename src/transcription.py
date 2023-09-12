import os
import tempfile
import traceback
import wave
import json
from typing import Any, Callable, Optional

from deepgram import Deepgram
import numpy as np
import openai
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv

DEBUG = False

if load_dotenv():
    openai.api_key = os.getenv("OPENAI_API_KEY")


def process_transcription(transcription: str, config=None) -> str:
    if config:
        if config["remove_trailing_period"] and transcription.endswith("."):
            transcription = transcription[:-1]
        if config["add_trailing_space"]:
            transcription += " "
        if config["remove_capitalization"]:
            transcription = transcription.lower()

    return transcription


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
    sample_rate = 16000
    frame_duration = 30  # 30ms, supported values: 10, 20, 30
    buffer_duration = 300  # 300ms
    silence_duration = config["silence_duration"] if config else 900  # 900ms

    vad = webrtcvad.Vad(3)  # Aggressiveness mode: 3 (highest)
    buffer = []
    recording = []
    num_silent_frames = 0
    num_buffer_frames = buffer_duration // frame_duration
    num_silence_frames = silence_duration // frame_duration
    try:
        if config["print_to_terminal"]:
            print("Recording...")
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=sample_rate * frame_duration // 1000,
            callback=lambda indata, frames, time, status: buffer.extend(indata[:, 0]),
        ):
            while not cancel_flag() and not stop_flag():
                if len(buffer) < sample_rate * frame_duration // 1000:
                    continue

                frame = buffer[: sample_rate * frame_duration // 1000]
                buffer = buffer[sample_rate * frame_duration // 1000 :]

                is_speech = vad.is_speech(np.array(frame).tobytes(), sample_rate)
                if is_speech:
                    recording.extend(frame)
                    num_silent_frames = 0
                else:
                    if len(recording) > 0:
                        num_silent_frames += 1

                    if num_silent_frames >= num_silence_frames:
                        if config and config["wait_for_stop_function"]:
                            continue
                        else:
                            break

        if cancel_flag():
            status_queue.put(("cancel", ""))
            return ""

        audio_data = np.array(recording, dtype=np.int16)
        audio_data_size = audio_data.size
        total_samples = audio_data_size // 2
        recording_duration_seconds = total_samples / sample_rate
        if config["print_to_terminal"]:
            print(
                f"Recording finished. Size: {audio_data_size} bytes. Duration: {recording_duration_seconds} seconds"
            )

        if recording_duration_seconds / 60 > 15:
            print(
                f"Recording too long. Duration: {recording_duration_seconds / 60} mins is greater than 15 minutes"
            )
            status_queue.put(("error", "Recording too long"))
            return ""

        # Save the recorded audio as a temporary WAV file on disk
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
        ) as temp_audio_file:
            with wave.open(temp_audio_file.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes (16 bits) per sample
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())

        status_queue.put(("transcribing", "Transcribing..."))
        if config["print_to_terminal"]:
            print("Transcribing audio file...")

        # # If configured, transcribe the temporary audio file using the OpenAI API
        # # if config["use_api"]:
        # api_options = config["api_options"]
        # prompt = prompt_override or api_options["initial_prompt"]
        # with open(temp_audio_file.name, "rb") as audio_file:
        #     response: Any = openai.Audio.transcribe(
        #         model=api_options["model"],
        #         file=audio_file,
        #         language=api_options["language"],
        #         prompt=prompt,
        #         temperature=api_options["temperature"],
        #     )

        if DEBUG:
            print("Transcribing with Deepgram...")
        # Transcribe with Deepgram
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise Exception("Deepgram API key not found")

        print(api_key)
        deepgram = Deepgram(api_key)
        deepgram_options = {
            "tier": "nova",
            "punctuate": True,
            "smart_format": True,
            "summarize": "false",
            "measurements": True,
            # "dictation": True,
        }
        with open(temp_audio_file.name, "rb") as audio:
            source = {"buffer": audio, "mimetype": "audio/wav"}
            response = deepgram.transcription.sync_prerecorded(source, deepgram_options)
            if DEBUG:
                print(json.dumps(response, indent=4))
        # Remove the temporary audio file
        os.remove(temp_audio_file.name)

        if cancel_flag():
            status_queue.put(("cancel", ""))
            return ""

        try:
            result = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        except Exception as exc:
            print("Error parsing response from Deepgram. Error:", exc)
            return ""

        if config["print_to_terminal"]:
            print("Transcription:", result)

        status_queue.put(("idle", ""))

        return process_transcription(result.strip(), config) if result else ""

    except Exception:
        traceback.print_exc()
        status_queue.put(("error", "Error"))
        return ""
