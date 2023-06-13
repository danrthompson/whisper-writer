import traceback
import numpy as np
import openai
import os
import sounddevice as sd
import tempfile
import wave
import webrtcvad
from dotenv import load_dotenv

from typing import Any

if load_dotenv():
    openai.api_key = os.getenv("OPENAI_API_KEY")


def process_transcription(transcription, config=None):
    if config:
        if config["remove_trailing_period"] and transcription.endswith("."):
            transcription = transcription[:-1]
        if config["add_trailing_space"]:
            transcription += " "
        if config["remove_capitalization"]:
            transcription = transcription.lower()

    return transcription


"""
Record audio from the microphone and transcribe it using the OpenAI API.
Recording stops when the user stops speaking.
"""


def record_and_transcribe(status_queue, cancel_flag, stop_flag, config=None):
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
        if config["print_to_terminal"]:
            print("Recording finished. Size:", audio_data.size)

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

        # If configured, transcribe the temporary audio file using the OpenAI API
        # if config["use_api"]:
        api_options = config["api_options"]
        with open(temp_audio_file.name, "rb") as audio_file:
            response: Any = openai.Audio.transcribe(
                model=api_options["model"],
                file=audio_file,
                language=api_options["language"],
                prompt=api_options["initial_prompt"],
                temperature=api_options["temperature"],
            )

        # Remove the temporary audio file
        os.remove(temp_audio_file.name)

        if cancel_flag():
            status_queue.put(("cancel", ""))
            return ""

        result = response.get("text")
        if config["print_to_terminal"]:
            print("Transcription:", result)

        status_queue.put(("idle", ""))

        return process_transcription(result.strip(), config) if result else ""

    except Exception:
        traceback.print_exc()
        status_queue.put(("error", "Error"))
