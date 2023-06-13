import json
import os
import queue
import threading
from transcription import record_and_transcribe
import pyperclip

from typing import Optional


class ResultThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(ResultThread, self).__init__(*args, **kwargs)
        self.result = None
        self.stop_transcription = False
        self.cancel_transcription = False

    def run(self):
        self.result = self._target(
            *self._args,
            cancel_flag=lambda: self.cancel_transcription,
            stop_flag=lambda: self.stop_transcription,
            **self._kwargs
        )

    def stop(self):
        self.stop_transcription = True

    def cancel(self):
        self.cancel_transcription = True


class WhisperApp:
    def __init__(self):
        self.config = load_config_with_defaults()
        self.status_queue = queue.Queue()
        self.status_window = None
        self.recording_thread: Optional[ResultThread] = None

    def run(self):
        self.clear_status_queue()

        self.status_queue.put(("recording", "Recording..."))
        self.recording_thread = ResultThread(
            target=record_and_transcribe,
            args=(self.status_queue,),
            kwargs={"config": self.config},
        )
        self.recording_thread.start()

    def stop_or_cancel(self, stop):
        if not self.recording_thread:
            print("No recording to stop or cancel")
            return
        if stop:
            self.recording_thread.stop()
        else:
            self.recording_thread.cancel()

        self.recording_thread.join()

        transcribed_text = self.recording_thread.result

        print(transcribed_text)
        pyperclip.copy(transcribed_text)

    def stop(self):
        self.stop_or_cancel(True)

    def cancel(self):
        self.stop_or_cancel(False)

    def clear_status_queue(self):
        while not self.status_queue.empty():
            try:
                self.status_queue.get_nowait()
            except queue.Empty:
                break


def load_config_with_defaults():
    default_config = {
        "use_api": True,
        "api_options": {
            "model": "whisper-1",
            "language": None,
            "temperature": 0.0,
            "initial_prompt": None,
        },
        "local_model_options": {
            "model": "base",
            "device": None,
            "language": None,
            "temperature": 0.0,
            "initial_prompt": None,
            "condition_on_previous_text": True,
            "verbose": False,
        },
        "activation_key": "ctrl+alt+space",
        "silence_duration": 900,
        "writing_key_press_delay": 0.008,
        "remove_trailing_period": True,
        "add_trailing_space": False,
        "remove_capitalization": False,
        "print_to_terminal": True,
    }

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as config_file:
            user_config = json.load(config_file)
            default_config.update(user_config)

    return default_config
