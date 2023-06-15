import json
import os
from queue import Empty, Queue
import threading
from transcription import record_and_transcribe
import pyperclip

from typing import Optional, Tuple


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
            **self._kwargs,
        )

    def stop(self):
        self.stop_transcription = True

    def cancel(self):
        self.cancel_transcription = True


class WhisperApp:
    def __init__(self):
        self.config = load_config_with_defaults()
        self.status_queue: Queue[Tuple[str, str]] = Queue()
        self.status_window = None
        self.recording_thread: Optional[ResultThread] = None

    def run(self, prompt_override: Optional[str] = None):
        self.clear_status_queue()

        kwargs = {"config": self.config}
        if prompt_override:
            kwargs["prompt_override"] = prompt_override
            print(f"Using prompt override: {prompt_override}")
        else:
            print(f"Using prompt: {self.config['api_options']['initial_prompt']}")

        self.status_queue.put(("recording", "Recording..."))
        self.recording_thread = ResultThread(
            target=record_and_transcribe,
            args=(self.status_queue,),
            kwargs=kwargs,
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

        pyperclip.copy(transcribed_text)

    def stop(self):
        self.stop_or_cancel(True)

    def cancel(self):
        self.stop_or_cancel(False)
        print("Cancelled")

    def clear_status_queue(self):
        while not self.status_queue.empty():
            try:
                self.status_queue.get_nowait()
            except Empty:
                break

    def update_configs(self):
        self.config = load_config_with_defaults()
        print("Updated configs")

    def update_prompt(self, new_prompt):
        self.config["api_options"]["initial_prompt"] = new_prompt
        print(f"Updated prompt to: {new_prompt}")

    def update_prompt_from_filename(self, prompt_filename):
        self.update_configs()
        with open(
            os.path.join(os.path.dirname(__file__), f"prompts/{prompt_filename}.txt"),
            "r",
        ) as prompt_file:
            new_prompt: str = prompt_file.read()
        if new_prompt:
            self.update_prompt(new_prompt)
        else:
            print(
                f"Prompt file is empty. Did not update. Current prompt: {self.config['api_options']['initial_prompt']}"
            )


def load_config_with_defaults():
    default_config = {
        "use_api": True,
        "wait_for_stop_function": True,
        "api_options": {
            "model": "whisper-1",
            "language": None,
            "temperature": 0.0,
            "initial_prompt": None,
        },
        "silence_duration": 900,
        "remove_trailing_period": True,
        "add_trailing_space": False,
        "remove_capitalization": False,
        "print_to_terminal": False,
    }

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as config_file:
            user_config = json.load(config_file)
            default_config |= user_config

    return default_config
