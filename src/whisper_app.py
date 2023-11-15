import json
import os
import threading
from datetime import datetime
from queue import Empty, Queue
from typing import Optional, Tuple

import pyperclip

from transcription import record_and_transcribe


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

        print(
            f"Initialized WhisperApp. Time: {datetime.now().strftime('%H:%M')}. Config: {self.config}"
        )

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

        return None

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

    def update_model(self, model_name):
        self.config["model"] = model_name
        print(f"Updated model to: {model_name}")

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
    config = {}
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as config_file:
            user_config = json.load(config_file)
            config = user_config

    if not config:
        raise RuntimeError("No config found")

    return config
