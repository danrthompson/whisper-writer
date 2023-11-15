from pyexpat import model
from whisper_app import WhisperApp

import time
import os

COMMAND_FILE = "/Users/danthompson/Code/Tools/cloned/whisper-writer/command.txt"
RESULT_FILE = "/Users/danthompson/Code/Tools/cloned/whisper-writer/result.txt"


def set_results(result):
    with open(RESULT_FILE, "w") as file:
        file.write(result)


app = WhisperApp()

while True:
    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)
    if os.path.exists(COMMAND_FILE):
        with open(COMMAND_FILE, "r") as file:
            command = file.read().strip()
        os.remove(COMMAND_FILE)

        if command == "start":
            app.run()
            set_results("done")
        elif command == "stop":
            app.stop()
            set_results("done")
        elif command == "cancel":
            app.cancel()
            set_results("done")
        elif command == "update configs":
            app.update_configs()
            set_results("done")
        elif command.startswith("update model"):
            split_command = command.split("|", 1)
            if len(split_command) == 1:
                print("ERROR: No valid model name provided. Taking no action")
                continue
            model_name = split_command[1].strip()
            app.update_model(model_name)
            set_results("done")
        elif command.startswith("update prompt"):
            split_command = command.split("|", 1)
            if len(split_command) == 1:
                print("ERROR: No valid prompt name provided. Taking no action")
                continue
            prompt_filename = split_command[1].strip()
            if prompt_filename.startswith("custom="):
                custom_prompt = prompt_filename.replace("custom=", "")
                app.update_prompt(custom_prompt)
            else:
                app.update_prompt_from_filename(prompt_filename)
            set_results("done")
        else:
            print("Unknown command. Taking no action")

    time.sleep(0.5)  # prevent busy-waiting
