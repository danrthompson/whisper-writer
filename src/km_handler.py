from whisper_app import WhisperApp

import time
import os

COMMAND_FILE = "/Users/danthompson/Code/Tools/whisper-writer/command.txt"
RESULT_FILE = "/Users/danthompson/Code/Tools/whisper-writer/result.txt"


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
        elif command == "update":
            app.update_configs()
            set_results("done")

    time.sleep(0.5)  # prevent busy-waiting
