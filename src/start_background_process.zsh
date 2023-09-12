source ~/Code/Tools/cloned/whisper-writer/.venv/bin/activate
nohup python -u ~/Code/Tools/cloned/whisper-writer/src/km_handler.py >> ~/Code/Tools/cloned/whisper-writer/nohup.out 2>&1 &
cd ~/Code/Tools/cloned/whisper-writer
tail -f nohup.out
