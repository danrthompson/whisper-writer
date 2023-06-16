source ~/Code/Tools/whisper-writer/.venv/bin/activate
nohup python -u ~/Code/Tools/whisper-writer/src/km_handler.py >> ~/Code/Tools/whisper-writer/nohup.out 2>&1 &
cd ~/Code/Tools/whisper-writer
tail -f nohup.out
