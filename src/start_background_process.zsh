source ~/Code/Tools/whisper-writer/venv/bin/activate
nohup python -u ~/Code/Tools/whisper-writer/src/km_handler.py &
cd ~/Code/Tools/whisper-writer
tail -f nohup.out
