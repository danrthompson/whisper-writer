pid=$(ps aux | grep "src/km_handler" | grep -v grep | awk '{print $2}')
if [ ! -z "$pid" ]; then kill $pid; echo "Process killed"; else echo "Process not found"; fi
