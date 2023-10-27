killed_process=false
pid=$(ps aux | grep "src/km_handler" | grep -v grep | awk '{print $2}')
if [ ! -z "$pid" ]; then
    kill $pid
    killed_process=true
else
    killed_process=false
fi
if [ "$killed_process" = true ]; then
    echo "Killed process"
else
    echo "No process to kill"
fi

while [ "$killed_process" = true ]; do
    sleep 1
    pid=$(ps aux | grep "src/km_handler" | grep -v grep | awk '{print $2}')
    if [ ! -z "$pid" ]; then
        kill $pid
        killed_process=true
    else
        killed_process=false
    fi
done
