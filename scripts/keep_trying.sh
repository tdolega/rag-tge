#!/bin/bash

cmd=$@

if [ -z "$cmd" ]; then
    echo "Usage: $0 <command>"
    exit 1
fi

while true; do
    $cmd && break
    echo "> command failed... retrying in 1 minute"
    sleep 60
done

echo "> command succeeded"
