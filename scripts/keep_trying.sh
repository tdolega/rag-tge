#!/bin/bash

cmd=$@

if [ -z "$cmd" ]; then
    echo "Usage: $0 <command>"
    exit 1
fi

RETRY_SECONDS=10

while true; do
    $cmd && break
    echo "> command failed... retrying in $RETRY_SECONDS seconds"
    sleep $RETRY_SECONDS
done

echo "> command succeeded"
