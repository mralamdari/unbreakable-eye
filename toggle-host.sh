#!/bin/bash
current=$(grep "^POSTGRES_HOST=" .env | cut -d= -f2 | tr -d ' ')
if [ "$current" = "localhost" ]; then
    sed -i 's/^POSTGRES_HOST=localhost/POSTGRES_HOST=postgres/' .env
    echo "Switched to Docker mode (POSTGRES_HOST=postgres)"
else
    sed -i 's/^POSTGRES_HOST=postgres/POSTGRES_HOST=localhost/' .env
    echo "Switched to local mode (POSTGRES_HOST=localhost)"
fi
