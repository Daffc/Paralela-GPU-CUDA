#/bin/bash

mkdir -p saidas

for entry in ./*
do
    if [ "$entry" != "$0" ] && [ "$entry" != "./saidas" ]; then
        echo "Iniciando $entry..."
        $entry > "./saidas/${entry:2}"
        echo "OK!"
    fi
done

