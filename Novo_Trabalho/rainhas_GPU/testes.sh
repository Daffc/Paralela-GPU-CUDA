#!/bin/bash

data=$(date +"%m-%d-%y%T")
printf "\n\nINICIANDO NOVA RODADA 'rainhas_GPU': %s\n" $data >> saidas
for i in {1..17}
do
    ./rainhas_GPU $i >> saidas
done