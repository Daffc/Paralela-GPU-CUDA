#!/bin/bash

data=$(date +"%m-%d-%y%T")
printf "\n\nINICIANDO NOVA RODADA 'rainhas_GPU_3': %s\n" $data >> saidas_3
for i in {1..16}
do
    ./rainhas_GPU_3 $i >> saidas_3
done
