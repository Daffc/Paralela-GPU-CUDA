#!/bin/bash

data=$(date +"%m-%d-%y%T")
printf "\n\nINICIANDO NOVA RODADA 'rainhas_GPU_2': %s\n" $data >> saidas_2
for i in {1..16}
do
    ./rainhas_GPU_2 $i >> saidas_2  
done