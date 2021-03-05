#!/bin/bash

data=$(date +"%m-%d-%y%T")
printf "\n\nINICIANDO NOVA RODADA 'saida_serial': %s\n" $data >> saidas_serial
for i in {1..16}
do
    ./rainhas_iterativo $i >> saidas_serial
done