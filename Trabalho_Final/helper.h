#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef  HELPER
#define  HELPER

    // TRATA DE ENTRADA DE ARGUMENTOS, IDENTIFICANDO ARQUIVO DE ENTRADA ('in') E SAIDA ('out').
    void identificaArquivosEntrada(FILE **in, FILE **out, int argc, char *argv[]);

    // RETORNA O TAMANHO, EM BYTES, DO ARQUIVO 'f_in'.
    long int identificaTamanhoEntrada(FILE *f_in);

    // RECUPERANDO DADOS DE ARQUIVO DE ENTRADA 'f_in' E ARMAZENANDO-OS EM 'vetor_entrada[]', SENDO 'tamanho' A QUANTIDADE DE BYTES EM 'f_in'.
    void recuperaDadosEntrada(FILE *f_in, char *vetor_entrada[], long int tamanho);

#endif