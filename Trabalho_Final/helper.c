#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// TRATA DE ENTRADA DE ARGUMENTOS, IDENTIFICANDO ARQUIVO DE ENTRADA ('in') E SAIDA ('out').
void identificaArquivosEntrada(FILE **in, FILE **out, int argc, char *argv[]){
    int i = 1;
    if(argc > 4){
        while(i < argc){
            if(!strcmp(argv[i], "-i")){
                i++;
                *in = fopen(argv[i], "r");
                if(!(*in)){
                    printf("Erro ao acessar arquivo de entrada '%s'.\n", argv[i]);
                    exit(1);
                }
                i++;
            }
            else
                if(!strcmp(argv[i], "-o")){
                    i++;
                    *out = fopen(argv[i], "w");
                    if(!(*out)){
                        printf("Erro ao acessar arquivo de saida '%s'.\n", argv[i]);
                        exit(1);
                    }
                    i++;
                }
                else{
                    printf("Arrgumento '%s' inapropriado. Terminando Programa.\n", argv[i]);
                    exit(1);
                }
        }
    }
    else{
        printf("Parametros '-i arq_entrada -o arq_saida' são obrigatórios.\n");
        exit(1);  
    }
}

// RETORNA O TAMANHO, EM BYTES, DO ARQUIVO 'f_in'.
long int identificaTamanhoEntrada(FILE *f_in){

    long int file_size;

    fseek(f_in, 0L, SEEK_END);
    file_size = ftell(f_in);

    printf("Arquivo de entrada possui %ld Bytes.\n", file_size);

    rewind(f_in);
    return file_size;
}

// RECUPERANDO DADOS DE ARQUIVO DE ENTRADA 'f_in' E ARMAZENANDO-OS EM 'vetor_entrada[]', SENDO 'tamanho' A QUANTIDADE DE BYTES EM 'f_in'.
void recuperaDadosEntrada(FILE *f_in, char *vetor_entrada[], long int tamanho){

    *vetor_entrada = (char *) malloc(tamanho + 1);
    if(!(*vetor_entrada)){
        printf("Erro ao alocar memória para Arquivo de entrada.\n");
    }
    
    for(int i = 0; i < tamanho; i++){
        (*vetor_entrada)[i] = fgetc(f_in);
    }

    rewind(f_in);
}