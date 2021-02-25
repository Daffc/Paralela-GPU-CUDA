#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern "C" { 
    #include "helper.h"
}

// #define NTHREADS_TOTAL 1024
#define NTHREADS_TOTAL 4
#define TAM_ARRAY 100


// FUNÇÃO QUE COMPARA DUAS STRINGS, RETORNANDO 0 CASO SEJAM IGUAIS, OU A DIFERENÇA ENTRE OS CARACTERES QUE A DIFEREM (NEGATIVO SE 's1' < 's2', POSITIVO CASO CONTRÁRIO).
__device__ char strncmpCUDA(char *s1, char *s2, int size){
    
    register unsigned char u1, u2;

    while (size-- > 0)
      {
            // SEPARA CARACTERES
            u1 = (unsigned char) *s1++;
            u2 = (unsigned char) *s2++;

            // SE DIFERENTE, RETORNA DIFERENÇA
            if (u1 != u2)
                return u1 - u2;

            // CASO AMBAS STRINGS TENHA TERMINHADO, RETORNAR 0.
            if (u1 == '\0')
                return 0;
      }

    // CASO AMBAS STRINGS TENHAM OS "size" CARACTERES IGUAIS, RETORNAR 0;
    return 0;
}

__global__ void imprimeOrdem(long int *ordemPonteiros, long int tamanho){

    // VERIFICA SE PIXEL "i" NÃO EXTRAPOLA IMAGEM ORIGINAL.
    for(long int i = 0; i < tamanho; i++){
        printf("%ld\n", ordemPonteiros[i]);
    }  
}

__global__ void inicializaOrdemPonteiros(long int *ordemPonteiros, long int tamanho){
    long int i = blockDim.x * blockIdx.x + threadIdx.x;

    // VERIFICA SE PIXEL "i" NÃO EXTRAPOLA IMAGEM ORIGINAL.
    if(i < tamanho){
        ordemPonteiros[i] = i;
    }  
}
__global__ void printChar(char *saida, char  *entrada, long int tamanho){
  
    char rotacao[TAM_ARRAY];
    int j;

    long int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // VERIFICA SE PIXEL "i" NÃO EXTRAPOLA IMAGEM ORIGINAL.
    if(i < tamanho){
        // ACESSA COORDENADAS LINEARES DE PIXEL entrada[i * 3] E ARMAZENA SUAS DIMENSÕES (R, G, B) EM INTEIRO NA MATRIZ saida[i]
        saida[i] = entrada[i];
        printf("%c\n", saida[i]);
        for(j=0; j < tamanho; j++){
            rotacao[j] = entrada[(i + j) % tamanho];
        }

        rotacao[tamanho] = '\0';
        printf("%s\n", rotacao);

    }  
}

int main(int argc, char *argv[]) {
    
    FILE *f_in, *f_out;
    char *hostArquivoEntrada;
    char *hostArquivoSaida;
    char *deviceArquivoEntrada;
    char *deviceArquivoSaida;
    long int *ordemPonteiros;
    long int in_size;

    // TRATAMENTO DE ARGUMENTOS DE ENTRADA.
    identificaArquivosEntrada(&f_in, &f_out, argc, argv);

    // RECUPERA TAMANHO DO ARQUIVO DE ENTRADA.
    in_size = identificaTamanhoEntrada(f_in);

    // COPIA INFORMAÇÕES DE ARQUIVO DE ENTRADA PARA VETOR 'hostArquivoEntrada'.
    recuperaDadosEntrada(f_in, &hostArquivoEntrada, in_size);

    hostArquivoSaida = (char *) malloc(in_size * sizeof(char));

    // ALOCA MEMÓRIA PARA 'deviceArquivoEntrada' e 'deviceArquivoSaida'.
    cudaMalloc((void **)&deviceArquivoEntrada, in_size * sizeof(char));
    cudaMalloc((void **)&deviceArquivoSaida, in_size * sizeof(char));
    cudaMalloc((void **)&ordemPonteiros, in_size * sizeof(long int));
    

    cudaMemcpy(deviceArquivoEntrada, hostArquivoEntrada, in_size * sizeof(char), cudaMemcpyHostToDevice);


    // DEFINIDO DIMENSÕES DE GRID E BLOCK LINEARES KERNEL 'inicializaOrdemPonteiros'.
    dim3 DimGridIniciaOrdem((in_size-1)/NTHREADS_TOTAL + 1, 1, 1);
    dim3 DimBlockIniciaOrdem(NTHREADS_TOTAL, 1, 1);

    inicializaOrdemPonteiros<<<DimGridIniciaOrdem, DimBlockIniciaOrdem>>>(ordemPonteiros, in_size);

    imprimeOrdem<<<1, 1>>>(ordemPonteiros, in_size);

    dim3 DimGrid((in_size-1)/NTHREADS_TOTAL + 1, 1, 1);
    dim3 DimBlock(NTHREADS_TOTAL, 1, 1);
    // KERNEL DE TESTES
    printChar<<<DimGrid, DimBlock>>>(deviceArquivoSaida, deviceArquivoEntrada, in_size);

    
    cudaMemcpy(hostArquivoSaida, deviceArquivoSaida, in_size * sizeof(char), cudaMemcpyDeviceToHost);
    
    // ESCREVE 'in_size' BYTES EM ARQUIVO DE SAIDA 'f_out'.
    fwrite(hostArquivoSaida, in_size, 1, f_out);


    free(hostArquivoEntrada);
    free(hostArquivoEntrada);

    cudaFree(deviceArquivoEntrada);
    cudaFree(deviceArquivoSaida);
    cudaFree(ordemPonteiros);

    fclose(f_in);
    fclose(f_out);

    return 0;
}