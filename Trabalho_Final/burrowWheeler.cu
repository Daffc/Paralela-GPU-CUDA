#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern "C" { 
    #include "helper.h"
}


// #define NTHREADS_TOTAL 1024
#define NTHREADS_TOTAL 4
#define BITONIC_BLOCK 4    // DEFINE A QUANTIDADE DE STRINGS QUE SERÃO ORDENADAS POR CADA CHAMADA DE BITONIC SORT (SENDO OBRIGATORIAMENTE UMA POTÊNCIA DE 2)
#define TAM_STR_MAX 100
#define TAM_CHAVE 4

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

// FUNÇÃO INICIALIZA VETOR 'orvemPonteiros' EM ORDEM CRESCENTE, DE 0 À  tamanho-1.
__global__ void inicializaOrdemPonteiros(long int *ordemPonteiros, long int tamanho){
    long int i = blockDim.x * blockIdx.x + threadIdx.x;

    // VERIFICA SE PIXEL "i" NÃO EXTRAPOLA IMAGEM ORIGINAL.
    if(i < tamanho){
        ordemPonteiros[i] = i;
    }  
}

__global__ void bitonicSortCUDA(char  *entrada, long int *ordemPonteiros, long int tamanho){

    long int posicao = blockDim.x * blockIdx.x + threadIdx.x;

    // APLICA BITONIC SORT SOMENTES SE HOUVEREM ELEMENTOS O SUFICIENTE PARA APLICAÇÃO.
    if(posicao < (tamanho/BITONIC_BLOCK)){
        int i,
            j;

        // ALOCA VETOR PARA STRINGS.
        char strings[BITONIC_BLOCK][TAM_STR_MAX];

        // APLICA VALOR DE CHAVES PARA 'strings[i]' DE ACORDO COM O PONTEIRO PARA ARQUELA POSIÇÃO INDICADO POR 'ordemPonteiros[posicao * BITONIC_BLOCK]'.
        for(j = 0; j < BITONIC_BLOCK; j++){
            for(i = 0; i < TAM_CHAVE; i++){
                strings[j][i] = entrada[(ordemPonteiros[(posicao * BITONIC_BLOCK)] + i + j) % tamanho];
            }   
            strings[j][i] = '\0';
        }         

        int k, l, stringSize;
        int cmp_ret, aux;
        int subordem[BITONIC_BLOCK];

        for(j = 0; j < BITONIC_BLOCK; j++){
            subordem[j] = ordemPonteiros[(posicao * BITONIC_BLOCK) + j];
        }

        for (k = 2; k <= BITONIC_BLOCK; k = 2*k) {
            for (j = k>>1; j > 0; j = j>>1) {
                for (i= 0; i < BITONIC_BLOCK; i++) {
                int ixj = i^j;
                    if ((ixj) > i) {
                        stringSize = TAM_CHAVE; 

                        while ((cmp_ret = strncmpCUDA(strings[subordem[i] % BITONIC_BLOCK], strings[subordem[ixj] % BITONIC_BLOCK], stringSize)) == 0){                            
                                                        
                            for(l = stringSize; l < stringSize + TAM_CHAVE; l++){
                                strings[subordem[i] % BITONIC_BLOCK][l] = entrada[(subordem[i] + l) % tamanho];
                                strings[subordem[ixj] % BITONIC_BLOCK][l] = entrada[(subordem[ixj] + l) % tamanho];
                            }  
                            strings[subordem[i] % BITONIC_BLOCK][l] = strings[subordem[ixj] % BITONIC_BLOCK][l] = '\0'; 
                            printf("PÓS COLISAO= %d  %s %d  %s \t %d\n",subordem[i], strings[subordem[i] % BITONIC_BLOCK], subordem[ixj], strings[subordem[ixj] % BITONIC_BLOCK], l);
                            
                            stringSize += TAM_CHAVE;
                        }


                        if ((i&k) == 0 &&  cmp_ret > 0){                            
                            aux = subordem[i];
                            subordem[i] = subordem[ixj];
                            subordem[ixj] = aux;
                        } 
                        if ((i&k)!=0 && cmp_ret < 0){
                            aux = subordem[i];
                            subordem[i] = subordem[ixj];
                            subordem[ixj] = aux;
                        }
                    }
                }
            }
        }

        
        for(j = 0; j < BITONIC_BLOCK; j++){
            ordemPonteiros[(posicao * BITONIC_BLOCK) + j] = subordem[j];
        }
        printf("%d %d %d %d\nBLOCO: %d, RESULTADO: %s, %s, %s, %s\n" , subordem[0],subordem[1], subordem[2], subordem[3], posicao * BITONIC_BLOCK, strings[0], strings[1], strings[2], strings[3]);
    }
}

__global__ void printChar(char *saida, char  *entrada, long int tamanho){
  
    char rotacao[TAM_STR_MAX];
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

// ========================================================
// ==================      DEBUG     ======================
// ========================================================

__global__ void imprimeOrdem(long int *ordemPonteiros, long int tamanho){

    // VERIFICA SE PIXEL "i" NÃO EXTRAPOLA IMAGEM ORIGINAL.
    for(long int i = 0; i < tamanho; i++){
        printf("%ld\n", ordemPonteiros[i]);
    }  
}


// ========================================================
// ==================      MAIN      ======================
// ========================================================
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

    dim3 DimGrid(((in_size / BITONIC_BLOCK)-1)/NTHREADS_TOTAL + 1, 1, 1);
    dim3 DimBlock(NTHREADS_TOTAL, 1, 1);
    // KERNEL DE TESTES
    printf("GRID %d, BLOCK %d\n", DimGrid.x, DimBlock.x);
    bitonicSortCUDA<<<DimGrid, DimBlock>>>(deviceArquivoEntrada, ordemPonteiros, in_size);
    
    imprimeOrdem<<<1, 1>>>(ordemPonteiros, in_size);

    cudaError_t err = cudaGetLastError();        // Get error code

    if ( err != cudaSuccess )
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));
       exit(-1);
    }

    
    cudaMemcpy(hostArquivoSaida, deviceArquivoSaida, in_size * sizeof(char), cudaMemcpyDeviceToHost);
    
    // ESCREVE 'in_size' BYTES EM ARQUIVO DE SAIDA 'f_out'.
    fwrite(hostArquivoSaida, in_size, 1, f_out);


    free(hostArquivoEntrada);
    free(hostArquivoSaida);

    cudaFree(deviceArquivoEntrada);
    cudaFree(deviceArquivoSaida);
    cudaFree(ordemPonteiros);

    fclose(f_in);
    fclose(f_out);

    return 0;
}