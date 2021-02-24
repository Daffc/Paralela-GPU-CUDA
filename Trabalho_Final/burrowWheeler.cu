#include <stdlib.h>
#include <stdio.h>
#include <string.h>


#define NTHREADS_TOTAL 1024

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


__global__ void printChar(char *saida, char  *entrada, int tamanho){
  
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // VERIFICA SE PIXEL "i" NÃO EXTRAPOLA IMAGEM ORIGINAL.
    if(i < tamanho){
      // ACESSA COORDENADAS LINEARES DE PIXEL entrada[i * 3] E ARMAZENA SUAS DIMENSÕES (R, G, B) EM INTEIRO NA MATRIZ saida[i]
      saida[i] = entrada[i];
      printf("%c\n", saida[i]);
    }  
    else{
        printf("-\n");
    }
  }

int main(int argc, char *argv[]) {
    
    FILE *f_in, *f_out;
    char *hostArquivoEntrada;
    char *hostArquivoSaida;
    char *deviceArquivoEntrada;
    char *deviceArquivoSaida;
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
    

    cudaMemcpy(deviceArquivoEntrada, hostArquivoEntrada, in_size * sizeof(char), cudaMemcpyHostToDevice);


    // DEFINIDO DIMENSÕES DE GRID E BLOCK LINEARES PARA KERNELS rgb2uintKernelSHM E uint2rgbKernelSHM.
    dim3 DimGrid((in_size-1)/NTHREADS_TOTAL + 1, 1, 1);
    dim3 DimBlock(NTHREADS_TOTAL, 1, 1);

    // EFETUANDO TRANSIÇÃO DE CHAR -> INT
    printChar<<<DimGrid, DimBlock>>>(deviceArquivoSaida, deviceArquivoEntrada, in_size);

    
    cudaMemcpy(hostArquivoSaida, deviceArquivoSaida, in_size * sizeof(char), cudaMemcpyDeviceToHost);
    
    // ESCREVE 'in_size' BYTES EM ARQUIVO DE SAIDA 'f_out'.
    fwrite(hostArquivoSaida, in_size, 1, f_out);


    free(hostArquivoEntrada);
    free(hostArquivoEntrada);

    cudaFree(deviceArquivoEntrada);
    cudaFree(deviceArquivoSaida);

    fclose(f_in);
    fclose(f_out);

    return 0;
}