#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>


#include <cstdio>

struct functor
{
  __host__ __device__
  void operator()(int val)
  {
      printf("Call for value : %d\n", val);
  }
};


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define NTHREADS 1024
// #define NTHREADS 8

struct timeval timestamp(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return tp;
}

__device__ int nao_valido(unsigned char * tabuleiro, int linha, int coluna)
{
    int i;

    // Verifica existência de rainhas na lateral e diagonais à esquerdas.
    for (i = coluna - 1; i >= 0; i--)
        if (tabuleiro[i] == linha || (abs(tabuleiro[i] - linha) == coluna - i))
            return 1;
    return 0;
}


__device__ int resolve_tabuleiro(unsigned char * tabuleiro, int rainhas)
{
    int solucoes = 0, coluna = 3, linha = 0;

    // Enquanto não encontrou distribuição válida ou não é possível voltar mais colunas no tabuleiro.
	while ((coluna >= 3)){

        // Percorre doas as linhas da rainha atual até que suas linhas se
        // esgotem ou que posição válida seja encontrada.
        while ((linha < rainhas) && nao_valido(tabuleiro, linha, coluna))
            linha ++;

        // Caso linha válida seja encontrada, salvar posição da rainha.
        if ( linha < rainhas ){
            tabuleiro[coluna] = linha;

            // Avança para proxima rainha.
            coluna ++;

            // Caso tenha achado posição válida e seja a última rainha,
            // somar solução e voltar a analizar ultima rainha em pŕoxima linha.
            if(coluna == rainhas){
                // Solução encontrada
                solucoes ++;
                coluna --;
                linha ++;
            }
            else{
                // Retorna linha para posição incial (nova rainha será analizada).
                linha = 0;
            }
        }
        // Caso não tenha achado posição valida, backtrak.
        else{

            // Retorna à rainha anterior.
            coluna --;
            // Avanaça linha da rainha anterior.
            linha = tabuleiro[coluna] + 1;
        }

    }

    return solucoes;
}


__global__ void resolveTabuleiro(unsigned int *resposta, int rainhas){
  
    int i = blockDim.x * blockIdx.x + threadIdx.x;


    // Caso o número de rainhas seja igual a 1, retornar 1 (existe apenas 1 possível tabuleiro, e é válido).
    if (rainhas == 1){
        resposta[i] = 1;
        return;
    }
    
    // Caso o número de rainhas seja igual a 2 ou 3, retornar 0 (não existem tabuleiros válidos para eles.).
    if (rainhas == 2 || rainhas == 3){
        resposta[i] = 0;
        return;
    }

    

    
    if(i < (rainhas  * rainhas * rainhas)){
        unsigned char tabuleiro[100];
        tabuleiro[1] = i / (rainhas * rainhas);


        if(nao_valido(tabuleiro + 1, (i / rainhas) % rainhas, 1)){
            resposta[i] = 0;
            return;
        }
        tabuleiro[2] = (i / rainhas) % rainhas;
        

        if(nao_valido(tabuleiro + 1, i % rainhas, 2)){
            resposta[i] = 0;
            return;
        }
        tabuleiro[3] = i % rainhas;
        resposta[i] = resolve_tabuleiro(tabuleiro + 1, rainhas);
    }  
}

__global__ void imprimeRespostas(unsigned int *resposta, int rainhas){

    for(long int i = 0; i < rainhas; i++){
        printf("%u\n", resposta[i]);
    }  
}

int main(int argc, char *argv[])
{
    int             rainhas;
    unsigned int    resposta;
    struct timeval  tempo_ini,
                    tempo_fim,
                    tempo_total,
                    tempo_ini_kernel,
                    tempo_fim_kernel,
                    tempo_kernel,
                    tempo_ini_reduction,
                    tempo_fim_reduction,
                    tempo_reduction;
    unsigned int    *respostas;
    
    if (argc <= 1 || (rainhas = atoi(argv[1])) <= 0)
        rainhas = 8;

    tempo_ini =   timestamp();
    
    cudaMalloc((void **)&respostas, rainhas * rainhas * rainhas * sizeof(unsigned int));    

    // thrust::device_vector<unsigned int> t_respostas(rainhas);
    // respostas = thrust::raw_pointer_cast(&t_respostas[0]);

    
    // DEFINIDO DIMENSÕES DE GRID E BLOCK LINEARES PARA KERNELS resolveTabuleiro E uint2rgbKernelSHM.
    dim3 DimGridTrans(((rainhas * rainhas * rainhas)-1)/NTHREADS + 1, 1, 1);
    dim3 DimBlockTrans(NTHREADS, 1, 1);
    
    tempo_ini_kernel = timestamp();
    resolveTabuleiro<<<DimGridTrans, DimBlockTrans>>>(respostas, rainhas);
    cudaDeviceSynchronize();
    tempo_fim_kernel = timestamp();


    tempo_ini_reduction = timestamp();

    thrust::device_ptr<unsigned int> t_respostas = thrust::device_pointer_cast(respostas);
    resposta = thrust::reduce(thrust::device, t_respostas, t_respostas + (rainhas * rainhas * rainhas), 0);

    cudaDeviceSynchronize();
    tempo_fim_reduction = timestamp();

    tempo_fim = timestamp();

    timersub(&tempo_fim, &tempo_ini, &tempo_total);
    timersub(&tempo_fim_kernel, &tempo_ini_kernel, &tempo_kernel);
    timersub(&tempo_fim_reduction, &tempo_ini_reduction, &tempo_reduction);

    printf("Rainhas: %d\tResposta: %d\tTempo Total: %ld,%06ld\tTempo Kernel: %ld,%06ld\tTempo Reduction: %ld,%06ld\n", rainhas, resposta, tempo_total.tv_sec, tempo_total.tv_usec, tempo_kernel.tv_sec, tempo_kernel.tv_usec, tempo_reduction.tv_sec, tempo_reduction.tv_usec);

    return 0;
}