#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void incrementaRespostas(unsigned int *resposta, int tamanho){

    for(long int i = 0; i < tamanho; i++){
        resposta[i] = i*2;
        printf("%u\n", resposta[i]);
    }
}


int main(){

    int tamanho = 8;
    unsigned int    *respostas;

    thrust::device_vector<unsigned int> array(tamanho);

    respostas = thrust::raw_pointer_cast(&array[0]);

    incrementaRespostas<<<1, 1>>>(respostas, tamanho);
    gpuErrchk( cudaPeekAtLastError() );


    thrust::copy(array.begin(), array.end(), std::ostream_iterator<float>(std::cout, " "));

    // // int data[6] = {1, 0, 3, 2, 1, 3};
    // // int result = thrust::reduce(thrust::device, t_respostas, t_respostas + 6, 0);
    unsigned int result = thrust::reduce(thrust::device, array.begin(), array.end(), 0);

    printf("\n%d\n", result);
    return 0;
}