// v0.2 modified by WZ

#define GPU_V 750

#if GPU_V == 480
  #define MP 15// number of mutiprocessors (SMs) in GTX480
  #define GRID1 MP*2// GRID sizefor rgb2uintKernelSHM and rgb2uintKernelSHM kernels
  #define NT1 768// number of threads per block in the //   rgb2uintKernelSHM and rgb2uintKernelSHM kernels//    this is perhaps the best value for GTX480
#elif GPU_V == 680
  #define MP 8// number of mutiprocessors (SMs) in GTX680
  #define GRID1 MP*2// GRID sizefor rgb2uintKernelSHM and rgb2uintKernelSHM kernels
  #define NT1 1024           // number of threads per block in the //   rgb2uintKernelSHM and rgb2uintKernelSHM kernels//    this is perhaps the best value for GTX680
#elif GPU_V == 750
  #define MP 5// number of mutiprocessors (SMs) in GTX750Ti
  #define GRID1 MP*2 // GRID sizefor rgb2uintKernelSHM and rgb2uintKernelSHM kernels
  #define NT1 1024           // number of threads per block in the //   rgb2uintKernelSHM and rgb2uintKernelSHM kernels//    this is perhaps the best value for GTX750Ti
#endif


#include "wb4.h" // use our lib instead (under construction)

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLUR_SIZE 5
#define NTHREADS 32
#define BLOCK_SIZE (NTHREADS - (2 * BLUR_SIZE))   // DIMENSÃO DE PIXELS QUE SOFRERÃO O BLUR
#define TILE_WIDTH (BLOCK_SIZE + (2 * BLUR_SIZE)) // DIMENSÃO TOTAO DE TILE (PIXEL DE BLUR + BORDA) 



//@@ INSERT CODE HERE
  //@@ INSERIR AQUI o codigo do seu kernel CUDA
__global__ void blurKernelSHM(unsigned int *saida, unsigned int  *entrada, int largura, int altura) {

  
  __shared__ unsigned int sh_mem[TILE_WIDTH][TILE_WIDTH];

  // ENDEREÇO DE "linha", "coluna" e "edereço_imagem"  COM SHIFT PARA ÁREA DE IMAGEM.
  const int coluna = blockIdx.x * BLOCK_SIZE + threadIdx.x - BLUR_SIZE;       
  const int linha = blockIdx.y * BLOCK_SIZE + threadIdx.y - BLUR_SIZE;       
  const int endereco_imagem = (linha * largura) + coluna;                    

  // ARMAZENA EM SHARED MEMORY APENAS OS PIXELS QUE EXITEM NA IMAGEM DE ENTRADA.
  if((linha >= 0) && (coluna >= 0) && (linha < altura) && (coluna < largura)) {

    sh_mem[threadIdx.y][threadIdx.x] = entrada[endereco_imagem]; 
  }
  
  __syncthreads();

  // VERIFICA SE THREAD TRATA DE UM PIXEL DA IMAGEM (DENTRO DE BLOCK_SIZE)
  if ((threadIdx.x >= BLUR_SIZE) && (threadIdx.x < (TILE_WIDTH - BLUR_SIZE)) && (threadIdx.y >= BLUR_SIZE) && (threadIdx.y < (TILE_WIDTH - BLUR_SIZE))) {
    // VERIFICA SE PIXEL EM QUESTÃO NÃO ESTRAPOLA DIMENsÕES DA IMAGEM.
    if((linha < altura) && (coluna < largura)){
      unsigned int valor_r = 0, valor_g = 0, valor_b = 0, cont = 0;
      
      int linha_atual,
          coluna_atual,
          linha_imagem,
          coluna_imagem;
  
      for(int blurLinha= -BLUR_SIZE; blurLinha <=  BLUR_SIZE; blurLinha++){
        for(int blurColun= -BLUR_SIZE; blurColun <=  BLUR_SIZE; blurColun++){
          
          // COORDENADAS DE PIXEL EM SHARED MEMORY.
          linha_atual = threadIdx.y  + blurLinha;
          coluna_atual = threadIdx.x + blurColun;

          //COORDENADA DE PIXEL EM IMAGEM REFERÊNCIA.
          linha_imagem = linha + blurLinha;
          coluna_imagem = coluna + blurColun;

          // VERIFICA SE PIXEL [LINHA_IMAGEM, COLUNA_IMAGEM], CONTIDO NA SHARED_MEMORY, ESTÁ CONTIDO NA IMAGEM.
          if((linha_imagem >= 0) && (coluna_imagem >= 0) && (linha_imagem < altura) && (coluna_imagem < largura)){
            valor_r += sh_mem[linha_atual][coluna_atual] >> 16;
            valor_g += sh_mem[linha_atual][coluna_atual] << 16 >> 24;
            valor_b += sh_mem[linha_atual][coluna_atual] << 24 >> 24;

            cont++;
          }
          
        }
      }
      saida[endereco_imagem] = ((valor_r / cont) << 16) + ((valor_g / cont) << 8) + (valor_b / cont);  
    }
  }  
}

__global__ void rgb2uintKernelSHM(unsigned int *saida, unsigned char  *entrada, int tamanho){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < tamanho){
    saida[i] = ((unsigned int)entrada[i * 3] << 16) + ((unsigned int)entrada[(i * 3) + 1] << 8) + (unsigned int)(unsigned int)entrada[(i * 3) + 2];
  }  
}

__global__ void uint2rgbKernelSHM(unsigned char *saida, unsigned int  *entrada, int tamanho){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if(i < tamanho){
    saida[i * 3] = entrada[i] >> 16;
    saida[(i * 3) + 1] = entrada[i] << 16 >> 24;
    saida[(i * 3) + 2] = entrada[i] << 24 >> 24;
  }
}



int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  unsigned char *hostInputImageData;
  unsigned char *hostOutputImageData;
  unsigned char *deviceInputImageData;
  unsigned int  *intDeviceInputImageData;
  unsigned char *deviceOutputImageData;
  unsigned int  *intDeviceOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 1);
  printf( "imagem de entrada: %s\n", inputImageFile );

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

// NOW: input and output images are RGB (3 channel)
  outputImage = wbImage_new(imageWidth, imageHeight, 3);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(unsigned char) * 3);
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(unsigned char) * 3);
  cudaMalloc((void **)&intDeviceInputImageData,
             imageWidth * imageHeight * sizeof(unsigned int));
  cudaMalloc((void **)&intDeviceOutputImageData,
             imageWidth * imageHeight * sizeof(unsigned int));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");

  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * sizeof(unsigned char) * 3, cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////



  wbTime_start(Compute, "Doing the computation on the GPU");

  int tamanho = imageWidth * imageHeight;

  dim3 DimGridTrans((tamanho-1)/NTHREADS + 1, 1, 1);
  dim3 DimBlockTrans(NTHREADS, 1, 1);

  // EFETUANDO TRANSIÇÃO DE CHAR -> INT
  rgb2uintKernelSHM<<<DimGridTrans, DimBlockTrans>>>(intDeviceInputImageData, deviceInputImageData, tamanho);
  cudaDeviceSynchronize();


  // DEFININDO GRID EM RELAÇÃO AO TAMANHO DA IMAGEM E DA QUANTIDADE DE PIXEL QUE RECEBERÃO O BLUS (BLOCK_SIZE X BLOCK_SIZE)
  dim3 DimGrid((imageWidth-1)/BLOCK_SIZE + 1, ((imageHeight)-1)/BLOCK_SIZE+1, 1);
  dim3 DimBlock(NTHREADS, NTHREADS, 1);
  blurKernelSHM<<<DimGrid,DimBlock>>>(intDeviceOutputImageData, intDeviceInputImageData,  imageWidth, imageHeight);
  cudaDeviceSynchronize();

  // EFETUANDO TRANSIÇÃO DE INT -> CHAR
  uint2rgbKernelSHM<<<DimGridTrans, DimBlockTrans>>>(deviceOutputImageData, intDeviceOutputImageData, tamanho);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");

  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * sizeof(unsigned char) * 3, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, outputImage);
  // DEBUG: if you want to see your image, 
  //   will generate file bellow in current directory
  wbExport( "blurred.ppm", outputImage );

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
