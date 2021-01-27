// v0.2 modified by WZ

//#include <wb.h>
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

__global__ void blurKernel(unsigned char *saida, unsigned char  *entrada, int largura, int altura) {
  int linha = blockIdx.y * blockDim.y +threadIdx.y;
  int coluna = blockIdx.x * blockDim.x + threadIdx.x;

  if(linha < altura && coluna < largura){
    int valor_r = 0,
        valor_g = 0,
        valor_b = 0,
        cont = 0,
        linha_atual,
        coluna_atual;

    for(int blurLinha= -BLUR_SIZE; blurLinha <=  BLUR_SIZE; blurLinha++){
      for(int blurColun= -BLUR_SIZE; blurColun <=  BLUR_SIZE; blurColun++){
        linha_atual = linha + blurLinha;
        coluna_atual = coluna + blurColun;

        if((linha_atual >= 0) && (linha_atual < altura) && (coluna_atual >= 0) && (coluna_atual < largura)){
          valor_r += entrada[((linha_atual * largura + coluna_atual) * 3) + 0];
          valor_g += entrada[((linha_atual * largura + coluna_atual) * 3) + 1];
          valor_b += entrada[((linha_atual * largura + coluna_atual) * 3) + 2];
          cont++;
        }
      }
    }

    saida[((linha * largura + coluna) * 3) + 0] = valor_r / cont;
    saida[((linha * largura + coluna) * 3) + 1] = valor_g / cont;
    saida[((linha * largura + coluna) * 3) + 2] = valor_b / cont;
  }
  
}

//@@ INSERT CODE HERE
  //@@ INSERIR AQUI o codigo do seu kernel CUDA


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
  unsigned char *deviceOutputImageData;

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
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");

  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * sizeof(unsigned char) * 3, cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////

  dim3 DimGrid((imageWidth-1)/32 + 1, ((imageHeight)-1)/32+1, 1);
  dim3 DimBlock(32, 32, 1);

  wbTime_start(Compute, "Doing the computation on the GPU");

  blurKernel<<<DimGrid,DimBlock>>>(deviceOutputImageData, deviceInputImageData, imageWidth, imageHeight);

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
