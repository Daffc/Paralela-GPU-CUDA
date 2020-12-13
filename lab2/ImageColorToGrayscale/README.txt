1. Conteúdo do arquivo 'run_tests' foi mudado de:

    ./ImageColorToGrayScale -i Dataset/0/input0.ppm Dataset/0/input1.ppm -o out0.ppm
    ./ImageColorToGrayScale -i Dataset/1/input0.ppm Dataset/1/input1.ppm -o out1.ppm
    ./ImageColorToGrayScale -i Dataset/2/input0.ppm Dataset/2/input1.ppm -o out2.ppm
    ./ImageColorToGrayScale -i Dataset/3/input0.ppm Dataset/3/input1.ppm -o out3.ppm
    ./ImageColorToGrayScale -i Dataset/4/input0.ppm Dataset/4/input1.ppm -o out4.ppm
    ./ImageColorToGrayScale -i Dataset/5/input0.ppm Dataset/5/input1.ppm -o out5.ppm
    ./ImageColorToGrayScale -i Dataset/6/input0.ppm Dataset/6/input1.ppm -o out6.ppm
    ./ImageColorToGrayScale -i Dataset/7/input0.ppm Dataset/7/input1.ppm -o out7.ppm
    ./ImageColorToGrayScale -i Dataset/8/input0.ppm Dataset/8/input1.ppm -o out8.ppm
    ./ImageColorToGrayScale -i Dataset/9/input0.ppm Dataset/9/input1.ppm -o out9.ppm

    para:

    ./ImageColorToGrayScale -i Dataset/0/input.ppm -o Dataset/0/output.pbm
    ./ImageColorToGrayScale -i Dataset/1/input.ppm -o Dataset/1/output.pbm
    ./ImageColorToGrayScale -i Dataset/2/input.ppm -o Dataset/2/output.pbm
    ./ImageColorToGrayScale -i Dataset/3/input.ppm -o Dataset/3/output.pbm
    ./ImageColorToGrayScale -i Dataset/4/input.ppm -o Dataset/4/output.pbm
    ./ImageColorToGrayScale -i Dataset/5/input.ppm -o Dataset/5/output.pbm
    ./ImageColorToGrayScale -i Dataset/6/input.ppm -o Dataset/6/output.pbm
    ./ImageColorToGrayScale -i Dataset/7/input.ppm -o Dataset/7/output.pbm
    ./ImageColorToGrayScale -i Dataset/8/input.ppm -o Dataset/8/output.pbm
    ./ImageColorToGrayScale -i Dataset/9/input.ppm -o Dataset/9/output.pbm




2. Em 'ImageColorToGrayScale.cu', todas as ocorrencias de 'float' foram substituidas por 'unsigned char'.


3. Entrada 'Dataset/2/input.ppm' retorna erro:
    'Size of image in file Dataset/2/input.ppm does not match its header. Expecting 393216 bytes, but got 393215'

4. Para compilar em 'Orval', arquivo 'compila' foi modificado de:
    'nvcc ImageColorToGrayScale.cu -o ImageColorToGrayScale'
para:

    'nvcc -arch=sm_50 ImageColorToGrayScale.cu -o ImageColorToGrayScale'
