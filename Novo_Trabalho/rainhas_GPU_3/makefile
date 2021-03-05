# DOUGLAS AFFONSO CLEMENTINO
# GRR20175877

CC     = nvcc -arch=sm_50

EXECS  = rainhas_GPU_3 

# arquivos-objeto
objects = rainhas_GPU_3.o

all:  $(EXECS)

rainhas_GPU_3: rainhas_GPU_3.o  
	$(CC) rainhas_GPU_3.o -o rainhas_GPU_3

rainhas_GPU_3.o: rainhas_GPU_3.cu
	$(CC) -c rainhas_GPU_3.cu

clean:
	-rm $(objects)
	-rm *~

purge: clean
	-rm $(EXECS)
