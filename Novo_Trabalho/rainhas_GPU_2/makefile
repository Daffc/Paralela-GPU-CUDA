# DOUGLAS AFFONSO CLEMENTINO
# GRR20175877

CC     = nvcc -arch=sm_50

EXECS  = rainhas_GPU_2 

# arquivos-objeto
objects = rainhas_GPU_2.o

all:  $(EXECS)

rainhas_GPU_2: rainhas_GPU_2.o  
	$(CC) rainhas_GPU_2.o -o rainhas_GPU_2

rainhas_GPU_2.o: rainhas_GPU_2.cu
	$(CC) -c rainhas_GPU_2.cu

clean:
	-rm $(objects)
	-rm *~

purge: clean
	-rm $(EXECS)
