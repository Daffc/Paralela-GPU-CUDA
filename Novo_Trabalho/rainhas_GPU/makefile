# DOUGLAS AFFONSO CLEMENTINO
# GRR20175877

CC     = nvcc -arch=sm_50

EXECS  = rainhas_GPU 

# arquivos-objeto
objects = rainhas_GPU.o

all:  $(EXECS)

rainhas_GPU: rainhas_GPU.o  
	$(CC) rainhas_GPU.o -o rainhas_GPU

rainhas_GPU.o: rainhas_GPU.cu
	$(CC) -c rainhas_GPU.cu

clean:
	-rm $(objects)
	-rm *~

purge: clean
	-rm $(EXECS)
