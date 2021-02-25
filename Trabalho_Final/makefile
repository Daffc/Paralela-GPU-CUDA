# DOUGLAS AFFONSO CLEMENTINO
# GRR20175877

NVCC   = nvcc -arch=sm_50
CC     = gcc

EXECS  = burrowWheeler 

# arquivos-objeto
objects = helper.o burrowWheeler.o

all:  $(EXECS)

burrowWheeler: burrowWheeler.o  helper.o
	$(NVCC) burrowWheeler.o  helper.o -o burrowWheeler

burrowWheeler.o: burrowWheeler.cu helper.h
	$(NVCC) -c burrowWheeler.cu

helper.o: helper.c helper.h 
	$(CC) -c helper.c

clean:
	-rm $(objects)
	-rm *~

purge: clean
	-rm $(EXECS)
