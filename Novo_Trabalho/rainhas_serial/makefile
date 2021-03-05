# DOUGLAS AFFONSO CLEMENTINO
# GRR20175877

CC     = gcc -std=gnu99

EXECS  = rainhas_iterativo 

# arquivos-objeto
objects = rainhas_iterativo.o

all:  $(EXECS)

rainhas_iterativo: rainhas_iterativo.o  
	$(CC) rainhas_iterativo.o -o rainhas_iterativo

rainhas_iterativo.o: rainhas_iterativo.c
	$(CC) -c rainhas_iterativo.c

clean:
	-rm $(objects)
	-rm *~

purge: clean
	-rm $(EXECS)
