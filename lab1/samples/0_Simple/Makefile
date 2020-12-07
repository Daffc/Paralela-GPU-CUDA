DIRECTORIES = $(wildcard */)


all: $(DIRECTORIES)
	for file in $^ ; do \
		cd $${file}\
		&& make &&\
		cd ..; \
	done

clean: $(DIRECTORIES)
	for file in $^ ; do \
		cd $${file}\
		&& make clean &&\
		cd ..; \
	done