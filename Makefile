all: 
	source env.sh; \
	make -C cpp; \
	make -C cuda
	make -C cilk

clean:
	make -C cpp clean
	make -C cuda clean
	make -C cilk clean
