all: 
	source env.sh; \
	make -C cpp; \
	make -C cuda


clean:
	make -C cpp clean
	make -C cuda clean
