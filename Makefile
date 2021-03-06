# on Edision we will benchmark you against the default vendor-tuned BLAS. The compiler wrappers handle all the linking. If you wish to compare with other BLAS implementations, check the NERSC documentation.
# This makefile is intended for the GNU C compiler. To change compilers, you need to type something like: "module swap PrgEnv-pgi PrgEnv-gnu" See the NERSC documentation for available compilers.

CC = cc 
OPT = -O1 -mavx
CFLAGS = -Wall -std=gnu99 $(OPT)
LDFLAGS = -Wall -framework Accelerate
# librt is needed for clock_gettime
#LDLIBS = -lrt

targets = benchmark-auto_opt benchmark-44
objects = dgemm-auto_opt.o dgemm-44blocked.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-auto_opt : benchmark.o dgemm-auto_opt.o
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)
benchmark-44 : benchmark.o dgemm-44blocked.o
	$(CC) -o $@ $^ $(LDLIBS) $(LDFLAGS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
