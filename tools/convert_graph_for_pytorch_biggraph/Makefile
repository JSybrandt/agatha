CC=g++
FLAGS=
FLAGS+=-std=c++17
FLAGS+=-Wall
INCLUDES=
INCLUDES+=-I./externals/argparse/include
INCLUDES+=-I./externals/HighFive/include
INCLUDES+=-I./externals/json/include
LIBS=
LIBS+=-lhdf5
LIBS+=-fopenmp
LIBS+=-lstdc++fs

tsvs_to_ptbg: main.cc
	$(CC) $(FLAGS) -o $@ $(INCLUDES) $< $(LIBS)

clean:
	rm tsvs_to_ptbg

test: tsvs_to_ptbg
	./tsvs_to_ptbg -i test/raw -c 10 -o test/ptbg
