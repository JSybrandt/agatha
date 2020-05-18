CC=g++
FLAGS=
FLAGS+=-std=c++17
FLAGS+=-Wall
EXTERN=../../externals
INCLUDES=-I../shared_headers
INCLUDES+=-I$(EXTERN)/argparse/include
INCLUDES+=-I$(EXTERN)/HighFive/include
INCLUDES+=-I$(EXTERN)/json/include
INCLUDES+=-I$(EXTERN)/include
LIBS=
LIBS+=-lhdf5
LIBS+=-fopenmp
LIBS+=-lstdc++fs

graph_to_ptbg: main.cc
	$(CC) $(FLAGS) -o $@ $(INCLUDES) $< $(LIBS)

clean:
	rm -rf graph_to_ptbg
