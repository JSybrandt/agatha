CC=g++
FLAGS=-std=c++11
INCLUDES=-I./externals/CLI11/include -I./externals/HighFive/include
LIBS=-lhdf5

tsvs_to_ptbg: main.cc
	$(CC) $(FLAGS) -o $@ $(INCLUDES) $< $(LIBS)

clean:
	rm tsvs_to_ptbg
