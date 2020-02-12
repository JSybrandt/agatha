CC=g++
FLAGS=
FLAGS+=-std=c++17
FLAGS+=-Wall
EXTERN=../../externals
INCLUDES=-I../shared_headers
INCLUDES+=-I$(EXTERN)/argparse/include
INCLUDES+=-I$(EXTERN)/sqlite_orm/include
INCLUDES+=-I$(EXTERN)/json/include
INCLUDES+=-I$(EXTERN)/include
LIBS=
LIBS+=-fopenmp
LIBS+=-lstdc++fs
LIBS+=-lsqlite3
LIB_PATHS=
LIB_PATHS+=-L/software/sqlite/3.21.0/lib

EX=sent_json_to_sqlite

$(EX): main.cc
	$(CC) $(FLAGS) $(LIB_PATHS) -o $@ $(INCLUDES) $< $(LIBS)

clean:
	rm $(EX)

test: $(EX)
	rm -rf example.sqlite
	./$(EX) -i test -o example.sqlite
