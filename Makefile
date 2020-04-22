all: install tools

install: build
	pip install -e .

test:
	pytest -Wignore -v --durations=0 agatha/

build: clean
	python3 setup.py build

tools: to_ptbg to_lookup to_edgelist

to_ptbg:
	cd tools/convert_graph_for_pytorch_biggraph && make

to_lookup:
	cd tools/create_lookup_table && make

to_edgelist:
	cd tools/sqlite_graph_to_edgelist && make

clean:
	python3 setup.py clean
	rm -rf build
	rm -rf Agatha.egg-info
	find . | grep __pycache__ | xargs rmdir
	cd tools/convert_graph_for_pytorch_biggraph && make clean
	cd tools/create_lookup_table && make clean
	cd tools/sqlite_graph_to_edgelist && make clean
