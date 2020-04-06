all: install tools

install: build
	pip install -e .

test:
	pytest -Wignore -v --durations=0 agatha/

build: clean
	python3 setup.py build

tools: to_graph_db to_ptbg to_sentdb to_entdb

to_graph_db:
	cd tools/convert_graph_for_sqlite && make

to_ptbg:
	cd tools/convert_graph_for_pytorch_biggraph && make

to_sentdb:
	cd tools/convert_sentences_for_sqlite && make

to_entdb:
	cd tools/write_pytoch_biggraph_ents_to_sqlite && make

clean:
	python3 setup.py clean
	rm -rf build
	rm -rf Agatha.egg-info
	find . | grep __pycache__ | xargs rmdir
	cd tools/write_pytoch_biggraph_ents_to_sqlite && make clean
	cd tools/convert_sentences_for_sqlite && make clean
	cd tools/convert_graph_for_pytorch_biggraph && make clean
	cd tools/convert_graph_for_sqlite && make clean
