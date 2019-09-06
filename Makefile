build:
	python3 setup.py build

install:
	pip install -e .

test: build
	pytest -Wignore -v

clean:
	python3 setup.py clean
	rm -rf build
	rm -rf PyMoliere.egg-info
	# Remove pycache dirs
	find . | grep __pycache__ | xargs rmdir

