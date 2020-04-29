from distutils.command.build_py import build_py as _build_py
from distutils.command.clean import clean as _clean
from distutils.spawn import find_executable
from agatha import __VERSION__
from setuptools import setup, Extension, find_packages
from setuptools.command.install import install as _install
import os
import subprocess
import sys


proto_src_files = [
    "agatha/config/config.proto",
    "agatha/construct/construct_config.proto",
    "agatha/ml/abstract_generator/sentencepiece.proto",
    "agatha/topic_query/topic_query_config.proto",
    "agatha/topic_query/topic_query_result.proto",
]


# Find the Protocol Compiler.
if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
  protoc = os.environ['PROTOC']
elif os.path.exists("../src/protoc"):
  protoc = "../src/protoc"
elif os.path.exists("../src/protoc.exe"):
  protoc = "../src/protoc.exe"
elif os.path.exists("../vsprojects/Debug/protoc.exe"):
  protoc = "../vsprojects/Debug/protoc.exe"
elif os.path.exists("../vsprojects/Release/protoc.exe"):
  protoc = "../vsprojects/Release/protoc.exe"
else:
  protoc = find_executable("protoc")

def generate_proto(source):
  """Invokes the Protocol Compiler to generate a _pb2.py from the given
  .proto file.  Does nothing if the output already exists and is newer than
  the input."""

  output = source.replace(".proto", "_pb2.py")

  if (not os.path.exists(output) or
      (os.path.exists(source) and
       os.path.getmtime(source) > os.path.getmtime(output))):
    print(f"Generating {output}...")

    if not os.path.exists(source):
      sys.stderr.write("Can't find required file: %s\n" % source)
      sys.exit(-1)

    if protoc == None:
      sys.stderr.write(
          "protoc is not installed nor found in ../src.  Please compile it "
          "or install the binary package.\n")
      sys.exit(-1)

    protoc_command = [ protoc, "-I.", "--python_out=.", source ]
    if subprocess.call(protoc_command) != 0:
      sys.exit(-1)
  else:
    print("Using pre-existing version of", output)

class clean(_clean):
  def run(self):
    # Delete generated files in the code tree.
    for (dirpath, dirnames, filenames) in os.walk("."):
      for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        if filepath.endswith("_pb2.py") or filepath.endswith(".pyc") or \
          filepath.endswith(".so") or filepath.endswith(".o") or \
          filepath.endswith('google/protobuf/compiler/__init__.py'):
          os.remove(filepath)
    _clean.run(self)

class build_py(_build_py):
  def run(self):
    for proto_src in proto_src_files:
      generate_proto(proto_src)
    _build_py.run(self)

class Install(_install):
  def run(self):
    _install.do_egg_install(self)

setup(
    name='Agatha',
    version=__VERSION__,
    author="Justin Sybrandt",
    author_email="jsybran@clemson.edu",
    description=("Automatic Graph-mining And Transformer based "
                 "Hypothesis generation Approach"),
    packages=find_packages(),
    url="https://github.com/JSybrandt/agatha",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    cmdclass={ 'clean': clean, 'build_py': build_py, "install":Install },
    setup_requires=["nltk"],
    install_requires=[
      "cloudpickle",
      "dask",
      "distributed",
      "gdown",  # Needed to download pretrained models
      "h5py",
      "numpy",
      "pandas",
      "protobuf",
       "pytorch-lightning",
      "sentencepiece",  # needed to run CBAG
      "sklearn", # needed to run Agatha predictor
      "sqlitedict", # needed for CBAG
      "torch",
      "tqdm",
    ],
    dependency_links=[
      # This replaces the pytorch-lightning until their next version
      'http://github.com/PyTorchLightning/pytorch-lightning/master'
    ]
)
