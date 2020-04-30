Use Sphinx for Documentation
============================

This guide details some basics on using [Sphinx][1] to document Agatha. The goal
is to produce a human-readable website on [ReadTheDocs.org][2] in the easiest
way possible.

Writing Function Descriptions Within Code
-----------------------------------------

I've configured Sphinx to accept [Google Docstrings][3] and to parse python3
[type-hints][4]. Here's a full example:

```python3
def parse_predicate_name(predicate_name:str)->Tuple[str, str]:
  """Parses subject and object from predicate name strings.

  Predicate names are formatted strings that follow this convention:
  p:{subj}:{verb}:{obj}. This function extracts the subject and object and
  returns coded-term names in the form: m:{entity}. Will raise an exception if
  the predicate name is improperly formatted.

  Args:
    predicate_name: Predicate name in form p:{subj}:{verb}:{obj}.

  Returns:
    The subject and object formulated as coded-term names.

  """
  typ, sub, vrb, obj = predicate_name.lower().split(":")
  assert typ == PREDICATE_TYPE
  return f"{UMLS_TERM_TYPE}:{sub}", f"{UMLS_TERM_TYPE}:{obj}"
```

Lets break that down. To document a function, first you should write a good
function signature. This means that the types for each input and the return
value should have associated hints. Here, we have a string input that returns a
tuple of two strings. Note, to get type hints for many python standard objects,
such as lists, sets, and tuples, you will need to import the `typing` module.

Assuming you've got a good function signature, you can now write a
google-formatted docstring. There are certainly more specific formate options
than listed here, but at a minimum you should include:

 - Single-line summary
 - Short description
 - Argument descriptions
 - Return description

These four options are demonstrated above. Note that this string should occur as
a multi-line string (three-quotes) appearing right below the function signature.

_Note: at the time of writing, preactically none of the functions follow this
guide. If you start modifying the code, try and fill in the backlog of missing
docstrings._

Writing Help Pages
------------------

Sometimes you will have to write guides that are supplemental to the codebase
itself (for instance, this page). To do so, take a look at the `docs`
subdirectory from the root of the project. Here, I have setup `docs/help`, and
each file within this directory will automatically be included in our online
documentation. Furthermore, you can write in either [reStructuredText][5] or
[Markdown][6]. I would recommend Markdown, only because it is simpler. These
files must end in either `.rst` or `.md` based on format.

Compiling the Docs
------------------

Note that this describes how to build the documentation locally, skip ahead to
see how we use ReadTheDocs to automate this process for us.

Assuming the Agatha module has been installed, including the additional modules
in `requirements.txt`, you should be good to start compiling. Inside `docs`
there is a `Makefile` that is preconfigured to generate the API documentation as
well as any extra help files, like this one. Just type `make html` while in
`docs` to get that process started.

First, this command will run `sphinx-apidoc` on the `agatha` project in order to
extract all functions and docstrings.  This process will create a `docs/_api`
directory to store all of the intermediate API-generated documentation.  Next,
it will run `sphinx-build` to compile `html` files from all of the user-supplied
and auto-generated `.rst` and `.md` files. The result will be placed in
`/docs/build`.

The compilation process may throw a lot of warnings, especially because there
are many incorrectly formatted docstrings present in the code that predate our
adoption of sphinx and google-docstrings. This is okay as long as the
compilation process completes.

Using ReadTheDocs
-----------------

We host our documentation on [ReadTheDocs.org][2]. This service is hooked into
our repository and will automatically regenerate our documentation every time we
push a commit to master. Behind the scenes this service will build our api
documentation read in all of our `.rst` and `.md` files for us. This process
will take a while, but the result should appear online after a few minutes.

### Updating Dependencies for Read the Docs

The hardest part about ReadTheDocs is getting the remote server to properly
install all dependencies needed to install the docs and build agatha. If the
build fails remotely, but works locally (by running `make html` in the `docs`
directory) then this is almost certainly the issue. Take a look at
`.readthedocs.yaml` to see how dependencies are loaded.

Pretty much, there are different types of dependencies that are specified in
various confutation and requirements files. As always 'requirements.txt' in the
root directory contains all the modules needed to run Agatha. However, some data
dependencies are large, and we don't actually need them to build the docs. For
example, spaCy models and BERT transformers. These now live in
`data_requirements.txt` and will not be loaded by the remote document server.

Lastly, there's a bunch of dependencies that aren't necessary for Agatha
generally, but are necessary if you're going to build the docs. For instance,
the particular version of [Sphinx][1] or the couple of extensions we use. These
dependencies. We've had some issues making sure these dependencies are loaded
properly through requirements files, so we turn to [conda].

ReadTheDocs loads conda first and loads whatever we specify in our
`docs/environment.yaml` file. Take a look at [these docs][9] for how conda loads
from yaml files. Conda also allows us to specify non-python dependencies, such
as [protobuf][10] that is necessary to build agatha.

To update the conda environment used by ReadTheDocs, you need to [create a conda
environment yaml file][8]. However, as developers, we often will just want a
`requirements.txt` file to add to an already-existing environment. Here's the
best way to add a documentation dependency, or a dependency that cannot be
installed via `pip`.

 1. Run `conda env create -n <unique_name> -f docs/environment.yaml` to create a
    new environment with all of the previously nessesary requirements.
 2. If the new dependency can be installed via pip, add it to the
    `docs/requirements.txt` file. Then run `pip install -r docs/requirements.txt`.
 3. If the new dependency is only available through conda, simply run `conda
    install <new_dep>`
 4. Run `conda env export > docs/environment.yaml` to update the stored conda
    environment.
 5. Add all changed files to a commit and push it online. Make sure the
    ReadTheDocs build succeeds.

[1]:https://www.sphinx-doc.org/en/master/index.html
[2]:https://readthedocs.org/
[3]:https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
[4]:https://docs.python.org/3/library/typing.html
[5]:https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
[6]:https://daringfireball.net/projects/markdown/syntax
[7]:https://www.anaconda.com/products/individual
[8]:https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment
[9]:https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
[10]:https://developers.google.com/protocol-buffers
