Local Web Workbench
===================

NEExT includes an optional local web workbench for interactive graph machine
learning workflows. The workbench is not part of the core install so the base
library stays lightweight for Python users.

Installation
------------

Install the optional web dependencies:

.. code-block:: bash

   pip install "NEExT[web]"

For development from a checkout:

.. code-block:: bash

   pip install -e ".[web,dev]"

Launch
------

Run the local web server:

.. code-block:: bash

   neext web

You can also choose a project directory and port:

.. code-block:: bash

   neext web ./my-neext-project --port 8765

If the optional web dependencies are not installed, the command prints a clear
install instruction. Normal imports such as ``from NEExT import NEExT`` do not
import FastAPI, uvicorn, or frontend assets.

Project Model
-------------

The workbench stores local project state in a folder containing:

* ``manifest.json`` for artifact and job metadata.
* ``data/`` for uploaded or copied source data.
* ``artifacts/`` for generated dataset, feature, embedding, and model outputs.
* ``exports/`` for reproducible scripts and result exports.

The UI is organized around artifacts. Data artifacts feed exploration and
feature jobs; feature artifacts feed embedding jobs; embedding artifacts feed
model runs and exports.

UI Mockups
----------

Static MATLAB-inspired mockups are available in the repository under
``sandbox/ui-mockups/``. Open ``sandbox/ui-mockups/index.html`` in a browser to
compare the proposed layouts.
