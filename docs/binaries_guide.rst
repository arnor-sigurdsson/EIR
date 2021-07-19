Using Binaries
==============

Using binaries is very similar
to using the Python modules
installed with e.g. ``pip``.

First, let's download one,
in this case we will use MacOS
(see `Releases`_ for more platforms and versions).

.. _Releases: https://github.com/arnor-sigurdsson/EIR/releases

.. code-block:: console

    $ wget https://github.com/arnor-sigurdsson/EIR/releases/download/0.1.12-alpha/eir-0.1.12-alpha_macos.zip
    $ unzip eir_macos.zip

We can check the version of
the binary in question
like so

.. code-block:: console

    $ ./eir --version
    0.1.12-alpha

.. note::

    It can take a couple of moments for the binaries to run.

Using the binaries is a slightly different
from using the Python modules.
As we bundle the relevant modules inside
the binary, we have to specify which ``action``
we want to perform.

So this...

.. code-block:: console

    $ eirtrain --help

... becomes this

.. code-block:: console

    $ ./eir --action train --help

We can see which actions are available
(e.g. ``train`` and ``predict`` )
by running the following command

.. code-block:: console

    $ ./eir --help