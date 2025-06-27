.. _configuration_guides:

Configuration Guides
====================

These guides are designed to kind of sit between the tutorials and the API reference.
While the tutorials provide step-by-step examples of how to use EIR for various tasks,
and the API reference provides a nice and dry documentation of all the configuration options,
these are meant to:

1. Start by having quick YAML templates for common tasks, with minimal explanation of the options.
2. Then there might be some more discussion diving a bit deeper after that, for example explaining:

    - Rationale behind some configurations and how/why they affect the model.
    - How one might want to change the configurations for different use cases.
    - We might also discuss some of the plots produced by EIR and how to interpret them.

This is not meant to be super comprehensive by e.g. discussing all available models,
we will focus n the "most common" ones.

**New to EIR?** Start with :ref:`Tutorials <tutorials_index>` for guided walkthroughes,
then return here for copy-paste configurations and optimization guidance.

**Looking for specific options?** See the full :doc:`../api/api_reference`
for comprehensive parameter documentation.

.. note::
    This section is currently as work in progress.
    Expect more guides to be added over time.

Global Configuration Template
-----------------------------

Every EIR experiment needs a global configuration.
**Start here** and copy this to ``globals.yaml``:

.. literalinclude:: global_template.yaml
   :language: yaml
   :caption: globals.yaml

Then choose your data-specific templates below.

Input Data Templates
--------------------

.. toctree::
   :maxdepth: 1

   inputs/genomics_guide
   inputs/sequence_guide