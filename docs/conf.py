import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../eir"))

"""Sphinx configuration."""
project = "EIR"
author = "Arnor Sigurdsson"
html_logo = "source/_static/img/EIR_logo_white.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}
copyright = f"{datetime.now().year}, {author}"
html_theme = "sphinx_rtd_theme"
extensions = [
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
]
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
