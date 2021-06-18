from datetime import datetime

"""Sphinx configuration."""
project = "EIR"
author = "Arnor Sigurdsson"
copyright = f"{datetime.now().year}, {author}"
html_theme = "sphinx_rtd_theme"
extensions = ["sphinx_copybutton"]
