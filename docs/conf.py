from datetime import datetime

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
extensions = ["sphinx_copybutton"]
