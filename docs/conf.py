import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../eir"))

project = "EIR"
author = "Arnór Ingi Sigurðsson"

copyright = f"{datetime.now().year}, {author}"

html_logo = "source/_static/img/EIR_logo_white.svg"
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 3,
}
html_theme = "sphinx_rtd_theme"
html_static_path = ["source/_static"]


extensions = [
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
]
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

python_maximum_signature_line_length = 80


def setup(app):
    app.add_css_file("custom.css")
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
