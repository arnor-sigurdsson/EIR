import tomlkit


def _get_module_version():
    with open("src/eir/__init__.py") as infile:
        file_contents = infile.readline()

    version = file_contents.split("__version__ = ")[-1].strip().replace('"', "")
    return version


def _get_project_meta():
    with open("pyproject.toml") as infile:
        file_contents = infile.read()

    return tomlkit.parse(file_contents)["project"]


module_version = _get_module_version()

project_meta = _get_project_meta()
toml_version = project_meta["version"]


if module_version != toml_version:
    raise ValueError(
        f"Got mismatched versions.\n"
        f"Module version: {module_version}\n"
        f"Pyproject version: {toml_version}"
    )
