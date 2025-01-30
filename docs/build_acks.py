from importlib import metadata

BASE = """
Acknowledgements
================

This project would not have been possible without the people developing,
maintaining and releasing open source projects online. Therefore, I would like
to thank the people that developed the packages this project
directly depends on, those that developed the packages those projects depend on,
and so on:

"""

OUT_PATH = "docs/acknowledgements.rst"
IN_PATH = "pyproject.toml"

packages = []
with open(IN_PATH) as f:
    toml = f.read()
    flag = False

    m1 = "[tool.poetry.dependencies]"
    m2 = "[tool.poetry.group.dev.dependencies]"
    for line in toml.splitlines():
        if line in (m1, m2):
            flag = True
        elif line.startswith("["):
            flag = False
        elif flag and line.strip():
            package = line.split("=")[0].strip()
            if package == "python":
                continue
            packages.append(package)

descriptions = {}
for package in packages:
    meta_info = metadata.metadata(package)
    description = meta_info.get("Summary", "")
    author = meta_info.get("Author", "")

    if author:
        author = f" – {author}"

    descriptions[package] = f"{author}: {description}\n"

for package_name, description in descriptions.items():
    BASE += f"- ``{package_name}`` {description}\n"

with open(OUT_PATH, "w") as f:
    f.write(BASE)
