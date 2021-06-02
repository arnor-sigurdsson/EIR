from setuptools import find_packages, setup


setup(
    name="snp_pred",
    packages=find_packages(),
    version="0.1.0",
    description="Supervised modelling of human origins data.",
    author="Arnor Ingi Sigurdsson",
    license="MIT",
    entry_points={
        "console_scripts": [
            "snptrain = snptrain.snp_pred.train.main",
        ],
    },
)
