#! /usr/bin/env bash

poetry export -f requirements.txt --without-hashes --output requirements.txt
echo "-e ." >> requirements.txt

poetry export -f requirements.txt --without-hashes --dev --output requirements-dev.txt
echo "-e ." >> requirements-dev.txt
