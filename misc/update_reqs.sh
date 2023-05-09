#! /usr/bin/env bash

poetry export -f requirements.txt --without-hashes --output requirements.txt
if ! grep -Eq "^\s*-e .\s*$" requirements.txt; then
  echo "-e ." >> requirements.txt
fi

poetry export -f requirements.txt --without-hashes --with dev --output requirements-dev.txt
if ! grep -Eq "^\s*-e .\s*$" requirements-dev.txt; then
  echo "-e ." >> requirements-dev.txt
fi

python docs/build_acks.py