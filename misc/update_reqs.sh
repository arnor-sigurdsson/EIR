#! /usr/bin/env bash

poetry export -f requirements.txt --without-hashes --output requirements.txt
if ! grep -F -q "-e ." requirements.txt; then
  echo "-e ." >> requirements.txt
fi

poetry export -f requirements.txt --without-hashes --dev --output requirements-dev.txt
if ! grep -F -q "-e ." requirements-dev.txt; then
  echo "-e ." >> requirements-dev.txt
fi
