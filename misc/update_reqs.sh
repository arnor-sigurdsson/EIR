#! /usr/bin/env bash

pipenv lock -r > requirements.txt
pipenv lock -r --dev >> requirements.txt
