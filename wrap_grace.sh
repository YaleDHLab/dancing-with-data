#!/bin/bash

source $HOME/.local/bin/virtualenvwrapper.sh
source setup_grace.sh

./train-gan.py $@
