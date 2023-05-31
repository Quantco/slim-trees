#!/usr/bin/env bash

# first get line with version_info, then extract thing in parantheses, then replace ", " with "."
echo $(cat $1 | grep "version_info =" | sed "s/^.*(\([^()]*\)).*$/\1/" | sed "s/, /./g")
