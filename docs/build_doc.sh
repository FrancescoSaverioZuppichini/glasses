#!/bin/bash
sphinx-apidoc .. -o . -d 1 -H glasses -f && make clean && make html