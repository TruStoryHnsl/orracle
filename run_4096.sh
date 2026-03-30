#!/bin/bash
export ORRACLE_SOURCE=/Users/coltonorr/mnt/vault/read/nifty/gay
cd /Users/coltonorr/projects/orracle
mv venv venv.bak 2>/dev/null
caffeinate -dims /usr/bin/python3 -u run_pipeline.py --max-chunk-tokens 4096 > output/pipeline.log 2>&1
mv venv.bak venv 2>/dev/null
