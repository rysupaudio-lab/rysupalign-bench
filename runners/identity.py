#!/usr/bin/env python3
"""The do-nothing aligner: an empty warp means identity (double time = guide time).
rab_score.py reports this as the 'raw' baseline for every run; the runner exists as the smallest
example of the runner contract.  usage: identity.py MODEL GUIDE.wav DOUBLE.wav"""
import json
print(json.dumps({'ok': True, 'moved': False, 'warp': []}))
