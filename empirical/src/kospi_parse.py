#!/usr/bin/env python3
with open('../data/kospi/oden.txt','rb') as f:
    header = f.read(17)
    invalid_char = header[-1]
    while f.read(1) == invalid_char:
        continue
    while f:
        try:
            print(f.read())
        except Exception as e:
            print(repr(e))
