#!/usr/bin/env python

import datasets

big = datasets.PartitionedDataset({"train": 60000, "val": 20000, "test": 20000})
big.generate()
