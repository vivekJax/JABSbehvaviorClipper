#!/usr/bin/env python3
import h5py
import sys

h5_file = sys.argv[1] if len(sys.argv) > 1 else 'jabs/features/org-3-prod.study_436.cage_4677.2025-08-11.16.22/0/features.h5'

with h5py.File(h5_file, 'r') as f:
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name} (dataset, shape: {obj.shape})")
        else:
            print(f"{name} (group)")
    
    print("HDF5 Structure:")
    f.visititems(visit)

