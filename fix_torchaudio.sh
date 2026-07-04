#!/bin/bash
# Fix torchaudio CUDA mismatch on AutoDL
python -c "
path = '/root/autodl-tmp/LLaMA-Factory/src/llamafactory/data/mm_plugin.py'
with open(path) as f:
    lines = f.readlines()
lines[29] = 'torchaudio = None  # patched: CUDA version mismatch\n'
with open(path, 'w') as f:
    f.writelines(lines)
print('Patched line 30.')
"
