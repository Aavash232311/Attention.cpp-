
import os
from pathlib import Path


''' Automated debugging script for CUDA kernel in attention.cpp
    File based dump verification
 '''

# Execute and compile with the debugger flag
os.system("nvcc -DDEBUG src/attention.cpp src/kernel/math.cu -o src/bin/attention")
os.system("./src/bin/attention")


