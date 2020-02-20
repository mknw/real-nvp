# Debug interactively with prun and preserve.

To debug your python application via interactive debugger (e.g. to inspect tensor and 

1. on fs0, type:
   - `preserve -# 1 -s $(date -d '+1 minute' +"%H:%M") -t 900 -q proq -native "-C gpunode"` \
   - or request a specific node with `-q "all.q@nodeXXX"`
      

2. Type: `preserve -llist` (or -long-list) to find allocated node and reservation number.

3. ssh into reserved node, and type:
		- `module load cuda10.1`
		- `module load prun`
		- `conda activate maip-venv`
		- `prun -reserve <reservation id> -np 1 train_mnist.py`











# vim: set conceallevel=0
