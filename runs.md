# Runs for real NVP exploration

- March, 4th 18:00
	- num_scales for real_NVP architecture: 3
	- residual blocks: 4
	- mid_channels: 64
	- batch_size=128
	- opt: adam w/ lr 1e-2


* March, 5th 19:57
	* num_scales == 2
	* residual blocks == 8
	* mid_channels == 64
	* batch_size == 128
	* opt: adam same same
	* sample folder: data/res_2-8-64

	- restarted at March, 6th 14:20
	- on server node071
	- from epoch 48 (same everything)
		- epoch 47 got overridden

- March, 6th 16:11
	- on `node070`
	- num_scales == 3
	- res. blocks == 8
	- mid_channels == 32
	- batch_size == 128
		
		- notable samples: epochs 69, 47, 


- March 9th, 15:40
	- net_type == densenet 
	- mid_channels == 128


- March 10th, 21:00
	- on `node078`
	- net_type: densenet
	- mid_channels: 256
	- batch_size= 64
	- n_scales: 3
	- directory: data/densetest_3-128
	- added GaussianNoise 
	- Outcome: clear digits already at 40/47th epoch. 

- March 11th, 21:00
	- on `node069`
	- net_type: densenet
	- mid_channels: 128
	- num_scales: 3
	- batch_size: 256
	- directory: data/rdense_3-128


- March 14th, 19:40
	- densenet with:
		* # not yet: dimensionality reduction at 1x1 convolutions in DenseLayer
		* doubling of x if mask == checkerboard
		* concatenation (x, -x) at before forward pass.

- March 16th, 9:50
	- densenet
		- concatenation at forward pass
		- doubling x at forward pass if mask == checkerboard
		- n_scales == 3
		- mid_channels==64
	- sample dir: wrongly labeled dense_3-16-128

- March 18th, 10:16

- March 21st, evening:
	- now on node078
	- 1 densnet on dense_test
	- 128 mid_c, 8 layers/densenet, 3 scales, batch_size 512

	- 1 densnet on dense_test4
	- 256 mid_c, 8 layers/densneet, 4 scales, batch size 128

# vim: conceallevel=0
