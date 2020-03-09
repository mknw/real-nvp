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





# vim: conceallevel=0
