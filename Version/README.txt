Here are main difference in these versions:
Ant_DDPG_v1:
	use_explore = False	
	batch_size = 256
	reward_factor = 10
	TAU = 0.005

Ant_DDPG_v2:
	use_explore = True
	explore_decay = 0.99995
	batch_size = 256
	reward_factor = 10
	TAU = 0.005

HalfCheetah_DDPG_v1:
	use_explore = True
	batch_size = 256
	

HalfCheetah_DDPG_v2:
	use_explore = True
	batch_size = 64


HalfCheetah_DDPG_v3:
	use_explore = False
	batch_size = 64

Hopper_DDPG_v1:
	use_explore = True
	batch size = 64
	memory_size = 1000000
	hidden_size = 256
	TAU = 0.001
	explore_decay = 0.99998
	reward_factor = 0.5

Hopper_DDPG_v2:
	without explore, batch size = 64
	memory_size = 1000000
	hidden_size = 256
	TAU = 0.001
	explore_decay = 0.99998
	reward_factor = 0.5

Hopper_DDPG_v3:
	use_explore = True 
	explore_decay = 0.99998
	reward_factor = 0.5
	smaller scale:
		batch size =64
		memory_size = 100000
		hidden_size = 128
		TAU = 0.005

Humanoid_DDPG_v1:
	 memory_size = 100000  
    	 hidden_size = 512
	TAU = 0.005
	EPISODE = 100
	batch_size = 256
	explore_decay = 0.999995
    	reward_factor = 50
	

Humanoid_DDPG_v2:
	larger scale:
	memory_size = 1000000  
    	 hidden_size = 1024
	TAU = 0.001
	EPISODE = 500
	batch_size = 64
	explore_decay = 0.999998
    	reward_factor = 2



HalfCheetah_PPO_v1:
	memory_batch = 128
	action_update_steps = 20
	critic_update_steps = 20

HalfCheetah_PPO_v2:
	memory_batch = 256
	action_update_steps = 30
	critic_update_steps = 30


HalfCheetah_PPO_v3:
	memory_batch = 128
	action_update_steps = 30
	critic_update_steps = 30

HalfCheetah_PPO_v4:
	memory_batch = 256
	action_update_steps = 20
	critic_update_steps = 20
	
Hopper_PPO_v1:
	use_explore = True
	method_index = 1
		

Hopper_PPO_v2:
	use_explore = False
	method_index = 1

Hopper_PPO_v3:
	use_explore = False
	method_index = 0

Hopper_PPO_v4:
	use_explore = False
	method_index = 0
	use advantage normalization
