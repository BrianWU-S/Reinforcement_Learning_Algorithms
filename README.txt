1. The main codes are in "RL_algorithms_Mujoco" folder.
2. The result images are in "images" folder.
3. The guidance to install Mujoco environment is in "Install_Mujoco"  folder.
4. Different experiment settings and their demo videos are in "Version" folder.
	README.txt decribes each experiment setting.
	"demo" folder in each version shows the real time simulation video.
5. The project report is in "report" folder.
6. DQN and its variants: Double DQN, Dueling DQN, Prioritized DQN are in "DQNs" folder.
7. AC, A3C algorithms are in "AC_A3C" folder
***********************************************************************************************************
***********************************************************************************************************
Commands to run:

python run.py --env_name xxx --method xxx

Available options:
legal_env = [['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'],
             ['Hopper-v1', 'Humanoid-v1', 'HalfCheetah-v1', 'Ant-v1']]
legal_method = [['DQN', 'DoubleDQN', 'DuelingDQN', 'PrioritizedDQN'], ['DDPG', 'DDPG_explore', 'PPO']]

Note: currently the method list only supports :
	PPO
	DDPG
	DDPG_explore
         the environment list only supports:
	Hopper-v1
	Humanoid-v1
	HalfCheetah-v1
	Ant-v1

e.g.
python run.py --env_name HalfCheetah-v1 --method DDPG
python run.py --env_name HalfCheetah-v1 --method DDPG_explore
python run.py --env_name HalfCheetah-v1 --method PPO
