import argparse
from DDPG_run import run_DDPG
from PPO_run import run_PPO

legal_env = [['VideoPinball-ramNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'PongNoFrameskip-v4', 'BoxingNoFrameskip-v4'],
             ['Hopper-v1', 'Humanoid-v1', 'HalfCheetah-v1', 'Ant-v1']]

legal_method = [['DQN', 'DoubleDQN', 'DuelingDQN', 'PrioritizedDQN'], ['DDPG', 'DDPG_explore', 'PPO']]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name')
    parser.add_argument('--method')
    args = parser.parse_args()
    if args.env_name not in legal_env[0] and args.env_name not in legal_env[1]:
        print('Environment Not Supported')
        return
    if args.method not in legal_method[0] and args.method not in legal_method[1]:
        print('Algorithm Not Supported')
        return
    if args.env_name in legal_env[0] and args.method in legal_method[0]:
        print("DQN algorithms are implemented, you can find them on the other folder.",
              " They haven't been transferred to Mujoco yet ")
    elif args.env_name in legal_env[1] and args.method in legal_method[1]:
        if args.method == 'DDPG':
            if args.env_name == 'Ant-v1':
                run_DDPG(env_name=args.env_name, seed=0, memory_size=100000, hidden_size=256, LR_A=0.0001, LR_C=0.001,
                         GAMMA=0.99, TAU=0.005, batch_size=256, EPISODE=100, start_step=25000, step_per_epoch=2500,
                         render_threshold=4000, use_explore=False, explore_decay=0.999995, reward_factor=10)
            elif args.env_name == 'Humanoid-v1':
                run_DDPG(env_name=args.env_name, seed=1, memory_size=100000, hidden_size=512, LR_A=0.0001, LR_C=0.001,
                         GAMMA=0.99, TAU=0.005, batch_size=256, EPISODE=100, start_step=25000, step_per_epoch=5000,
                         render_threshold=24500, use_explore=False, explore_decay=0.999995, reward_factor=50)
            elif args.env_name == 'HalfCheetah-v1':
                run_DDPG(env_name=args.env_name, seed=0, memory_size=1000000, hidden_size=256, LR_A=0.0001, LR_C=0.001,
                         GAMMA=0.99, TAU=0.001, batch_size=64, EPISODE=200, start_step=5000, step_per_epoch=1000,
                         render_threshold=300, use_explore=False, explore_decay=0.99998, reward_factor=1)
            elif args.env_name == 'Hopper-v1':
                run_DDPG(env_name=args.env_name, seed=0, memory_size=1000000, hidden_size=512, LR_A=0.0001, LR_C=0.001,
                         GAMMA=0.99, TAU=0.001, batch_size=64, EPISODE=400, start_step=10000, step_per_epoch=1000,
                         render_threshold=2000, use_explore=False, explore_decay=0.99998, reward_factor=0.5)
        
        elif args.method == 'DDPG_explore':
            if args.env_name == 'Ant-v1':
                run_DDPG(env_name=args.env_name, seed=0, memory_size=100000, hidden_size=256, LR_A=0.0001, LR_C=0.001,
                         GAMMA=0.99, TAU=0.005, batch_size=256, EPISODE=100, start_step=25000, step_per_epoch=2500,
                         render_threshold=4000, use_explore=True, explore_decay=0.999995, reward_factor=10)
            elif args.env_name == 'Humanoid-v1':
                run_DDPG(env_name=args.env_name, seed=1, memory_size=100000, hidden_size=512, LR_A=0.0001, LR_C=0.001,
                         GAMMA=0.99, TAU=0.005, batch_size=256, EPISODE=100, start_step=25000, step_per_epoch=5000,
                         render_threshold=24500, use_explore=True, explore_decay=0.999995, reward_factor=50)
            elif args.env_name == 'HalfCheetah-v1':
                run_DDPG(env_name=args.env_name, seed=0, memory_size=1000000, hidden_size=256, LR_A=0.0001, LR_C=0.001,
                         GAMMA=0.99, TAU=0.001, batch_size=64, EPISODE=200, start_step=5000, step_per_epoch=1000,
                         render_threshold=300, use_explore=True, explore_decay=0.99998, reward_factor=1)
            elif args.env_name == 'Hopper-v1':
                run_DDPG(env_name=args.env_name, seed=0, memory_size=1000000, hidden_size=256, LR_A=0.0001, LR_C=0.001,
                         GAMMA=0.99, TAU=0.001, batch_size=64, EPISODE=400, start_step=10000, step_per_epoch=1000,
                         render_threshold=3000, use_explore=True, explore_decay=0.99998, reward_factor=0.5)
        elif args.method == 'PPO':
            if args.env_name == 'HalfCheetah-v1':
                run_PPO(env_name=args.env_name, seed=0, hidden_size=64, LR_A=0.0001, LR_C=0.001, EPISODE=50,
                        done_step=1000, step_per_epoch=5000, render_threshold=6000, use_explore=False,
                        explore_decay=0.99995, reward_factor=8, memory_batch=128, action_update_steps=20,
                        critic_update_steps=20, beta_low=1. / 1.5, beta_high=1.5, alpha=2,
                        method_index=1, smooth_factor=0.9, lam=0.5, kl_target=0.01, epsilon=0.2)
            elif args.env_name == 'Hopper-v1':
                run_PPO(env_name=args.env_name, seed=0, hidden_size=64, LR_A=0.0001, LR_C=0.0003, EPISODE=50,
                        done_step=1000, step_per_epoch=1000, render_threshold=5000, use_explore=False,
                        explore_decay=0.99995, reward_factor=8, memory_batch=128, action_update_steps=20,
                        critic_update_steps=20, beta_low=1. / 3., beta_high=3., alpha=2,
                        method_index=1, smooth_factor=0.9, lam=0.98, kl_target=0.01, epsilon=0.2)
            else:
                print("Haven't try PPO on Humanoid-v1 and Ant-v1.")
        else:
            pass
    else:
        print('Not suitable Environment & Algorithm')
        return


if __name__ == '__main__':
    run()
