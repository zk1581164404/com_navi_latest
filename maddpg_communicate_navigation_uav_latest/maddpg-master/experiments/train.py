import argparse
import numpy as np
import tensorflow as tf
import time
import pickle

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers #包含优化器  正则化  初始化  提供函数构造

def parse_args():
    #创建解释器
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    #超参数
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")  #25 - 100
    parser.add_argument("--num-episodes", type=int, default=30000, help="number of episodes") #60000 - 30000
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")  #这里可以换为ddpg
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./policy50/", help="directory in which training state and model should be saved") #/tmp/policy
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True) #change by zk
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args() #解析参数

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    #定义agents 网络结构，三层全连接层，有64个输出单元，激活函数是relu，最后一层没有激活函数（类似回归任务，输出action的概率值）
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.whole_reward, scenario.success_rate,scenario,scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.whole_reward,scenario.success_rate,scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    #这里是整体的输入
    # 每个智能体定义标号、训练模型、状态集、动作集、arglist
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))  #默认是ddpg  超参数default是maddpg
    for i in range(num_adversaries, env.n):   #env.n是智能体个数
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers  #trainers可调用maddpgagentrainer类里的方法


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]  #观测状态集
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)  #观测空间
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
        #这里可以使用超参数进行调节
        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)  #从/tmp/policy中加载训练结果

        episode_rewards = [0.0]  # sum of rewards for all agents
        episode_com_nai = [0.0]
        episode_success_sum = [0.0]
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset() #重置状态集
        episode_step = 0
        whole_flag = 0
        train_step = 0
        pre_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            #记录轨迹
            # print("mean_reward : " ,np.mean(episode_rewards[-arglist.save_rate:])) #mean reward
            # get action 只能通过当前智能体自身的观测选择动作 actor-critc方法
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # print ("action_n : ",action_n)  无人机进行的动作
            # environment step
            new_obs_n, rew_n, whole_rew_n,success_sum_n,done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)  #添加入记忆库
            obs_n = new_obs_n  #用新的状态信息更新原来的状态信息

            #累积一个episode的reward值，和agent的reward值
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            max_rew_step_len = -999
            sum_rew_one_game = 0
            # print("whole_rew_n : ", whole_rew_n)
            for i,rew in enumerate(whole_rew_n):
                sum_rew_one_game += rew
            max_rew_step_len = max(max_rew_step_len,sum_rew_one_game)
            # print("max_rew",episode_com_nai[-1])

            flag = 0
            for i,sucess in enumerate(success_sum_n):  #无人机的维度
                if(sucess):
                    if(flag == 0 and whole_flag == 0):
                        out = episode_step
                        if(episode_step - pre_step > 0):
                            out = episode_step - pre_step
                        # print("success step: ",out)
                        flag = 1
                        whole_flag = 1
                        pre_step = episode_step
                    episode_success_sum[-1] = sucess
                    episode_com_nai[-1] = max_rew_step_len

            #结束标志 达到训练步数 重置环境
            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                whole_flag = 0
                episode_rewards.append(0)
                episode_com_nai.append(0)
                episode_success_sum.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])
            #上面可以记录结果信息

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            #如果benchmark等于true 那么就对learned pilicy进行测试 然后存储数据
            if arglist.benchmark:  #默认false
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)  #将对象agent_info里除了最后一个元素外的所有元素保存到文件fp
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)
                # if loss != None:
                #     print ("loss : ",loss)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, com_navi_reward: {},time: {}, success:{},save_rate:{}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), np.sum(episode_com_nai[-arglist.save_rate:])/1000 ,round(time.time()-t_start, 3),np.sum(episode_success_sum[-arglist.save_rate:]),arglist.save_rate))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, com_navi_reward: {},time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], np.sum(episode_com_nai[-arglist.save_rate:])/1000 ,round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                # final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))  #记录最后1000个episode_reward的均值
                final_ep_rewards.append(np.mean(episode_com_nai[-arglist.save_rate:]))  #记录最后1000个episode_reward的均值
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))  #记录最后1000个agent_reward的均值

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
