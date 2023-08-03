import pickle
import ssl
import socket
import csv
import threading

from agent_ddpg import DDPG, handle_ob
from conn.conn import client_communication_process

from gym_torcs import TorcsEnv
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#####################  hyper parameters  ####################
MAX_EPISODE = 200000  # 最大回合数 20w
MAX_STEPS = 20000  # 每回合最大步数 2w
EPISODE_COUNT = 0  # 回合计数
GLOBAL_STEP = 0  # 全局步数

vision = False  # 返回图像标识 False不返回图像
done = False  # 完成标识
restor = False  # 恢复模型标识
is_train = True  # 训练标识
MODEL_PATH = '/home/azha/Torcs/Test/models'  # 模型存储路径
RESTORE_MODEL_PATH = '/home/azha/Torcs/Test/models/plain_torcs_ddpg_554_320000.pth'  # 模型加载路径
LOG_PATH = '/home/azha/Torcs/Test/fed_log.csv'
np.random.seed(1337)  # 随机数生成


class ListenThread(threading.Thread):
    """
    监听线程类
    """
    def __init__(self, func, client_agent, lock):
        """
        线程初始化
        """
        threading.Thread.__init__(self)

        self.client_agent = client_agent
        self.lock = lock
        self.func = func
        self.daemon = True

    def run(self):
        """
        线程启动
        """
        self.func(self.client_agent, self.lock)



if __name__ == "__main__":

    env = TorcsEnv(vision=vision, throttle=False)  # Generate a Torcs environment 创建一个Torcs环境
    agent = DDPG()  # 创建一个采用DDPG算法的agent

    print('启动通信模块...')
    lock = threading.Lock()
    # 启动线程监听 创建客户端数量的连接
    lt = ListenThread(client_communication_process, agent, lock)
    lt.start()




    # 加载已有模型
    if restor:
        state = torch.load(RESTORE_MODEL_PATH)

        lock.acquire()
        # EPISODE_COUNT = state['EPISODE_COUNT']
        # GLOBAL_STEP = state['GLOBAL_STEP']
        agent.Actor_eval.load_state_dict(state['state_actor_eval'])
        agent.Actor_target.load_state_dict(state['state_actor_target'])
        agent.Critic_eval.load_state_dict(state['state_critic_eval'])
        agent.Critic_target.load_state_dict(state['state_critic_target'])
        agent.atrain.load_state_dict(state['optimizer_atrain'])
        agent.ctrain.load_state_dict(state['optimizer_ctrain'])
        # agent.memory = state['memory']
        # agent.memory_counter = state['memory_counter']
        # agent.epsilon = state['epsilon']
        # agent.cost_his_a = state['cost_his_a']
        # agent.cost_his_c = state['cost_his_c']
        lock.release()

    # 训练模型
    if is_train:
        print('TORCS实验开始')
        for i in range(MAX_EPISODE):
            print("Episode : " + str(EPISODE_COUNT))

            # 重置环境
            if np.mod(i, 100) == 0:
                # Sometimes you need to relaunch TORCS because of the memory leak error
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset()

            # 处理观察内容
            ob_net = handle_ob(ob)

            total_reward = 0.  # 定义初始奖励
            for j in range(MAX_STEPS):

                action = agent.choose_action(ob_net)  # agent选择动作
                action = np.squeeze(action)

                # print("action: ", action.shape)#[3,]
                ob_, reward, done, _ = env.step(action)
                ob_net_ = handle_ob(ob_)

                # 将该回合存入经验回放池
                agent.store_transaction(ob_net, action, reward, ob_net_, done)

                # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
                if GLOBAL_STEP > 500:
                    lock.acquire()
                    agent.learn()
                    lock.release()

                # 将下一个 state_ 变为 下次循环的 state
                ob_net = ob_net_
                total_reward += reward
                GLOBAL_STEP += 1

                # 存储模型
                if GLOBAL_STEP > 50000 and GLOBAL_STEP % 5000 == 0:
                    MODEL_PATH = '/home/azha/Torcs/Test/models/fed_torcs_ddpg_' + str(
                        EPISODE_COUNT) + '_' + str(GLOBAL_STEP) + '.pth'
                    state = {
                        # 'EPISODE_COUNT': EPISODE_COUNT,
                        # 'GLOBAL_STEP': GLOBAL_STEP,
                        'state_actor_eval': agent.Actor_eval.state_dict(),
                        'state_actor_target': agent.Actor_target.state_dict(),
                        'state_critic_eval': agent.Critic_eval.state_dict(),
                        'state_critic_target': agent.Critic_target.state_dict(),
                        'optimizer_atrain': agent.atrain.state_dict(),
                        'optimizer_ctrain': agent.ctrain.state_dict(),
                        # 'memory': agent.memory,
                        # 'memory_counter': agent.memory_counter,
                        # 'epsilon': agent.epsilon,
                        # 'cost_his_a': agent.cost_his_c,
                        # 'cost_his_c': agent.cost_his_c,
                    }
                    torch.save(state, MODEL_PATH)

                if done:
                    break

            # 控制台打印该回合信息
            print(str(EPISODE_COUNT) + " -th Episode  ---------------  " + "TOTAL REWARD : " + str(total_reward))
            if len(agent.cost_his_a) > 0:
                print("loss_a: " + str(agent.cost_his_a[len(agent.cost_his_a) - 1]))
                print("loss_c: " + str(agent.cost_his_c[len(agent.cost_his_c) - 1]))
            print("epsilon: ", agent.epsilon)
            print("Total Step: " + str(GLOBAL_STEP))
            print("")

            # csv保存相关信息/home/azha/Torcs/Test
            with open(LOG_PATH, 'a+', encoding='utf8', newline='') as f:
                csv_write = csv.writer(f)
                data_row = [EPISODE_COUNT, GLOBAL_STEP, total_reward]
                csv_write.writerow(data_row)

            EPISODE_COUNT = EPISODE_COUNT + 1

    else:

        agent.epsilon = 0.0
        print("TORCS Experiment Start.")
        for i in range(MAX_EPISODE):

            # speedx = []
            # speedy = []
            # re = []

            print("Episode : " + str(EPISODE_COUNT))

            if np.mod(i, 100) == 0:
                # Sometimes you need to relaunch TORCS because of the memory leak error
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset()

            ob_net = handle_ob(ob)

            for j in range(MAX_STEPS):

                action = agent.choose_action(ob_net)
                action = np.squeeze(action)

                ob_, reward, done, _ = env.step(action)
                ob_net_ = handle_ob(ob_)

                # speedx.append(ob_net_[2] * 300.0)
                # speedy.append(ob_net_[3] * 300.0)
                # re.append(reward)

                # 将下一个 state_ 变为 下次循环的 state
                ob_net = ob_net_
                GLOBAL_STEP += 1

                if done:
                    break
            EPISODE_COUNT = EPISODE_COUNT + 1

            # MODEL_PATH = '/home/qzw/PycharmProjects/my_torcs/ckpt/test_ddpg_road.pth'
            # state = {
            #     'speedx': speedx,
            #     'speedy': speedy,
            #     're': re,
            # }
            # torch.save(state, MODEL_PATH)
            # print("模型已保存!")

    env.end()  # This is for shutting down TORCS
    print("Finish.")
