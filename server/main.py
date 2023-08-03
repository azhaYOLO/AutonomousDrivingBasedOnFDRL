import copy
import json
import ssl
import time
import threading

import torch

from server.fed_server import fed_server as fs
from agent_ddpg import DDPG
from conn.conn import *

# 服务器端模型
server_agent = DDPG()
# 聚合时间间隔 单位s
time_slot = 120

def server_federate(agents):
    """
    服务器端联邦聚合
    :param agents 客户端模型
    :return: agent 客户端智能体
    """
    num = len(agents)
    agent = DDPG()

    # 初始化存放模型参数均值的网络
    Actor_eval_avg = copy.deepcopy(agents[0].Actor_eval.state_dict())
    Actor_target_avg = copy.deepcopy(agents[0].Actor_target.state_dict())
    Critic_eval_avg = copy.deepcopy(agents[0].Critic_eval.state_dict())
    Critic_target_avg = copy.deepcopy(agents[0].Critic_target.state_dict())

    # 对模型参数进行联邦平均聚合
    for key in agent.Actor_eval.state_dict().keys():
        for i in range(1, num):
            Actor_eval_avg[key] += agents[i].Actor_eval.state_dict()[key]
        Actor_eval_avg[key] = torch.div(Actor_eval_avg[key], num)
    agent.Actor_eval.load_state_dict(Actor_eval_avg)

    for key in agent.Actor_target.state_dict().keys():
        for i in range(1, num):
            Actor_target_avg[key] += agents[i].Actor_target.state_dict()[key]
        Actor_target_avg[key] = torch.div(Actor_target_avg[key], num)
    agent.Actor_target.load_state_dict(Actor_target_avg)

    for key in agent.Critic_eval.state_dict().keys():
        for i in range(1, num):
            Critic_eval_avg[key] += agents[i].Critic_eval.state_dict()[key]
        Critic_eval_avg[key] = torch.div(Critic_eval_avg[key], num)
    agent.Critic_eval.load_state_dict(Critic_eval_avg)

    for key in agent.Critic_target.state_dict().keys():
        for i in range(1, num):
            Critic_target_avg[key] += agents[i].Critic_target.state_dict()[key]
        Critic_target_avg[key] = torch.div(Critic_target_avg[key], num)
    agent.Critic_target.load_state_dict(Critic_target_avg)

    print('*****联邦模型已聚合*****')
    return agent


class ListenThread(threading.Thread):
    """
    监听线程类
    """
    def __init__(self, func, fs):
        """
        线程初始化
        """
        threading.Thread.__init__(self)

        self.fs = fs
        self.func = func
        self.daemon = True

    def run(self):
        """
        线程启动
        """
        self.func(self.fs)


def save_history():
    """
    保存训练历史
    """
    pass


if __name__ == "__main__":
    print('启动模型模块...')
    fs = fs()  # 联邦服务器类
    print('初始化模型...')
    fs.set_server_agent(server_agent)  # 设置服务器端模型

    print('启动通信模块...')
    print('\n')
    # 启动线程监听 创建客户端数量的连接
    lt = ListenThread(server_listen_process, fs)
    lt.start()

    start_time = time.time()  # 记录时间间隔起始时间

    while True:
        current_time = time.time()  # 记录当前时间
        # 当前通信指令为 发送模型 SEND
        if fs.get_command() == 'SEND':
            # 联邦服务器端发送的模型数量达到了已连接的客户端数量
            if fs.get_sent_models_num() >= fs.get_clients_num():
                fs.set_command('PENDING')  # 设置通信指令为 挂起 PENDING
                continue
            else:
                continue

        # 当前通信指令为 挂起 PENDING
        elif fs.get_command() == 'PENDING':
            fs.reset_sent_models_num()  # 重置已发送模型数量
            # 满足时间间隔
            if current_time - start_time >= time_slot:
                wait(15)
                fs.set_command('CALL')  # 设置通信指令为 请求客户端发送模型 CALL
                print('\n')
            # 不满足时间间隔
            else:
                wait(10)
                continue

        # 当前通信指令为 请求客户端发送模型 CALL
        elif fs.get_command() == 'CALL':
            # 联邦服务器端接收到客户端模型数量达到了已连接的客户端数量
            if fs.get_recved_models_num() >= fs.clients_num:
                federated_agent = server_federate(fs.get_clients_agents())  # 联邦聚合
                fs.set_server_agent(federated_agent)  # 更新联邦服务器端模型智能体
                fs.reset_clients_agents()  # 重置接收到的客户端发送的模型智能体和数量
                fs.set_command('SEND')  # 设置通信指令为 发送模型 SEND
                start_time = time.time()  # 重置起始时间为当前时间
            else:
                continue

        # 通信指令不存在
        else:
            print('通信指令不存在! 应为SEND/PENDING/CALL')
