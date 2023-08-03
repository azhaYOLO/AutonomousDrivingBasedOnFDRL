import copy
import pickle
import time
import threading
import socket
import ssl

import numpy as np
import torch

from agent_ddpg import DDPG

ROOT = '/home/azha/Torcs/Test/models/'  # 根目录
COMMAND_RECVED = 'COMMAND_RECVED'   # 通信指令已接收
END_OF_MSG = 'msg transport complete！'  # 消息发送完毕指令
MODEL_RECVED = 'MODEL_RECVED'  # 模型已接收指令
SEND = 'SEND'   # 联邦服务器端下发模型 通信指令
PENDING = 'PENDING'   # 挂起 通信指令
CALL = 'CALL'   # 联邦服务器端请求客户端模型 通信指令


class SocketThread(threading.Thread):
    """
    通信线程类
    """
    def __init__(self, fs, client_socket, client_addr, buffer_size=10240):
        """
        线程初始化
        :param fs: 联邦服务器类对象
        :param client_socket: 与客户端建立的套接字连接
        :param client_addr: 该线程连接的客户端IP以及port
        :param buffer_size: 套接字缓冲区大小
        """
        threading.Thread.__init__(self)

        self.fs = fs
        self.client_socket = client_socket
        self.client_addr = client_addr
        self.buffer_size = buffer_size
        self.daemon = False
        self.send_state = False
        self.recv_state = False

    def get_send_state(self):
        """
        获取该线程模型发送状态
        :return: state 发送状态
        """
        return self.send_state

    def set_send_state(self, state):
        """
        设置该线程模型发送状态
        :param state: 发送状态
        """
        self.send_state = state

    def get_recv_state(self):
        """
        获取该线程模型接收状态
        :return: state 接收状态
        """
        return self.recv_state

    def set_recv_state(self, state):
        """
        设置该线程模型接收状态
        :param state: 接收状态
        """
        self.recv_state = state

    def run(self):
        """"
        线程启动
        """
        while True:
            # 通信指令为 发送模型 SEND
            if self.fs.COMMAND == 'SEND':
                # 进入模型发送阶段，设置模型已接收状态为False
                self.set_recv_state(False)

                # 该线程还未发送模型
                if not self.get_send_state():
                    # 拼接需要传输的模型网络参数
                    server_agent = self.fs.get_server_agent()
                    server_model_dict = retrieval_model(server_agent)

                    # 发送指令 发送模型
                    send_command(self.client_socket, self.fs.COMMAND)
                    send_model(self.client_socket, server_model_dict)
                    self.fs.add_sent_models_num()

                    # 设置该线程模型发送状态为True
                    self.set_send_state(True)

                # 该模型已发送
                else:
                    print('该线程已发送模型: SEND PENDING')

                    # 发送指令 等待
                    send_command(self.client_socket, 'PENDING')
                    wait(10)

                    continue

            # 通信指令为 挂起 PENDING
            elif self.fs.COMMAND == 'PENDING':
                # 进入挂起阶段 设置模型已发送状态为False
                self.set_send_state(False)
                # 发送指令 等待
                send_command(self.client_socket, self.fs.COMMAND)
                wait(10)

                continue

            # 通信指令为 请求客户端发送模型 CALL
            elif self.fs.COMMAND == 'CALL':

                # 该线程还未接收模型
                if not self.get_recv_state():

                    # 发送指令 接受模型
                    send_command(self.client_socket, self.fs.COMMAND)
                    agent = recv_model(self.client_socket)

                    # 将接收到的模型智能体存入客户端模型智能体列表中
                    # lock.require()
                    self.fs.add_client_agent(agent)
                    # lock.release()

                    self.set_recv_state(True)

                # 该线程已发送模型
                else:
                    print('CALL PENDING')

                    # 发送指令 等待
                    send_command(self.client_socket, 'PENDING')
                    wait(10)

                    continue

            # 通信指令不存在
            else:
                print('通信指令不存在! 应为SEND/PENDING/CALL')


def encode_model(model):
    """
    编码模型 服务器端与客户端共用
    :param model: 待编码的模型
    :return: encoded_model 编码后的模型
    """
    encoded_model = pickle.dumps(model)

    return encoded_model


def decode_model(encoded_model):
    """
    解码模型 服务器端和客户端共用
    :param encoded_model: 已编码的模型
    :return: decoded_agent 解码后的模型拼接的智能体
    """
    decoded_agent = DDPG()

    model_dict = pickle.loads(encoded_model)
    decoded_agent.Actor_eval.load_state_dict(model_dict['state_actor_eval'])
    decoded_agent.Actor_target.load_state_dict(model_dict['state_actor_target'])
    decoded_agent.Critic_eval.load_state_dict(model_dict['state_critic_eval'])
    decoded_agent.Critic_target.load_state_dict(model_dict['state_critic_target'])
    decoded_agent.atrain.load_state_dict((model_dict['optimizer_atrain']))
    decoded_agent.ctrain.load_state_dict(model_dict['optimizer_ctrain'])

    return decoded_agent


def retrieval_model(agent):
    """
    提取模型网络参数
    :param agent: 模型
    :return: model_dict 模型网络参数
    """
    model_dict = {
        'state_actor_eval': agent.Actor_eval.state_dict(),
        'state_actor_target': agent.Actor_target.state_dict(),
        'state_critic_eval': agent.Critic_eval.state_dict(),
        'state_critic_target': agent.Critic_target.state_dict(),
        'optimizer_atrain': agent.atrain.state_dict(),
        'optimizer_ctrain': agent.ctrain.state_dict(),
    }

    return model_dict


def send_command(conn, command):
    """
    发送通信指令
    :param conn: 套接字
    :param command: 通信指令
    """
    print('发送通信指令:', command)
    conn.send(pickle.dumps(command))
    print('客户端已接收通信指令:', conn.recv(10240).decode('utf-8'))


def recv_command(conn):
    """
    接收通信指令
    :param conn: 套接字
    :return: command 通信指令
    """
    command = conn.recv(10240)
    print('接收通信指令:',pickle.loads(command))
    conn.send(COMMAND_RECVED.encode('utf-8'))
    return command


def send_model(conn, model):
    """
    发送模型 服务器端和客户端共用代码
    :param conn: 套接字
    :param model: 需要发送的模型
    """

    print('发送Model')
    encoded_model = encode_model(model)
    conn.send(encoded_model)  # 发送编码后的模型

    print('发送EDM')
    conn.send(pickle.dumps(END_OF_MSG))  # 发送传输完成标识
    print('对方已接收模型：', conn.recv(10240).decode('utf-8'), sep='')  # 接收模型已接收标识


def recv_model(conn):
    """
    接收模型 服务器端与客户端共同
    :param conn: 套接字
    :return: agent_info 接收到的模型网络参数
    """
    model_info = b''

    print('接收Model')
    temp = conn.recv(10240)  # 接收模型 接收缓存10240bytes
    while temp != pickle.dumps(END_OF_MSG):  # 接收到的消息不是EDM 则拼接模型，继续接收
        model_info += temp
        temp = conn.recv(10240)

    print('接收模型完成')
    agent_info = decode_model(model_info)  # 解码接收到的模型

    conn.send(MODEL_RECVED.encode('utf-8'))  # 发送模型已接收标识

    return agent_info  # 返回模型智能体


def server_listen_process(fs):
    """
    服务器端监听线程 创建客户端数量的连接
    :param fs: 联邦服务器类对象
    """
    # 创建默认上下文
    cxt = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    cxt.load_default_certs(ssl.Purpose.CLIENT_AUTH)

    # 加载证书
    try:
        cxt.load_cert_chain(certfile='./py.cer', keyfile='./py.key')
    except BaseException as e:
        cxt.load_cert_chain(certfile='/home/azha/Torcs/Test/conn/py.cer', keyfile='/home/azha/Torcs/Test/conn/py.key')

    server_address = ('172.18.8.142', 6666)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) as sock:
        # 绑定服务器IP,port
        sock.bind(server_address)
        # 等待连接
        sock.listen(5)
        # 将socket打包成SSL socket
        with cxt.wrap_socket(sock, server_side=True) as ssock:
            for i in range(fs.clients_num):
                client_socket, client_addr = ssock.accept()
                print('连接来自:', client_addr)
                st = SocketThread(fs, client_socket, client_addr)
                st.start()


def client_communication_process(agent, lock):
    """
    客户端通信线程 与服务器端建立连接并保持通信
    """
    server_address = ('172.18.8.142', 6666)  # 服务器端地址
    cxt = ssl._create_unverified_context()

    # 与服务器建立连接
    with socket.socket() as sock:
        with cxt.wrap_socket(sock, server_hostname=server_address[0]) as ssock:
            print("连接服务器!")
            ssock.connect(server_address)  # 连接服务器端

            while True:
                command = recv_command(ssock)  # 接收通信指令

                # 当前通信指令为 发送模型 SEND
                if command == pickle.dumps(SEND):
                    fed_agent = recv_model(ssock)  # 接收服务器端发送的模型参数并拼接成智能体

                    lock.acquire()
                    agent.Actor_eval.load_state_dict(fed_agent.Actor_eval.state_dict())
                    agent.Actor_target.load_state_dict(fed_agent.Actor_target.state_dict())
                    agent.Critic_eval.load_state_dict(fed_agent.Critic_eval.state_dict())
                    agent.Critic_target.load_state_dict(fed_agent.Critic_target.state_dict())
                    agent.atrain.load_state_dict(fed_agent.atrain.state_dict())
                    agent.ctrain.load_state_dict(fed_agent.ctrain.state_dict())
                    lock.release()

                # 当前通信指令为 挂起 PENDING
                elif command == pickle.dumps(PENDING):
                    continue  # 不作处理

                # 当前通信指令为 请求客户端发送模型 CALL
                elif command == pickle.dumps(CALL):
                    
                    
                    model_dict = retrieval_model(agent)  # 提取智能体模型网络参数
                    send_model(ssock, model_dict)  # 发送模型

                # 通信指令不存在
                else:
                    print('通信指令不存在! 应为SEND/PENDING/CALL')


def wait(seconds):
    """
    等待
    """
    time.sleep(seconds)


def get_model(id):
    """
    模拟服务器根据客户端id获取模型 弃用
    :param id: 模型id
    :return: agent 模型智能体
    """
    client_model_path = ROOT + 'plain_torcs_ddpg_554_{}.pth'.format(id)
    state = torch.load(client_model_path)

    agent = DDPG()
    agent.Actor_eval.load_state_dict(state['state_actor_eval'])
    agent.Actor_target.load_state_dict(state['state_actor_target'])
    agent.Critic_eval.load_state_dict(state['state_critic_eval'])
    agent.Critic_target.load_state_dict(state['state_critic_target'])
    agent.atrain.load_state_dict(state['optimizer_atrain'])
    agent.ctrain.load_state_dict(state['optimizer_ctrain'])

    return (agent)


if __name__ == '__main__':
    print('conn.py')