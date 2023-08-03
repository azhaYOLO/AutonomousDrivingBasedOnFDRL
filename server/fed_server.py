import copy
from agent_ddpg import DDPG


class fed_server():
    """
    联邦服务器端类
    """

    def __init__(self):
        """
        类初始化
        """
        self.COMMAND = 'SEND'
        self.server_agent = DDPG()  # 服务器端模型智能体 联邦模型
        self.clients_num = 2  # 客户端数量
        self.sent_models_num = 0  # 已发送客户端数量
        self.recved_models_num = 0  # 已接收客户端模型数量
        self.clients_agents = []  # 客户端传来的模型智能体

    def get_command(self):
        """
        获取通信指令
        :return: 通信指令
        """
        return self.COMMAND

    def set_command(self, command):
        """
        设置通信指令
        :param command: 通信指令
        """
        self.COMMAND = command

    def get_server_agent(self):
        """
        获取服务器模型智能体
        :return: 联邦模型智能体
        """
        return self.server_agent

    def set_server_agent(self, agent):
        """
        设置服务器端模型参数
        :param agent: 服务器端模型智能体
        """
        self.server_agent.Actor_eval = copy.deepcopy(agent.Actor_eval)
        self.server_agent.Actor_target = copy.deepcopy(agent.Actor_target)
        self.server_agent.Critic_eval = copy.deepcopy(agent.Critic_eval)
        self.server_agent.Critic_target = copy.deepcopy(agent.Critic_target)
        self.server_agent.atrain = copy.deepcopy(agent.atrain)
        self.server_agent.ctrain = copy.deepcopy(agent.ctrain)

    def get_clients_num(self):
        """
        获取客户端数量
        :return: clients_num 客户端数量
        """
        return self.clients_num

    def set_clients_num(self, num):
        """
        设置客户端数量
        :param num: 客户端数量
        """
        self.clients_num = num

    def get_sent_models_num(self):
        """
        获取已发送模型数量
        :return: sent_models_num 已发送模型数量
        """
        return self.sent_models_num

    def add_sent_models_num(self):
        """
        添加已发送模型数量
        """
        self.sent_models_num += 1

    def reset_sent_models_num(self):
        """
        重置已发送模型数量
        """
        self.sent_models_num = 0

    def get_recved_models_num(self):
        """
        获取客户端模型数量
        :return: 客户端模型数量
        """
        return self.recved_models_num

    def get_clients_agents(self):
        """
        获取客户端模型
        :return: 客户端模型
        """
        return self.clients_agents.copy()

    def add_client_agent(self, agent):
        """
        添加从客户端获得的模型
        :param model: 从客户端获得的模型
        """
        self.clients_agents.append(agent)
        self.recved_models_num += 1
        # if self.recved_models_num >= self.clients_num:
        #     self.set_command('SEND')

    def reset_clients_agents(self):
        """
        清空从客户端获得的模型
        """
        self.clients_agents.clear()
        self.recved_models_num = 0
