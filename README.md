# 基于联邦深度强化学习的无人驾驶决策与控制
## 使用说明。
需要至少三台机器运行该程序：
其中一台运行server文件夹下的main.py充当服务器；
其余机器运行client_agent_ddpg.py充当客户端。
## 文件说明
client文件夹：存放客户端脚本的文件夹
-client_agent_ddpg.py：客户端执行脚本，开启监听线程，用于根据通信指令进行模型的收发，并同时根据接收到的联邦模型在本地进行模型训练

conn文件夹：存放通信相关脚本的文件夹
- conn.py：通信相关的方法，包括联邦服务器的通信线程类，通用的模型编码及解码方法、提取模型网络参数方法、通信指令的发送及接收方法、模型的发送及接收方法、服务器监听方法、客户端监听方法

server文件夹：
- fed_server.py：联邦服务器端类，包含了一系列联邦服务器所具备的属性和方法
- main.py：联邦服务器执行脚本，开启监听线程，用于根据通信指令进行模型的收发，并根据收集到的模型进行联邦聚合

agent_ddpg.py：DDPG模型，包含Critic和Actor的预测和评估模型，将根据观察到的场景进行模型的训练
autostart.sh：自动选择无人车驾驶地图的脚本
gym_torcs.py：底层无人车驾驶场景设置以及无人车控制文件，用于对场景做出观察并计算奖励
snakeoil_gym.py：TORCS客户端底层控制脚本


## 备注
更详细内容请参考论文：[Autonomous Driving Decision Making and Controlling Based on Federated Deep Reinforcement Learning](https://github.com/azhaYOLO/AutonomousDrivingBasedOnFDRL/blob/master/Autonomous%20Driving%20Decision%20Making%20and%20Controlling%20Based%20on%20Federated%20Deep%20Reinforcement%20Learning.pdf)
