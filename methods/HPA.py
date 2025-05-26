import copy
from env import environment
import numpy as np


class HPA(object):
    """
    理想的贪婪选择,不限制单次弹性调度实例数目 from the paper:
    "ProScale: Proactive Autoscaling for Microservice With Time-Varying Workload at the Edge"
    """
    def __init__(self, env: environment.DataCenterEnvironment):
        self.env = env  # 环境的引用，实时监控env运行过程的状态变化
        self.actoin_space_dim = np.zeros((self.env.ms_nums, self.env.server_node_nums), dtype=int)
        self.max_delta = self.env.config.max_instance_update_num
        self.Um = copy.deepcopy(self.env.Node_list)     # 虚拟节点列表，用于判断资源使用情况
        self.ro_max = 0.6
        self.ro_min = 0.4
        
    def greedy_device_chosen(self, m, Um, pre_deploy_info, delta):
        """
        实现贪婪选择节点
        Input:
            m: 当前待部署的微服务
            Um: 当前微服务可以部署的设备集合
            delta: 迁移伸缩数目,(-1, 1)
        Output:
            d*: 选择的设备
            tau_min: 在该设备上的最小总延迟
        """
        tau_min = float('inf')
        d_star = None

        for d in Um:  # 遍历所有可选设备
            if d.is_resource_enough(m, delta):  # 检查设备是否有足够资源
                # 构造一个虚拟的部署信息以供计算时延
                delpoy_info = copy.deepcopy(pre_deploy_info)
                delpoy_info[m.id][d.id] += delta
                # 计算路由时延、处理和排队时延等总时延
                # TODO 这里相当于运行了一次仿真，是不对的
                tau = np.mean(self.env.cal_total_access_delay(delpoy_info)[0])

                if tau < tau_min:
                    tau_min = tau
                    d_star = d

        return d_star, tau_min

    def get_action(self, state):
        """
        不限制单次迁移伸缩实例数量
        根据预测的到达率判断哪些微服务需要弹性伸缩
        然后贪婪地选择本时隙时延最小的一个部署方案
        """
        predict_lamda_list = state[3,:,0]
        pre_deploy_info = state[0]
        action = np.zeros(self.actoin_space_dim.shape, dtype=int)
        self.Um = copy.deepcopy(self.env.Node_list)     # 每个step都要重新拷贝一下
        delta = 0

        for ms in self.env.MS_list:  # 遍历所有微服务
            lamda = predict_lamda_list[ms.id]
            image_num = np.sum(pre_deploy_info[ms.id])
            ro = lamda / (image_num*ms.mu + 1e-6)
            tau_min= 0

            while ro >= self.ro_max:
                # 统一迁移数目
                if delta >= self.max_delta:
                    break

                # 使用贪婪算法选择设备
                node, tau_min = self.greedy_device_chosen(ms, self.Um, pre_deploy_info, 1)

                # 集群资源不足
                if node is None:
                    # raise ValueError(f"The total resource is not enough to scale ms.{ms.id}!")
                    print(f"The total resource is not enough to scale ms.{ms.id}!")
                    return action

                # 更新action
                action[ms.id][node.id] += 1

                # 更新虚拟节点列表Um的资源使用情况
                self.Um[node.id].delpoy(ms, 1)

                # 更新pre_deploy_info
                pre_deploy_info[ms.id][node.id] += 1

                # 更新ro
                image_num = np.sum(pre_deploy_info[ms.id])
                ro = lamda / (image_num*ms.mu + 1e-6)

                delta += 1
            
            while ro < self.ro_min:
                # 统一迁移数目
                if delta >= self.max_delta:
                    break

                # 使用贪婪算法选择设备
                node, tau_min = self.greedy_device_chosen(ms, self.Um, pre_deploy_info, -1)

                # 集群资源不足
                if node is None:
                    # raise ValueError(f"The total resource is not enough to scale ms.{ms.id}!")
                    print(f"The total resource is not enough to scale ms.{ms.id}!")
                    return action

                # 更新action
                action[ms.id][node.id] += -1

                # 更新虚拟节点列表Um的资源使用情况
                self.Um[node.id].delpoy(ms, -1)

                # 更新pre_deploy_info
                pre_deploy_info[ms.id][node.id] += -1

                # 更新ro
                image_num = np.sum(pre_deploy_info[ms.id])
                ro = lamda / (image_num*ms.mu + 1e-6)

                delta += 1

        return action