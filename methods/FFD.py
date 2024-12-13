from env import config
import numpy as np
import copy


class FFD:
    def __init__(self, MS_list, ms_image_list, Node_list, state) -> None:
        """
        FFD类
        :param MS_list: 场景中的微服务列表
        :param ms_image_list: 各个微服务的实例/镜像数量
        :param Node_list: 场景中的服务器节点列表
        :param state: 当前状态，一个三通道矩阵
        """
        self.config = config.EnvConfig()
        self.MS_list = copy.deepcopy(MS_list)
        self.ms_image_list = copy.deepcopy(ms_image_list)
        self.Node_list = copy.deepcopy(Node_list)
        self.state = copy.deepcopy(state)
        pass

    def _update_deploy_state(self, ms, deploy_num):
        for node in self.Node_list:
            # 当前节点能部署该微服务的最大实例数
            max_deploy_num = int(min(min(node.get_cpu()//ms.get_cpu(), node.get_memory()//ms.get_memory()), deploy_num))
            rest_deploy_num = deploy_num - max_deploy_num

            if max_deploy_num == 0:
                if self.Node_list.index(node) == len(self.Node_list) - 1:
                    print(f"服务{ms.id}部署失败，还剩余{rest_deploy_num}个实例等待部署！")
                    return
                else:
                    continue
            
            # 更新state
            self.state[0][ms.id][node.id] += max_deploy_num
            self.state[1][:, node.id] -= ms.get_cpu() * max_deploy_num
            self.state[2][:, node.id] -= ms.get_memory() * max_deploy_num

            # 更新node
            node.delpoy(ms, max_deploy_num)

            # 递归拆分部署
            if rest_deploy_num > 0:
                self._update_deploy_state(ms, rest_deploy_num)
            
            # 部署成功
            break

    def deploy(self):
        """
        FFD部署算法
        :return: 部署后的新state
        """
        # 按需求从大到小排序
        w = self.config.weight_cpu_and_memory
        ms_list = sorted(self.MS_list, key=lambda ms: ms.cpu + w * ms.memory, reverse=True)
        
        # 遍历所有微服务，进行部署
        for ms in ms_list:
            image_num = self.ms_image_list[ms.id]
            self._update_deploy_state(ms, image_num)
        
        return self.state, self.Node_list
