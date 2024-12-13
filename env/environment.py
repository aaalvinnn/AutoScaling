import numpy as np
import datastruct
import gymnasium as gym
from config import EnvConfig
from methods import FFD


class DataCenterEnvironment(gym.Env):
    """ 数据中心场景 """

    def __init__(
            self,
            env_config: EnvConfig,
            init_deploy_strategy : FFD.FFD,
            ):
        """ 初始化参数 """
        # 环境参数配置
        self.config = env_config
        self.timeslot = datastruct.TimeSlot(self.config.time_slot_start, self.config.time_slot_end)
        self.ms_number = self.config.ms_number
        self.ms_image_list = env_config.init_ms_instance_list
        self.server_node_number = self.config.node_number
        self.request_flow_posion_lambda = self.config.request_flow_posion_lambda
        self.min_request_chain_length = self.config.min_request_chain_length
        self.max_request_chain_length = self.config.max_request_chain_length
        self.max_instance_update_num  = self.config.max_instance_update_num
        self.MS_list = self._generate_ms(self.ms_number)
        self.Node_list = self._generate_node(self.server_node_number)
        self.RequestFlow_list = self._generate_request_flow()
        # 初始部署算法
        self.init_deploy_strategy = init_deploy_strategy
        # 随机种子
        self.seed = env_config.seed
        # 动作、状态空间
        self.state = np.zeros((3, self.ms_number, self.server_node_number))    # 服务器微服务实例部署情况、CPU剩余资源、内存剩余资源

        pass

    def _generate_ms(self, n_ms):
        """ 生成各种微服务 """
        return [datastruct.MSInstance(i) for i in range(n_ms)]

    def _generate_request_flow(self):
        """ 生成各种请求流 """
        num_requests = np.random.poisson(self.request_flow_posion_lambda)
        return [datastruct.Request(i, self.ms_number, self.min_request_chain_length, self.max_request_chain_length) for i in range(num_requests)]

    def _generate_node(self, n_node_server):
        """ 生成各种服务器节点 """
        return [datastruct.Node(i) for i in range(n_node_server)]

    def _process_request_chain(self, request: datastruct.Request):
        """ 处理请求链 """
        total_time = 0
        for ms_id in request.ms_list:
            # 根据 ms_id 获取目标微服务实例
            ms_instance = self.MS_list[ms_id]

            # 模拟请求到达微服务，入队任务
            ms_instance.task_in(task_size=1)  # 假设任务大小为 1
            execution_time = ms_instance.task_execute()  # 模拟任务执行
            total_time += execution_time

            # 将任务路由到下一个微服务（这里是顺序执行）
        return total_time

    def _update_deployed_state(self, action: np.ndarray):
        """ 部署微服务实例弹性伸缩 """
        total_update_instance_number = 0
        penalty = 0

        if action.shape != self.state[0].shape:
            raise ValueError(f"Action shape {action.shape} does not match state shape {self.state[0].shape}!")
        
        for ms_idx in range(self.state[0].shape[0]):
            for node_idx in range(self.state[0].shape[1]):
                delta = action[ms_idx, node_idx]
                ms = self.MS_list[ms_idx]  # 微服务
                node = self.Node_list[node_idx]  # 服务器节点

                if node.is_resource_enough(ms, delta):
                    cpu, memory = node.delpoy(ms, delta)
                    self.state[0][ms_idx, node_idx] += delta
                    self.state[1][:, node_idx] = np.full((self.state[1].shape[0],), cpu)
                    self.state[2][:, node_idx] = np.full((self.state[2].shape[0],), memory)
                    total_update_instance_number += delta
                else:
                    penalty += self.config.penalty


        return total_update_instance_number, penalty

    def _reset_datastruct(self):
        self.MS_list = self._generate_ms(self.ms_number)
        self.Node_list = self._generate_node(self.server_node_number)
        self.RequestFlow_list = self._generate_request_flow()
        self.ms_image_list = self.config.init_ms_image_list
        self.state = np.zeros((3, self.ms_number, self.server_node_number))
        for node_idx in range(self.state[0].shape[1]):
            node = self.Node_list[node_idx]
            self.state[1][:, node_idx] = node.get_cpu()
            self.state[2][:, node_idx] = node.get_memory()

    def _init_deploy(self):
        """ 第一次部署 """
        self.state, self.Node_list = self.init_deploy_strategy.deploy(self.MS_list, self.ms_image_list, self.Node_list, self.state)
        pass

    def reset(self):
        self.timeslot.reset()   # 重置时间
        self._reset_datastruct()    # 重置各数据结构
        self._init_deploy()     # 第一次部署
        return self.state

    def step(self, action):
        """
        执行一个时隙
        TODO
        """
        # 请求流到达
        request_flows = self._generate_request_flow()

        # 节点并行处理

        reward = 0
        # 预测下一时隙的请求流到达率


        # Autoscaling 部署策略
        self._update_deployed_state(action)

        # 状态转移
        self.timeslot.add_time()
        done = self.timeslot.is_end()
        
        return self.state, reward, done, {}
    
    def render(self):
        pass

    def close(self):
        pass