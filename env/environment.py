import numpy as np
from datastruct import *
import gymnasium as gym
from config import EnvConfig
from methods import FFD
import math
import copy

class DataCenterEnvironment(gym.Env):
    """ 数据中心场景 """

    def __init__(
            self,
            env_config: EnvConfig
            ):
        """ 初始化参数 """
        # 环境参数配置
        self.seed = env_config.seed
        self.config = env_config
        self.timeslot = TimeSlot(self.config.time_slot_start, self.config.time_slot_end)
        self.ms_nums = self.config.ms_nums
        self.ms_image_list = env_config.init_ms_image_list
        self.server_node_nums = self.config.node_nums
        self.request_flow_nums = self.config.request_flow_nums
        # 初始化微服务、节点、节点带宽图、微服务依赖数据图、用户请求
        self.MS_list = []
        self.Node_list = []
        self.RequestFlow_list = []
        self.Node2Node_bandwidth_graph = None
        self.MS2MS_data_graph = None
        # TODO 动作、状态空间
        self.state = None
        pass

    def _reset_seed(self):        
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _reset_datastruct(self):
        """ 重置数据结构  """
        # 微服务种类列表
        self.MS_list = self._generate_ms(self.ms_nums)
        # 服务器节点列表
        self.Node_list = self._generate_node(self.server_node_nums)
        # 用户请求流种类列表
        self.RequestFlow_list = self._generate_request_flow()
        # 节点间带宽图
        self.Node2Node_bandwidth_graph = self._generate_node2node_bandwidth_graph(self.server_node_nums)
        # 微服务间依赖数据大小图
        self.MS2MS_data_graph = self._generate_ms2ms_data_graph(self.ms_nums)
        # 初始化微服务实例镜像数量
        self.ms_image_list = self.config.init_ms_image_list
        # 服务器微服务实例部署情况、CPU剩余资源、内存剩余资源
        self.state = np.zeros((3, self.ms_nums, self.server_node_nums))
        for node_idx in range(self.state[0].shape[1]):
            node = self.Node_list[node_idx]
            self.state[1][:, node_idx] = node.cpu
            self.state[2][:, node_idx] = node.memory

    def _init_deploy(self):
        """ 第一次部署 """
        init_deploy_strategy = FFD.FFD(self.MS_list, self.ms_image_list, self.Node_list, self.state)
        self.state, self.Node_list = init_deploy_strategy.deploy()
        pass

    def _get_state(self):
        """ 返回状态副本，防止返回引用被外部修改 """
        return copy.deepcopy(self.state)

    def _generate_ms(self, n_ms):
        """ 生成各种微服务 """
        return [MSInstance(i) for i in range(n_ms)]

    def _generate_request_flow(self):
        """ 生成各种请求流 """
        num_requests = self.request_flow_nums
        return [Request(i) for i in range(num_requests)]

    def _generate_node(self, n_node_server):
        """ 生成各种服务器节点 """
        return [Node(i) for i in range(n_node_server)]
    
    def _generate_node2node_bandwidth_graph(self, n_node_server):
        """ 生成对称的节点间带宽图，节点与自身的带宽为一个较大值 """
        # 初始化带宽图，所有元素为0
        bandwidth_graph = np.full((n_node_server, n_node_server), 1e6)

        # 生成对称的带宽值
        for i in range(n_node_server):
            for j in range(i + 1, n_node_server):
                bandwidth = random.randint(self.config.node2node_min_bandwidth, self.config.node2node_max_bandwidth)
                bandwidth_graph[i][j] = bandwidth
                bandwidth_graph[j][i] = bandwidth

        return bandwidth_graph
    
    def _generate_ms2ms_data_graph(self, ms_num):
        """ 生成所有微服务间依赖数据大小矩阵 """
        # 初始化矩阵，所有元素为0
        data_graph = np.zeros((ms_num, ms_num))

        # 生成对称的数据大小
        for i in range(ms_num):
            for j in range(i + 1, ms_num):
                data_size = random.uniform(self.config.ms2ms_min_data, self.config.ms2ms_max_data)
                data_graph[i][j] = data_size
                data_graph[j][i] = data_size

        return data_graph

    def _update_deployed_state(self, action: np.ndarray):
        """ 部署微服务实例弹性伸缩 """
        total_update_instance_nums = 0
        penalty = 0

        # 若输入动作为一个长度为3的向量：(node_id, ms_id, delta)
        if action.shape == (3,):
            node_idx, ms_idx, delta = action
            ms = self.MS_list[ms_idx]  # 微服务
            node = self.Node_list[node_idx]  # 服务器节点

            if node.is_resource_enough(ms, delta):
                # 更新self.Node_list
                cpu, memory = node.delpoy(ms, delta)
                # 更新self.state
                self.state[0][ms_idx, node_idx] += delta
                self.state[1][:, node_idx] = cpu
                self.state[2][:, node_idx] = memory
                # 更新self.ms_image_list
                self.ms_image_list[ms_idx] += delta
                # 更新返回值
                total_update_instance_nums += delta
            else:
                penalty += self.config.penalty

            return total_update_instance_nums, penalty
        
        # 若输入动作为一个二维矩阵，则对每个微服务进行部署
        elif action.shape == self.state[0].shape:
            for ms_idx in range(self.state[0].shape[0]):
                for node_idx in range(self.state[0].shape[1]):
                    delta = action[ms_idx, node_idx]
                    ms = self.MS_list[ms_idx]  # 微服务
                    node = self.Node_list[node_idx]  # 服务器节点

                    if node.is_resource_enough(ms, delta):
                        # 更新self.Node_list
                        cpu, memory = node.delpoy(ms, delta)
                        # 更新self.state
                        self.state[0][ms_idx, node_idx] += delta
                        self.state[1][:, node_idx] = cpu
                        self.state[2][:, node_idx] = memory
                        # 更新self.ms_image_list
                        self.ms_image_list[ms_idx] += delta
                        # 更新返回值
                        total_update_instance_nums += delta
                    else:
                        penalty += self.config.penalty

            return total_update_instance_nums, penalty
        
        raise ValueError(f"Action shape {action.shape} does not match state shape {self.state[0].shape}!")

    def _get_route_prob_matrix(self, ms1_id, ms2_id):
        """
        计算两个微服务在当前状态下的路由概率矩阵
        param ms1: 源微服务 id
        param ms2: 目标微服务 id
        """
        w1 = 1
        w2 = 1
        ms1_nodes = np.where(self.state[0][ms1_id] > 0)[0]
        ms2_nodes = np.where(self.state[0][ms2_id] > 0)[0]
        res = np.zeros((len(ms1_nodes), len(ms2_nodes)))
        for i, node1_id in enumerate(ms1_nodes):
            for j, node2_id in enumerate(ms2_nodes):
                bw = self.Node2Node_bandwidth_graph[node1_id, node2_id]
                image_nums = self.state[0][ms2_id][node2_id]
                sum_bw = np.sum(self.Node2Node_bandwidth_graph[node1_id][ms2_nodes])
                sum_image_nums = np.sum(self.state[0][ms2_id][ms2_nodes])
                res[i][j] = (w1*bw + w2*image_nums) / (sum_bw + sum_image_nums)

        return res
    
    def _get_first_route_prob(self, ms1_id):
        """ 计算由中央虚拟总节点路由到部署了ms1_id节点的概率 """
        ms1_nodes = np.where(self.state[0][ms1_id] > 0)[0]
        res = np.zeros(len(ms1_nodes))
        for i, node1_id in enumerate(ms1_nodes):
            image_nums = self.state[0][ms1_id][node1_id]
            sum_image_nums = np.sum(self.state[0][ms1_id][ms1_nodes])
            res[i] = image_nums / sum_image_nums

        return res
    
    def _get_node2node_bw_matrix(self, ms1_id, ms2_id):
        """
        计算两个微服务在当前状态下的节点间传输带宽矩阵
        param ms1: 源微服务 id
        param ms2: 目标微服务 id
        """
        ms1_nodes = np.where(self.state[0][ms1_id] > 0)[0]
        ms2_nodes = np.where(self.state[0][ms2_id] > 0)[0]
        res = np.zeros((len(ms1_nodes), len(ms2_nodes)))
        for i, node1_id in enumerate(ms1_nodes):
            for j, node2_id in enumerate(ms2_nodes):
                bw = self.Node2Node_bandwidth_graph[node1_id, node2_id]
                res[i][j] = bw

        return res
        
    # 时延计算
    def _cal_execution_delay(self, request: Request):
        """ 计算单个请求的执行延迟（包括处理和排队延迟） """
        # TODO 确认计算方法是正确的
        t_exe = 0
        for ms_id in request.ms_list:
            ms = self.MS_list[ms_id]
            image_num = self.ms_image_list[ms_id]
            ro = ms.lamda / (image_num * ms.mu + 1e-6)

            if ro > 1:
                # raise ValueError(f"ms {ms.id} ERROR: ro={ro} > 1")
                return 1e3  # a error data
            
            p0 = 0
            for i in range(image_num):
                p0 += 1 / math.factorial(i) * (ms.lamda/ms.mu)**i
            p0 = 1 /(p0 + ((1 / (math.factorial(image_num)*(1-ro))) * (ms.lamda/ms.mu)**image_num))
            lq = ((image_num*ro)**image_num / (math.factorial(image_num)*(1 - ro)**2)) * ro * p0
            wq = lq / ms.lamda
            t_exe += wq + 1/ms.mu

        return t_exe
    
    def _cal_route_delay(self, request: Request):
        """ 计算单个请求的路由延迟 """
        if request.length <= 1:
            raise ValueError(f"Request {request.id} length {request.length} <= 1, can not be routed ever.")
        
        t_route = 0
        start, end = 0, 1
        pre_route_probs = self._get_first_route_prob(request.ms_list[0])
        while end < request.length:
            ms1_id, ms2_id = request.ms_list[start], request.ms_list[end]
            # 计算概率路由矩阵
            route_probs = self._get_route_prob_matrix(ms1_id, ms2_id)
            # 计算节点间传输带宽矩阵
            node2node_bw = self._get_node2node_bw_matrix(ms1_id, ms2_id)
            # 计算微服务依赖数据大小
            ms2ms_data = self.MS2MS_data_graph[ms1_id][ms2_id]
            # 先沿列逐行求和，得到每个start节点的路由延迟，然后再乘上选到start节点的概率
            t_route_tmp = np.sum(route_probs * ms2ms_data / node2node_bw, axis=1)
            t_route += np.dot(t_route_tmp, pre_route_probs)
            # 更新pre_route_probs
            pre_route_probs = pre_route_probs @ route_probs
            start += 1
            end += 1
            
        return t_route

    def _cal_total_access_delay(self):
        """ 计算这个时隙所有请求的总访问延迟 """
        t_exe = 0
        t_route = 0
        for request in self.RequestFlow_list:
            # execution delay (including processing delay and queuing delay)
            t_exe += self._cal_execution_delay(request)
            # request routing delay
            t_route += self._cal_route_delay(request)

        t_total = t_exe + t_route
        return t_total, t_exe, t_route

    def _cal_load_variance(self, eta):
        """ 计算负载方差 """
        # 每个节点的资源利用率
        ucpu_list = []
        umem_list = []
        for node in self.Node_list:
            cpuo = node.cpu / node.cpu_bound[1]
            memo = node.memory / node.mem_bound[1]
            ucpu_list.append(cpuo)
            umem_list.append(memo)
        ucpu_ave = sum(ucpu_list) / len(ucpu_list)
        umem_ave = sum(umem_list) / len(umem_list)

        # 计算方差
        vcpu = sum([(ucpu - ucpu_ave)**2 for ucpu in ucpu_list]) / len(ucpu_list)
        vmem = sum([(umem - umem_ave)**2 for umem in umem_list]) / len(umem_list)

        # 计算加权负载
        vload = eta * vcpu + (1 - eta) * vmem

        return vload
    
    def _is_ms_needed(self, ms_id):
        """ 判断用户请求中是否需要该微服务 """
        for request in self.RequestFlow_list:
            if ms_id in request.ms_list:
                return True
        return False

    def _cal_average_service_intensity(self):
        """ 计算平均服务强度 """
        rho_list = []
        for ms in self.MS_list:
            if not self._is_ms_needed(ms.id):
                continue
            image_nums = self.ms_image_list[ms.id]
            rho_list.append(ms.lamda / (ms.mu * image_nums + 1e-6))

        return sum(rho_list) / len(rho_list)
    
    def _cal_average_lamda(self):
        """ 计算平均到达率 """
        lamda_list = []
        for ms in self.MS_list:
            lamda_list.append(ms.lamda)

        return sum(lamda_list) / len(lamda_list)


    def _update_arrival_rate(self, request_lamda):
        """ 更新请求到达率 """
        for ms in self.MS_list:
            ms.lamda = request_lamda * ms.init_lamda
        pass

    def reset(self):
        self._reset_seed()
        self.timeslot.reset()   # 重置时间
        self._reset_datastruct()    # 重置各数据结构
        self._init_deploy()     # 第一次部署
        return self._get_state()

    def step(self, action):
        """
        执行一个时隙
        TODO
        """
        if self.state is None:
            raise ValueError("Environment not initialized, run reset() first.")
        
        # Autoscaling 部署策略
        ns, penalty = self._update_deployed_state(action)

        # 统计信息
        delay, t_exe, t_route = self._cal_total_access_delay()
        vload = self._cal_load_variance(0.5)
        lamda = self._cal_average_lamda()
        ave_ro = self._cal_average_service_intensity()

        # 预测下一时隙的请求流到达率
        # TODO
        next_lamda = log_request_flow(1, self.timeslot.get_now())
        self._update_arrival_rate(next_lamda)

        # 状态转移
        self.timeslot.add_time()
        reward = - (self.config.w_ns_and_delay * ns + (1-self.config.w_ns_and_delay) * delay) + penalty
        done = self.timeslot.is_end()
        
        return self._get_state(), reward, done, {"delay": (delay, t_exe, t_route), "vload": vload, "ns": ns, "lamda": lamda, "ave_ro": ave_ro}
    
    def render(self):
        pass

    def close(self):
        pass


# ------------------- 请求流模拟 -------------------
def constant_request_flow(c):
    return c

def log_request_flow(c, step):
    res = c * np.log10(step+10)
    return res
