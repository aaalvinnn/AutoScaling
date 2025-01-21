import numpy as np
from datastruct import *
import gymnasium as gym
from methods import FFD, Predicter
import math
import copy
import torch

class DataCenterEnvironment(gym.Env):
    """ 数据中心场景 """

    def __init__(
            self,
            id: int,
            env_config,
            is_train = False,
            ):
        """ 初始化参数 """
        super(DataCenterEnvironment, self).__init__()
        # 环境参数配置
        self.id = id
        self.seed = env_config.seed
        self.config = env_config
        self.is_train = is_train
        self.timeslot = TimeSlot(self.config.time_slot_start, self.config.time_slot_end)
        self.ms_nums = self.config.ms_nums
        self.ms_image_list = env_config.init_ms_image_list
        self.server_node_nums = self.config.node_nums
        self.request_flow_nums = self.config.request_flow_nums
        # 初始化微服务、节点、节点带宽图、微服务依赖数据图、用户请求
        self.MS_list = []
        self.Node_list = []
        self.RequestFlow_list = []
        self.request_lamda_list = []    # 读取data中的到达率
        self.Node2Node_bandwidth_graph = None
        self.MS2MS_data_graph = None
        # 请求流预测器
        self.predicter = Predicter.SMAPredictor(self.ms_nums, self.config.predicter_window_size)
        # 开销
        self.C = self.config.C
        self.Qt = None  # 积压量
        # 动作、状态空间
        self.state = None   # to be filled in reset()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(7, self.ms_nums, self.server_node_nums), dtype=np.float32)
        if self.config.is_las:
            self.action_space = gym.spaces.Discrete(self.server_node_nums * self.ms_nums * (self.config.max_instance_update_num * 2 + 1))
        else:
            self.action_space = gym.spaces.Tuple((
                gym.spaces.Discrete(self.server_node_nums),
                gym.spaces.Discrete(self.ms_nums),
                gym.spaces.Discrete(self.config.max_instance_update_num * 2 + 1)
            ))
        pass

    def _reset_seed(self, seed):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        if not self.is_train:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

    def _reset_datastruct(self):
        """ 重置数据结构  """
        # 微服务种类列表
        self.MS_list = self._generate_ms(self.ms_nums)
        # 服务器节点列表
        self.Node_list = self._generate_node(self.server_node_nums)
        # 用户请求流种类列表
        self.RequestFlow_list = self._generate_request_flow()
        # 初始化请求流到达率的随机数生成矩阵
        self.lamda_random_matrix = np.random.uniform(self.config.ms_min_lamda, self.config.ms_max_lamda, (self.request_flow_nums, self.timeslot.get_slot_length()))
        # 节点间带宽图
        self.Node2Node_bandwidth_graph = self._generate_node2node_bandwidth_graph(self.server_node_nums)
        # 微服务间依赖数据大小图
        self.MS2MS_data_graph = self._generate_ms2ms_data_graph(self.ms_nums)
        # 初始化微服务实例镜像数量
        self.ms_image_list = copy.deepcopy(self.config.init_ms_image_list)
        # 初始化开销积压量Q(t)
        self.Qt = 0
        # 服务器微服务实例部署情况、CPU剩余资源、内存剩余资源、预测的请求到达率
        self.state = {
            "deploy_info": np.zeros((self.ms_nums, self.server_node_nums)),
            "cpus": np.zeros(self.server_node_nums),
            "memories": np.zeros(self.server_node_nums),
            "predicted_lamda": np.zeros(self.ms_nums),
            "history_lamda": np.zeros((self.config.history_lamda_length, self.ms_nums))
        }
        for node_idx in range(self.server_node_nums):
            node = self.Node_list[node_idx]
            self.state["cpus"][node_idx] = node.cpu
            self.state["memories"][node_idx] = node.memory
        for ms_idx in range(self.ms_nums):
            self.state["predicted_lamda"][ms_idx] = self.config.estimated_max_lamda / 2

    def _init_deploy(self):
        """ 第一次部署 """
        # 这里假定已知最初的到达率，以确定一个合理的实例数量
        for i in range(len(self.ms_image_list)):
            ms = self.MS_list[i]
            self.ms_image_list[i] = math.ceil(ms.lamda / ms.mu)     # 向上取整

        init_deploy_strategy = FFD.FFD(self.MS_list, self.ms_image_list, self.Node_list, self.state)
        self.state, self.Node_list = init_deploy_strategy.deploy()
        
        return self.ms_image_list

    def get_state(self):
        """ 返回状态副本，防止返回引用被外部修改 """
        return copy.deepcopy(self.state)
    
    def get_observation(self):
        """ 返回状态副本的元组 """
        res = np.zeros(self.observation_space.shape)
        res[0] = self.state["deploy_info"]
        res[1] = self.state["cpus"]
        res[2] = self.state["memories"]
        res[3][:,0] = self.state["predicted_lamda"]
        # 用两个矩阵来储存历史的到达率
        for i in range(len(self.state["history_lamda"])):
            if i < self.server_node_nums:
                res[4][:,i] = self.state["history_lamda"][i]
            else:
                res[5][:,i-self.server_node_nums] = self.state["history_lamda"][i]
        # 储存当前step，转化为one-hot编码
        now_step_one_hot = math.floor((self.timeslot.get_now() - self.timeslot.start)* self.config.history_step_length/self.timeslot.get_slot_length() )
        res[6][now_step_one_hot//res.shape[2],now_step_one_hot%res.shape[1]] = 1

        return res

    def _generate_ms(self, n_ms) -> list[MSInstance]:
        """ 生成各种微服务 """
        return [MSInstance(i) for i in range(n_ms)]

    def _generate_request_flow(self) -> list[Request]:
        """ 生成各种请求流 """
        num_requests = self.request_flow_nums
        return [Request(i) for i in range(num_requests)]

    def _generate_node(self, n_node_server) -> list[Node]:
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

        # 若输入动作为一个整数：（node_id * ms_id * delta）
        if isinstance(action, np.int64):
            max_delta_size = self.config.max_instance_update_num*2+1
            delta = action % max_delta_size - self.config.max_instance_update_num
            ms_idx = (action // max_delta_size) % self.ms_nums
            node_idx = (action // (max_delta_size * self.ms_nums)) % self.server_node_nums
            ms = self.MS_list[ms_idx]  # 微服务
            node = self.Node_list[node_idx]  # 服务器节点

            for _ in range(abs(delta)):
                if node.is_resource_enough(ms, np.sign(delta)):
                    # # 若是一个新的服务器节点，则增加惩罚
                    # if len(node.ms_instance_list) == 0:
                    #     penalty += self.config.penalty/5

                    # 更新self.Node_list
                    cpu, memory = node.delpoy(ms, np.sign(delta))
                    # 更新self.state
                    self.state["deploy_info"][ms_idx, node_idx] += np.sign(delta)
                    self.state["cpus"][node_idx] = cpu
                    self.state["memories"][node_idx] = memory
                    # 更新self.ms_image_list
                    self.ms_image_list[ms_idx] += np.sign(delta)
                    # 更新返回值
                    total_update_instance_nums += np.sign(delta)
                else:
                    penalty += self.config.penalty

            return total_update_instance_nums, penalty

        # 若输入动作为一个长度为3的向量：(node_id, ms_id, delta)
        elif len(action) == 3:
            node_idx, ms_idx, delta = action
            delta = delta - self.config.max_instance_update_num     # 将delta从0-5映射到-2-2
            ms = self.MS_list[ms_idx]  # 微服务
            node = self.Node_list[node_idx]  # 服务器节点

            for _ in range(abs(delta)):
                if node.is_resource_enough(ms, np.sign(delta)):
                    # # 若是一个新的服务器节点，则增加惩罚
                    # if len(node.ms_instance_list) == 0:
                    #     penalty += self.config.penalty/5

                    # 更新self.Node_list
                    cpu, memory = node.delpoy(ms, np.sign(delta))
                    # 更新self.state
                    self.state["deploy_info"][ms_idx, node_idx] += np.sign(delta)
                    self.state["cpus"][node_idx] = cpu
                    self.state["memories"][node_idx] = memory
                    # 更新self.ms_image_list
                    self.ms_image_list[ms_idx] += np.sign(delta)
                    # 更新返回值
                    total_update_instance_nums += np.sign(delta)
                else:
                    penalty += self.config.penalty

            return total_update_instance_nums, penalty
        
        # 若输入动作为一个二维矩阵，则对每个微服务进行部署
        elif action.shape == self.state["deploy_info"].shape:
            for ms_idx in range(self.ms_nums):
                for node_idx in range(self.server_node_nums):
                    delta = action[ms_idx, node_idx]
                    ms = self.MS_list[ms_idx]  # 微服务
                    node = self.Node_list[node_idx]  # 服务器节点

                    if delta == 0:
                        continue
                    
                    for _ in range(abs(delta)):
                        if node.is_resource_enough(ms, np.sign(delta)):
                            # # 若是一个新的服务器节点，则增加惩罚
                            # if len(node.ms_instance_list) == 0:
                            #     penalty += self.config.penalty/5

                            # 更新self.Node_list
                            cpu, memory = node.delpoy(ms, np.sign(delta))
                            # 更新self.state
                            self.state["deploy_info"][ms_idx, node_idx] += np.sign(delta)
                            self.state["cpus"][node_idx] = cpu
                            self.state["memories"][node_idx] = memory
                            # 更新self.ms_image_list
                            self.ms_image_list[ms_idx] += np.sign(delta)
                            # 更新返回值
                            total_update_instance_nums += np.sign(delta)
                        else:
                            penalty += self.config.penalty

            return total_update_instance_nums, penalty
        
        raise ValueError(f"Action shape {action.shape} does not match state shape {self.state['deploy_info'].shape}!")

    def _update_state_lamda(self, lamda: list):
        """ 更新微服务请求的到达率状态输入 """
        for ms_idx in range(self.ms_nums):
            # 预测值
            self.state["predicted_lamda"][ms_idx] = lamda[ms_idx]
            # 历史值
            for i in range(min(self.config.history_lamda_length, self.predicter.get_buffer_len())):
                self.state["history_lamda"][i][ms_idx] = self.predicter.buffer[-i][ms_idx]

        pass


    def _get_route_prob_matrix(self, ms1_id, ms2_id, deploy_info):
        """
        计算两个微服务在当前状态下的路由概率矩阵
        param ms1: 源微服务 id
        param ms2: 目标微服务 id
        """
        ms1_nodes = np.where(deploy_info[ms1_id] > 0)[0]
        ms2_nodes = np.where(deploy_info[ms2_id] > 0)[0]
        res = np.zeros((len(ms1_nodes), len(ms2_nodes)))
        for i, node1_id in enumerate(ms1_nodes):
            for j, node2_id in enumerate(ms2_nodes):
                image_nums = deploy_info[ms2_id][node2_id]
                sum_image_nums = np.sum(deploy_info[ms2_id][ms2_nodes])
                res[i][j] = image_nums / sum_image_nums

        return res
    
    def _get_first_route_prob(self, ms1_id, deploy_info):
        """ 计算由中央虚拟总节点路由到部署了ms1_id节点的概率 """
        ms1_nodes = np.where(deploy_info[ms1_id] > 0)[0]
        res = np.zeros(len(ms1_nodes))
        for i, node1_id in enumerate(ms1_nodes):
            image_nums = deploy_info[ms1_id][node1_id]
            sum_image_nums = np.sum(deploy_info[ms1_id][ms1_nodes])
            res[i] = image_nums / sum_image_nums

        return res
    
    def _get_node2node_bw_matrix(self, ms1_id, ms2_id, deploy_info):
        """
        计算两个微服务在当前状态下的节点间传输带宽矩阵
        param ms1: 源微服务 id
        param ms2: 目标微服务 id
        """
        ms1_nodes = np.where(deploy_info[ms1_id] > 0)[0]
        ms2_nodes = np.where(deploy_info[ms2_id] > 0)[0]
        res = np.zeros((len(ms1_nodes), len(ms2_nodes)))
        for i, node1_id in enumerate(ms1_nodes):
            for j, node2_id in enumerate(ms2_nodes):
                bw = self.Node2Node_bandwidth_graph[node1_id, node2_id]
                res[i][j] = bw

        return res
        
    # 时延计算
    def _cal_execution_delay(self, request: Request, deploy_info):
        """ 计算单个请求的执行延迟（包括处理和排队延迟） """
        # TODO 请求失败处理逻辑
        t_exe_list = []
        for ms_id in request.ms_list:
            ms = self.MS_list[ms_id]
            image_num_list = deploy_info[ms_id]

            # 请求失败:
            # 该微服务请求到达率大于1（可以证明，因为是线性分流，只要总的服务强度大于等于1，则任意的节点的该微服务服务强度也大于等于1）
            if ms.lamda / (np.sum(image_num_list)*ms.mu + 1e-6) >= 1:
                return request.T_max    # 直接返回用户时延约束阈值

            t_exe = 0
            for node in self.Node_list:
                image_num = int(image_num_list[node.id])
                lamda = ms.lamda * image_num / np.sum(image_num_list)   # 根据概率路由进行分流
                if np.sum(image_num_list) == 0:
                    raise ValueError(f"Image num list {image_num_list} is not valid!")
                
                if lamda == 0:
                    continue

                ro = lamda / (image_num * ms.mu)
                p0 = 0
                for i in range(image_num):
                    p0 += 1 / math.factorial(i) * (lamda/ms.mu)**i
                p0 = 1 /(p0 + ((1 / (math.factorial(image_num)*(1-ro))) * (lamda/ms.mu)**image_num))
                lq = ((image_num*ro)**image_num / (math.factorial(image_num)*(1 - ro)**2)) * ro * p0
                wq = lq / lamda
                t_exe += wq + 1/ms.mu*(image_num/np.sum(image_num_list))    # 1/ms.mu 是平均处理时间

            # 如果计算出的时延大于该请求的最大时延约束
            if t_exe > request.T_max:
                t_exe = request.T_max

            t_exe_list.append(t_exe)

        # debug
        # print(f"Request {request.id} execution delay: {np.sum(t_exe_list)}")
        return np.sum(t_exe_list)
    
    def _cal_route_delay(self, request: Request, deploy_info):
        """ 计算单个请求的路由延迟 """
        if request.length <= 1:
            raise ValueError(f"Request {request.id} length {request.length} <= 1, can not be routed ever.")
        
        t_route_list = []
        start, end = 0, 1
        pre_route_probs = self._get_first_route_prob(request.ms_list[0], deploy_info)
        while end < request.length:
            ms1_id, ms2_id = request.ms_list[start], request.ms_list[end]
            # 计算概率路由矩阵
            route_probs = self._get_route_prob_matrix(ms1_id, ms2_id, deploy_info)
            # 计算节点间传输带宽矩阵
            node2node_bw = self._get_node2node_bw_matrix(ms1_id, ms2_id, deploy_info)
            # 计算微服务依赖数据大小
            ms2ms_data = self.MS2MS_data_graph[ms1_id][ms2_id]
            # 先沿列逐行求和，得到每个start节点的路由延迟，然后再乘上选到start节点的概率
            t_route_tmp = np.sum(route_probs * ms2ms_data / node2node_bw, axis=1)
            t_route_list.append(np.dot(t_route_tmp, pre_route_probs))
            # 更新pre_route_probs
            pre_route_probs = pre_route_probs @ route_probs
            start += 1
            end += 1
        
        # debug
        # print(f"Request {request.id} route delay: {np.sum(t_route_list)}")
        return np.sum(t_route_list)

    def cal_total_access_delay(self, deploy_info):
        """ 计算这个时隙所有请求的总访问延迟 """
        t_exe_list = []
        t_route_list = []
        for request in self.RequestFlow_list:
            # execution delay (including processing delay and queuing delay)
            t_exe_list.append(self._cal_execution_delay(request, deploy_info))
            # request routing delay
            t_route_list.append(self._cal_route_delay(request, deploy_info))

        t_exe_list = np.array(t_exe_list)
        t_route_list = np.array(t_route_list)
        t_total_list = np.add(t_exe_list, t_route_list)
        return t_total_list, t_exe_list, t_route_list

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
            if request.is_ms_needed(ms_id):
                return True
        return False
    
    def _get_congested_queue_nums(self):
        """ 计算当前时隙拥塞的队列数目 """
        nums = 0
        for ms in self.MS_list:
            image_nums = self.ms_image_list[ms.id]
            rho = ms.lamda / (ms.mu * image_nums + 1e-6)
            if rho >= 1:
                nums += 1

        return nums

    def _cal_average_service_intensity(self):
        """ 计算平均服务强度 """
        # TODO 有些问题，因为没有考虑请求流
        rho_list = []
        for ms in self.MS_list:
            if not self._is_ms_needed(ms.id):
                continue
            image_nums = self.ms_image_list[ms.id]
            rho_list.append(ms.lamda / (ms.mu * image_nums + 1e-6))

        return sum(rho_list) / len(rho_list)
    
    def _cal_lamda_list(self):
        """ 计算平均到达率 """
        lamda_list = []
        for ms in self.MS_list:
            lamda_list.append(ms.lamda)

        return lamda_list

    def _update_arrival_rate(self, request_lamda, request_flow_list: list[Request], lamda_random_matrix):
        """ 更新请求到达率 """
        # 清零
        for ms in self.MS_list:
            ms.lamda = 0

        # 线性更新
        for i, request in enumerate(request_flow_list):
            request.lamda = (request_lamda + lamda_random_matrix[i, self.timeslot.get_now()]) / 3
            for ms_idx in request.ms_list:
                self.MS_list[ms_idx].lamda += request.lamda

    def _update_Qt(self, cost):
        # self.Qt = max(self.Qt+cost-self.C, 0)
        Q_max = self.config.Q_max
        Q_min = self.config.Q_min
        self.Qt = max(min(self.Qt + cost - self.C, Q_max), Q_min)
        return self.Qt

    def _cal_cost(self, ns, nodes):
        """ 计算开销 """
        (w1, w2, w3) = self.config.cost_w_list
        static_cost = w1*np.sum(self.ms_image_list) + w3*nodes
        dynamic_cost = w2*abs(ns)

        return static_cost+dynamic_cost, static_cost, dynamic_cost
    
    def _cal_node_using_num(self):
        num = 0
        for node in self.Node_list:
            if len(node.ms_instance_list) > 0:
                num += 1

        return num
    
    def _cal_request_success_rate(self, t_total_list):
        """ 计算请求成功率 """
        success_num = 0
        for request, t in zip(self.RequestFlow_list, t_total_list):
            if request.T_max > t:
                success_num += 1

        return success_num / len(self.RequestFlow_list)
        

    def _standardize_state(self, state):
        """ 标准化状态 """
        res = np.zeros((4,) + state["deploy_info"].shape)
        res[0] = state["deploy_info"] / min(
            self.config.node_max_cpu_resource / self.config.ms_max_cpu_resource,
            self.config.node_min_memory_resource / self.config.ms_min_memory_resource
        )
        res[1] = state["cpus"] / self.config.node_max_cpu_resource
        res[2] = state["memories"] / self.config.node_max_memory_resource
        res[3] = (state["predicted_lamda"] / self.config.estimated_max_lamda)[:,np.newaxis]
        # res[4] = self.MS2MS_data_graph / self.config.ms2ms_max_data

        return res
        
    def reset(self, seed=None, options=None):
        if seed is None:
            seed = self.seed
        # 重置
        self._reset_seed(seed)
        self.timeslot.reset()   # 重置时间
        self._reset_datastruct()    # 重置各数据结构
        self.request_lamda_list = read_data(self.config.data_path)
        self.predicter.reset(self.request_lamda_list[0]/3)    # 假定知道第一个到达率，让绘图美观一些

        # 初始到达率
        self._update_arrival_rate(self.request_lamda_list[0], self.RequestFlow_list, self.lamda_random_matrix)

        # 预测一个初始值
        self.predicter.record(self._cal_lamda_list())   # 假设知道第一个值，绘图美观，同时方便训练
        self._update_state_lamda(self.predicter.predict())

        # 初次部署
        init_ms_image_num_list = self._init_deploy()

        observation = self.get_observation()

        return observation, {}

    def step(self, action):
        """
        执行一个时隙
        TODO
        """
        if self.state is None:
            raise ValueError("Environment not initialized, run reset() first.")
        
        # 预测和更新完到达率后，更新Autoscaling 部署策略
        ns, penalty = self._update_deployed_state(action)

        # 在每个时隙结束时从环境采样和统计信息

        t_total_list, t_exe_list, t_route_list = self.cal_total_access_delay(self.state["deploy_info"])
        node_using_num = self._cal_node_using_num()
        cost, s_cost, d_cost = self._cal_cost(ns, node_using_num)
        Qt = self._update_Qt(cost)
        request_success_rate = self._cal_request_success_rate(t_total_list)
        vload = self._cal_load_variance(0.5)
        ave_ro = self._cal_average_service_intensity()
        

        # 状态转移
        self.timeslot.add_time()
        y = 0
        # reward = 目标函数 + 异常动作惩罚 + 请求成功率奖励
        if self.is_train:
            # y = self.config.w_ns_and_delay*((np.log1p(cost*Qt))+cost) + np.mean(t_total_list)   # training use
            y = self.config.w_ns_and_delay*cost*Qt + np.mean(t_total_list)
        else:
            y = self.config.w_ns_and_delay*cost*Qt + np.mean(t_total_list)    # test use
        # reward = -y + penalty + 50*request_success_rate
        reward = -y + 30 + penalty
        # print(reward)

        # # debug
        # if Qt > 0:
        #     tmp = 10 * np.log1p(cost*Qt)
        #     pass

        # 在该时隙结束时，收集统计本时隙到达率
        lamda_list = self._cal_lamda_list()
        self.predicter.record(lamda_list)

        done = self.timeslot.is_end()

        if not done:
            # 预测下一时隙的到达率
            self._update_state_lamda(self.predicter.predict())

            # 更新下一时隙的到达率
            lamda = self.request_lamda_list.pop(0)
            self._update_arrival_rate(lamda, self.RequestFlow_list, self.lamda_random_matrix)

        observation = self.get_observation()

        # extra info
        info = {
            "y": y,
            "t_all": np.mean(t_total_list),
            "t_exe": np.mean(t_exe_list),
            "t_route": np.mean(t_route_list),
            "vload": vload,
            "ns": ns,
            "cost": cost,
            "static_cost": s_cost,
            "dynamic_cost": d_cost,
            "Qt": Qt,
            "penalty": penalty/(self.config.penalty+1e-6),
            "node_using_num": node_using_num,
            "image_nums": np.sum(self.ms_image_list),
            "predict_lamda": np.mean(self.predicter.predict()),
            "lamda": np.mean(lamda_list),
            "lamda_list": lamda_list,
            "ave_ro": ave_ro,
            "request_success_rate": request_success_rate,
            "r": reward
        }
        
        return observation, reward, done, False, info
    
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

def read_data(path):
    data = []
    with open(path, "r") as f:
        for line in f.readlines():
            data.append(float(line.strip()))
    return data