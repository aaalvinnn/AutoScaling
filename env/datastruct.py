import numpy as np
import random
from config import EnvConfig


config = EnvConfig()

#  ------------------------------------- TimeSlot -------------------------------------  #
class TimeSlot(object):
    """ 时隙属性及其操作 """

    def __init__(
                self,
                start: int,
                end: int
        ) -> None:
        self.start = start
        self.end = end
        self.slot_length = self.end - self.start

        self.now = start
        self.reset()

    def add_time(self) -> None:
        """ 增加时间 """
        self.now += 1

    def is_end(self) -> bool:
        """ 检查时隙是否结束 """
        return self.now >= self.end

    def get_slot_length(self) -> int:
        """ 返回时隙长度 """
        return self.slot_length

    def get_now(self) -> int:
        """ 返回当前时隙 """
        return self.now

    def reset(self) -> None:
        """ 重置时隙 """
        self.now = self.start

#  ------------------------------------- Microservice Model -------------------------------------  #
class MSInstance(object):
    """ 微服务实例属性 """
    def __init__(
                self,
                id: int,
                min_cpu: int = config.ms_min_cpu_resource,
                max_cpu: int = config.ms_max_cpu_resource,
                min_memory: int = config.ms_min_memory_resource,
                max_memory: int = config.ms_max_memory_resource,
                min_mu: int = config.ms_min_mu,
                max_mu: int = config.ms_max_mu
        ) -> None:
        self.id = id
        self.cpu = random.randint(min_cpu, max_cpu)
        self.memory = random.randint(min_memory, max_memory)
        self.lamda = 0   # 通过request.lamda来更新
        self.mu = random.randint(min_mu, max_mu)

    def __eq__(self, other):
        """ 用于判断两个微服务实例是否相等,调用于count() """
        if isinstance(other, MSInstance):
            if self.id == other.id and (self.cpu != other.cpu or self.memory != other.memory):
                raise ValueError(f"MSInstance {self.id} has different resource, please check seed!")
            return self.id == other.id
        
        return False
    
    
#  ------------------------------------- Request Flow Model -------------------------------------  #
class Request(object):
    """ 请求属性 """
    def __init__(
                self,
                id,
                ms_nums = config.ms_nums,
                min_chain_length = config.min_request_chain_length,
                max_chain_length = config.max_request_chain_length,
                min_request_T = config.min_request_T,
                max_request_T = config.max_request_T,
                min_lamda: int = config.ms_min_lamda,
                max_lamda: int = config.ms_max_lamda,
        ) -> None:
        self.id = id
        self.length = random.randint(min_chain_length, max_chain_length)
        self.T_max = min_request_T + (max_request_T-min_request_T)*(self.length-min_chain_length)/(max_chain_length-min_chain_length)
        self.ms_list = random.sample(range(ms_nums), self.length)
        self.min_lamda = min_lamda
        self.max_lamda = max_lamda
        self.lamda = 0

    def is_ms_needed(self, ms_id) -> bool:
        """ 检查请求链中是否包含该微服务 """
        return ms_id in self.ms_list


# ------------------------------------ Server Node Model -------------------------------------  #
class Node(object):
    """ 节点属性 """
    def __init__(
                self,
                id,
                min_cpu: int = config.node_min_cpu_resource,
                max_cpu: int = config.node_max_cpu_resource,
                min_memory: int = config.node_min_memory_resource,
                max_memory: int = config.node_max_memory_resource,
        ) -> None:
        self.id = id
        self.cpu = random.randint(min_cpu, max_cpu)
        self.memory = random.randint(min_memory, max_memory)
        self.cpu_bound = [0, self.cpu]
        self.mem_bound = [0, self.memory]
        self.ms_instance_list = []  # 节点部署的微服务实例列表
    
    def is_resource_enough(self, ms_instance: MSInstance, ms_instance_nums) -> bool:
        """ 检查剩余资源是否足够 """
        if ms_instance_nums >= 0:
            additional_cpu = ms_instance.cpu * ms_instance_nums
            additional_memory = ms_instance.memory * ms_instance_nums
            return self.cpu >= additional_cpu and self.memory >= additional_memory
        else:
            current_instance_count = self.ms_instance_list.count(ms_instance)
            return current_instance_count >= abs(ms_instance_nums)
    
    def delpoy(self, ms_instance: MSInstance, ms_instance_nums) -> None:
        """
        更新剩余资源
        TODO: 考虑是否要修改成截断的做法（即节点上只有1个实例，但要减少2个，那就可以操作减少1个）
        """
        # 如果节点上只有1个实例，但要减少2个，也认为操作不成功  TODO: 考虑是否要修改
        if self.is_resource_enough(ms_instance, ms_instance_nums) == False:
            raise ValueError(f"Node {self.id} has not enough resource or not enough instance to remove")
        
        if ms_instance_nums >= 0:
            for _ in range(ms_instance_nums):
                self.ms_instance_list.append(ms_instance)
                self.cpu = max(self.cpu_bound[0], self.cpu - ms_instance.cpu)
                self.memory = max(self.mem_bound[0], self.memory - ms_instance.memory)
        else:
            for _ in range(abs(ms_instance_nums)):
                if ms_instance not in self.ms_instance_list:
                    raise ValueError(f"instance {ms_instance.id} is not in node {self.id}")
                self.ms_instance_list.remove(ms_instance)
                self.cpu = min(self.cpu + ms_instance.cpu, self.cpu_bound[1])
                self.memory = min(self.memory + ms_instance.memory, self.mem_bound[1])

        return self.cpu, self.memory
