import numpy as np
import random
from config import EnvConfig


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
                min_cpu_resource: int,
                max_cpu_resource: int,
                min_memory_resource: float,
                max_memory_resource: float
        ) -> None:
        self.id = id
        self.cpu = random.randint(min_cpu_resource, max_cpu_resource)
        self.memory = random.uniform(min_memory_resource, max_memory_resource)
        self.task_queue = 0

    def get_cpu(self) -> float:
        """ 返回该微服务实例所需要的CPU """
        return float(self.cpu)
    
    def get_memory(self) -> float:
        """ 返回该微服务实例所需要的内存 """
        return float(self.memory)
    
    def task_in(self, task_size) -> None:
        """ 请求入队 """
        self.task_queue += task_size
        pass
    
    def task_execute(self, node_compute_speed) -> float:
        """ 执行请求 """
        processing_delay = 0
        return processing_delay
        pass

    
#  ------------------------------------- Request Flow Model -------------------------------------  #
class Request(object):
    """ 请求属性 """
    def __init__(
                self,
                id,
                ms_number,
                min_chain_length,
                max_chain_length,
        ) -> None:
        self.id = id
        self.length = random.randint(min_chain_length, max_chain_length)
        self.ms_list = random.sample(range(ms_number), self.length)
        pass

# ------------------------------------ Server Node Model -------------------------------------  #
class Node(object):
    """ 节点属性 """
    def __init__(
                self,
                id,
                min_compute_speed: float,
                max_compute_speed: float,
                min_cpu_resource: int,
                max_cpu_resource: int,
                min_memory_resource: float,
                max_memory_resource: float,
        ) -> None:
        self.id = id
        self.compute_speed = random.uniform(min_compute_speed, max_compute_speed)
        self.cpu_resource = random.randint(min_cpu_resource, max_cpu_resource)
        self.memory_resource = random.uniform(min_memory_resource, max_memory_resource)
        self.cpu_resource_bound = [0, self.cpu_resource]
        self.memory_resource_bound = [0, self.memory_resource]
        self.ms_instance_list = []  # 节点部署的微服务实例列表

        pass

    def get_cpu(self):
        """ 返回当前CPU资源 """
        return float(self.cpu_resource)
    
    def get_memory(self):
        """ 返回当前内存资源 """
        return float(self.memory_resource)

    def is_resource_enough(self, ms_instance: MSInstance, ms_instance_number) -> bool:
        """ 检查剩余资源是否足够 """
        if ms_instance_number >= 0:
            additional_cpu = ms_instance.get_cpu() * ms_instance_number
            additional_memory = ms_instance.get_memory() * ms_instance_number
            return self.cpu_resource >= additional_cpu and self.memory_resource >= additional_memory
        else:
            current_instance_count = self.ms_instance_list.count(ms_instance)
            return current_instance_count >= abs(ms_instance_number)
    
    def delpoy(self, ms_instance: MSInstance, ms_instance_number) -> None:
        """
        更新剩余资源
        TODO: 考虑是否要修改成截断的做法（即节点上只有1个实例，但要减少2个，那就可以操作减少1个）
        """
        # 如果节点上只有1个实例，但要减少2个，也认为操作不成功  TODO: 考虑是否要修改
        if self.is_resource_enough(ms_instance, ms_instance_number) == False:
            raise ValueError(f"Node {self.id} has not enough resource or not enough instance to remove")
        
        if ms_instance_number >= 0:
            for _ in range(ms_instance_number):
                self.ms_instance_list.append(ms_instance)
                self.cpu_resource = max(self.cpu_resource_bound[0], self.cpu_resource - ms_instance.get_cpu())
                self.memory_resource = max(self.memory_resource_bound[0], self.memory_resource - ms_instance.get_memory())
        else:
            for _ in range(abs(ms_instance_number)):
                if ms_instance not in self.ms_instance_list:
                    raise ValueError(f"instance {ms_instance.id} is not in node {self.id}")
                self.ms_instance_list.remove(ms_instance)
                self.cpu_resource = min(self.cpu_resource + ms_instance.get_cpu(), self.cpu_resource_bound[1])
                self.memory_resource = min(self.memory_resource + ms_instance.get_memory(), self.memory_resource_bound[1])

        return self.cpu_resource, self.memory_resource
