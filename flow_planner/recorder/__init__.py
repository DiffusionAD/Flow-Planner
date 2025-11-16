from abc import abstractmethod
class RecorderBase:

    @abstractmethod
    def record_loss(self, loss):
        raise NotImplementedError
    
    @abstractmethod
    def record_metric(self, metrics):
        raise NotImplementedError