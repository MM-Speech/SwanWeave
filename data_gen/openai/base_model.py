from abc import abstractmethod
from typing import Any, List, Union, Optional

import numpy as np

from threading import Lock


class SingletonArgMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instances = {}

    _lock: Lock = Lock()
    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        changes to the value of the `__init__` argument do affect
        the returned instance.
        """
        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.
        with cls._lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singleton field with
            # specific arguments is already initialized, the thread won't create a new object.
            if cls.__name__+str(args)+str(kwargs) not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls.__name__+str(args)+str(kwargs)] = instance
        return cls._instances[cls.__name__+str(args)+str(kwargs)]


class Model(metaclass=SingletonArgMeta):
    """an abstrct model"""

    def __init__(self, model_name: Union[str, List[str]], ak: Union[str, List[str]], token_stat_percent: Optional[float] = None) -> None:
        self.clients = self._init_clients(model_name, ak)
        if token_stat_percent is not None:
            self._init_token_stat(token_stat_percent)

    def _init_token_stat(self, token_stat_percent):
        self.token_stat_percent = token_stat_percent
        self.token_sort = []
        self.token_stat = {'max_token': 0, 'mean_token': 0,
                           'count': 0, f'p{token_stat_percent*100}_token_num': 0}
        self.token_stat_percent = token_stat_percent

    def _init_clients(self, model_name, ak):
        if not isinstance(model_name, list):
            model_name = [model_name]
        if not isinstance(ak, list):
            ak = [ak]
        clients = []
        if len(ak) > 1 and len(model_name) == 1:
            model_name = model_name*len(ak)
        elif len(ak) == 1 and len(model_name) > 1:
            ak = ak*len(model_name)

        assert len(ak) == len(
            model_name), f"length of ak = {len(ak)} != length of model_name = {len(model_name)}"
        for model, ak in zip(model_name, ak):
            client = self._creat_client(model, ak)
            clients.append(client)
        # print(f"init {len(clients)} clients!!!")
        return clients

    def _update(self, token_num):
        self.token_sort.append(token_num)
        self.token_stat[f'p{self.token_stat_percent*100}_token_num'] = round(
            np.percentile(self.token_sort, self.token_stat_percent*100), 2)
        self.token_stat['count'] = len(self.token_sort)
        self.token_stat['mean_token'] = round(np.mean(self.token_sort), 2)
        self.token_stat['max_token'] = np.max(self.token_sort)

    @abstractmethod
    def _creat_client(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError
