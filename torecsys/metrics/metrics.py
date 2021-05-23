import abc


class Metrics(abc.ABC):
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def calculate(self) -> float:
        raise NotImplementedError


class Accuracy(Metrics):
    def __init__(self):
        super().__init__()
        self._num_correct = 0
        self._num_samples = 0

    @property
    def num_correct(self) -> int:
        return self._num_correct

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @num_correct.setter
    def num_correct(self, num_correct: int):
        if not isinstance(num_correct, int):
            raise TypeError(f'{type(num_correct).__name__} not allowed')
        self._num_correct = num_correct

    @num_samples.setter
    def num_samples(self, num_samples: int):
        if not isinstance(num_samples, int):
            raise TypeError(f'{type(num_samples).__name__} not allowed')
        self._num_samples = num_samples

    def add_num_correct(self, add_num_correct: int):
        if not isinstance(add_num_correct, int):
            raise TypeError(f'{type(add_num_correct).__name__} not allowed')
        self._num_correct += add_num_correct

    def add_num_samples(self, add_num_samples: int):
        if not isinstance(add_num_samples, int):
            raise TypeError(f'{type(add_num_samples).__name__} not allowed')
        self._num_samples += add_num_samples

    def calculate(self) -> float:
        return float(self._num_correct / self._num_samples)

    def update_for(self, add_num_correct: int, add_num_samples: int) -> float:
        self.add_num_correct(add_num_correct)
        self.add_num_samples(add_num_samples)
        return self.calculate()
