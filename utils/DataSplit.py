from abc import ABC
from dataclasses import dataclass, field

from numpy.random import RandomState
from sklearn.model_selection import train_test_split


@dataclass
class DataSplit(ABC):
    """Реализует различные варианты разбиения набора даннных"""
    
    def split(self, data):
        raise NotImplementedError("Subclasses of DataSplit should implement method split.")


@dataclass
class ObjectDataSplit(DataSplit):
    """Реализует разбиение набора данных на выборки: тренировочную, валидационную (опционально), тестовую.
    Инициализируемые параметры:
    test_size: если float - то доля от общей выборки, если int - конкретное число элементов выборки.
    Если не указан,
    valid_size: аналогично test_size, но доля от обучающей выборки
    random_state: seed
    shuffle: флаг, отвечающий за перемешивание общей выборки"""
    
    test_size: float | int = field(default=None)
    valid_size: float | int = field(default=None)
    random_state: int | RandomState = field(default=None)
    shuffle: bool = field(default=False)
    
    def split(self, data):
        # TODO: требуется рефакторинг
        train_data, test_data = train_test_split(data,
                                                 test_size=self.test_size,
                                                 random_state=self.random_state,
                                                 shuffle=self.shuffle)
        
        if self.valid_size is not None:
            train_data, valid_data = train_test_split(train_data,
                                                      test_size=self.valid_size,
                                                      random_state=self.random_state,
                                                      shuffle=self.shuffle)
            return train_data, valid_data, test_data
        else:
            return train_data, test_data
