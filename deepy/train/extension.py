from pathlib import Path
import typing

import torch
import torch.nn as nn

from deepy.train.trainer import Trainer

class Trigger(object):
    """A Base class of triggers used for calling extension modules in training.

    Args:
    """
    def __init__(self) -> typing.NoReturn:
        super().__init__()
    
    def __call__(self, trainer: Trainer) -> bool:
        return True


class IntervalTrigger(Trigger):
    """Trigger based on a fixed interval.
    This trigger accepts iterations divided by a given interval.

    Args:
        period: Length of the interval
    """

    def __init__(self, period: int) -> typing.NoReturn:
        super().__init__()
        self.period = period
        self._previous_epoch = 0.

    def __call__(self, trainer: Trainer) -> bool:
        """Decides whether the extension should be called on this iteration.
        Args:
            trainer (deepy.train.Trainer):
                Trainer object that this trigger is associated with.
                The epoch information in this trainer is used to
                determine if the trigger should fire.
        Returns:
            True if the corresponding extension should be invoked in this
            iteration.
        """
        epoch = trainer.epoch
        previous_epoch = self._previous_epoch
        fire = previous_epoch // self.period != epoch // self.period
        self._previous_epoch = trainer.epoch

        return fire


class BestValueTrigger(Trigger):
    """Trigger based on the best value.

    Args:
        mode: ``train`` or ``validation``.
        key: Key of compared value.
        compare: Compare function which takes current best value and
            new value and returns whether new value is better than current
            best.
        trigger: Trigger that decides the comparison interval between current
            best value and new value.
    """

    def __init__(self, mode:str, key:str, compare:typing.Callable[[], bool],
                 trigger: IntervalTrigger=None) -> typing.NoReturn:
        super().__init__()
        self.mode = mode
        self.key = key
        self.compare = compare
        self.best_value = None
        if trigger is None:
            trigger = IntervalTrigger(1)
        self.trigger = trigger
    
    def __call__(self, trainer: Trainer) -> bool:
        """Decides whether the extension should be called on this iteration.

        Args:
            trainer: Trainer object that this trigger is associated with.
                The ``history`` of this trainer
                is used to determine if the trigger should fire.

        Returns:
            ``True`` if the corresponding extension should be invoked in
            this iteration.
        """

        if not self.trigger(trainer):
            return False

        history = trainer.history[self.mode][-1]
        key = self.key
        value = float(history[key])  # copy to CPU

        if self.best_value is None or self.compare(self.best_value, value):
            self.best_value = value
            return True
        return False



class MaxValueTrigger(BestValueTrigger):

    """Trigger invoked when specific value becomes maximum.
    For example you can use this trigger to take snapshot on the epoch the
    validation accuracy is maximum.

    Args:
        mode: ``train`` or ``validation``.
        key: Key of compared value.
        trigger: Trigger that decides the comparison interval between current
            best value and new value.
    """

    def __init__(self, mode:str, key:str, trigger: IntervalTrigger=None) -> typing.NoReturn:
        super().__init__(
            mode, key,
            lambda max_value, new_value: new_value > max_value, trigger)


class MinValueTrigger(BestValueTrigger):

    """Trigger invoked when specific value becomes minimum.
    For example you can use this trigger to take snapshot on the epoch the
    validation loss is minimum.

    Args:
        mode: ``train`` or ``validation``.
        key: Key of compared value.
        trigger: Trigger that decides the comparison interval between current
            best value and new value.
    """

    def __init__(self, mode:str, key:str, trigger: IntervalTrigger=None) -> typing.NoReturn:
        super().__init__(
            mode, key,
            lambda max_value, new_value: new_value < max_value, trigger)


class Extension(object):
    """A Base class of extensions for training models.

    Args:
        trigger: 
    """
    def __init__(self, trigger: Trigger) -> typing.NoReturn:
        super().__init__()
        self.trigger = trigger

    def initialize(self, trainer: Trainer) -> typing.NoReturn:
        """ Automatically called before training. Optional.

        Args:

        Returns:
        
        """
        pass

    def __call__(self, trainer: Trainer) -> typing.NoReturn:
        """ Called when the associated trigger is fired.
        """
        raise NotImplementedError()

    def state_dict(self):
        """ Used to serialize the state. Optional.
        """
        pass

    def load_state_dict(self, state):
        """ Used to deserialize the state. Optional.
        """
        pass


class ModelSaver(Extension):
    """Extension saving intermediate models in training.

    Args:
        directory: A directory where models will be saved.
        name: A function that returns a file name for saved model.
        trigger:
    """

    def __init__(self, directory: Path,
                 name: typing.Callable[[Trainer], str],
                 trigger: Trigger) -> typing.NoReturn:
        super().__init__()
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.name = name
        self.trigger = trigger
    
    def __call__(self, trainer: Trainer) -> typing.NoReturn:
        net = trainer.net
        name = self.name(trainer)
        torch.save(net.state_dict(), str(self.directory / name))