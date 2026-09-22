from types import SimpleNamespace

from sparklightgbm._execution import resolve_num_workers


class _RDD:
    def __init__(self, partitions):
        self.partitions = partitions

    def getNumPartitions(self):
        return self.partitions


def _frame(master="spark://cluster", parallelism=8, partitions=5):
    context = SimpleNamespace(master=master, defaultParallelism=parallelism)
    return SimpleNamespace(sparkSession=SimpleNamespace(sparkContext=context), rdd=_RDD(partitions))


def _estimator(num_workers=None, validation_data=None, early_stopping_rounds=None, kind="regressor"):
    return SimpleNamespace(num_workers=num_workers, validation_data=validation_data, early_stopping_rounds=early_stopping_rounds, kind=kind)


def test_auto_workers_are_bounded_by_slots_and_partitions():
    assert resolve_num_workers(_estimator(), _frame(parallelism=8, partitions=5)) == 5
    assert resolve_num_workers(_estimator(), _frame(parallelism=3, partitions=10)) == 3


def test_auto_workers_use_driver_for_local_and_validation_modes():
    assert resolve_num_workers(_estimator(), _frame(master="local[8]")) == 1
    assert resolve_num_workers(_estimator(validation_data=object()), _frame()) == 1
    assert resolve_num_workers(_estimator(), _frame(), validation_data=object()) == 1
    assert resolve_num_workers(_estimator(early_stopping_rounds=5), _frame()) == 1
    assert resolve_num_workers(_estimator(kind="ranker"), _frame()) == 1


def test_explicit_worker_count_takes_precedence():
    assert resolve_num_workers(_estimator(num_workers=7, validation_data=object()), _frame(master="local[2]")) == 7
