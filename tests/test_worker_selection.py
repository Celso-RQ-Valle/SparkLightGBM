from types import SimpleNamespace

from sparklightgbm._execution import _worker_host, resolve_num_workers


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


def test_auto_workers_reserve_capacity_and_use_a_conservative_cap():
    assert resolve_num_workers(_estimator(), _frame(parallelism=8, partitions=20)) == 4
    assert resolve_num_workers(_estimator(), _frame(parallelism=4, partitions=20)) == 3
    assert resolve_num_workers(_estimator(), _frame(parallelism=8, partitions=2)) == 2
    assert resolve_num_workers(_estimator(), _frame(parallelism=1, partitions=20)) == 1


def test_auto_workers_use_driver_for_local_and_ranking_modes():
    assert resolve_num_workers(_estimator(), _frame(master="local[8]")) == 1
    assert resolve_num_workers(_estimator(kind="ranker"), _frame()) == 1


def test_validation_does_not_disable_cluster_distribution():
    assert resolve_num_workers(_estimator(validation_data=object()), _frame()) == 4
    assert resolve_num_workers(_estimator(), _frame(), validation_data=object()) == 4
    assert resolve_num_workers(_estimator(early_stopping_rounds=5), _frame()) == 4


def test_explicit_worker_count_takes_precedence():
    assert resolve_num_workers(_estimator(num_workers=7, validation_data=object()), _frame(master="local[2]")) == 7


def test_local_worker_host_does_not_force_configured_loopback(monkeypatch):
    monkeypatch.setenv("SPARK_LOCAL_IP", "127.0.0.1")
    monkeypatch.setattr("socket.getfqdn", lambda: "worker.example")
    monkeypatch.setattr("socket.gethostbyname", lambda name: "192.0.2.10")
    assert _worker_host(allow_loopback=True) == "192.0.2.10"


def test_cluster_worker_host_honors_configured_address(monkeypatch):
    monkeypatch.setenv("SPARK_LOCAL_IP", "10.0.0.8")
    assert _worker_host(allow_loopback=False) == "10.0.0.8"
