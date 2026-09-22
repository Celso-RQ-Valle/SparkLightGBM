# Benchmarks

Run the scalable synthetic benchmark with an existing Spark installation:

```bash
python benchmarks/benchmark.py --rows 1000000 --features 50 --workers 4 --iterations 100 --output result.json
```

The generator is Spark-native and does not build the dataset on the driver. Results include generation, training, and materialized inference time; throughput; model size; row counts; and a prediction checksum. Run each configuration multiple times after a warm-up and retain the JSON output with Spark, Java, Python, LightGBM, machine, executor, and cluster configuration metadata.

For fair SynapseML comparisons, use the same persisted input DataFrame, feature representation, objective, seed, boosting parameters, worker count, and materializing action. Measure cluster CPU, peak executor/driver memory, and network traffic with the platform's metrics system; process-local measurements cannot accurately represent a Spark cluster.
