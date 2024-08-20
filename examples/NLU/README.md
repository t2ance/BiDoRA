## Code for NLU

Please follow these steps for running the code:

1.Create and activate conda env

```
cd examples/NLU
conda env create -f environment.yml
conda activate NLU
```

2.Install loralib

```
cd ..
pip install -e ..

```

3.Install transformers

```
cd NLU
pip install -e .

```

4.Run the scripts

Please run corresponding scripts for the results.

For example:

```
bash roberta_base_cola.sh
bash roberta_large_mnli.sh
```
