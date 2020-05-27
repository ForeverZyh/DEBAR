# DEBAR: *DE*tecting Numerical *B*ugs in Neural Network *AR*chitectures


This repository contains the implementation and the evaluation of our upcoming ESEC/FSE 2020 paper:  Detecting Numerical Bugs in Neural Network Architectures.

DEBAR can detect numerical bugs in neural networks at the architecture level (without concrete weights and inputs, before the training session of the neural network).

## Collected Datasets

We share our two collected datasets and evaluation results [online](https://drive.google.com/uc?export=download&id=1GBHFd-fPIBWqJOpIC8ZO8g3F1LoIZYNn). 

The first dataset is a set of 9 buggy architectures collected by existing studies. The buggy architectures come from two studies: eight architectures were collected by a previous [empirical study on TensorFlow bugs](https://github.com/ForeverZyh/TensorFlow-Program-Bugs) (Github/StackOverflow-IPS-id.pbtxt) and one architecture was obtained from the study that proposes and evaluates [TensorFuzz](https://github.com/brain-research/tensorfuzz/blob/master/bugs/collection_bug.py) (TensorFuzz.pbtxt). 

The second dataset contains 48 architectures from a large collection of research projects in TensorFlow Models repository. Overall, our second dataset contains a great diversity of neural architectures like CNN, RNN, GAN, HMM, and so on. Please note that we have no knowledge about whether the architectures in this dataset contain numerical bugs when collecting the dataset.

For every architecture in two datasets, we extract the computation graph by using a TensorFlow API. Each extracted computation graph is represented by a Protocol Buffer file, which provides the operations (nodes) and the data flow relations (edges).

## Setups

There are two ways you can run DEBAR:

1. Run in docker.
2. Run in virtual environments with virtualenv or conda.

### Setups for docker

Install docker and type the following command to build the image.

```bash
docker build -t debar .
```

Then type the following command to start a bash into the image.

```bash
docker run -it debar:latest bash
```

### Setups for virtual environments with virtualenv or conda

#### Environment 

DEBAR runs on python3 (>=3.5).

We encourage users to use virtual environments such as virtualenv or conda. Make sure you are in a virtual environment and then follow the steps:

```bash
pip install -r requirements.txt
```

The current implementation of DEBAR only supports detecting numerical bugs in static computation graphs in TensorFlow.  If you want a GPU version of TensorFlow, which can accelerate the loading process of protocol buffer files into (GPU) memory.

```bash
pip install tensorflow-gpu==1.13.1
```

or a CPU version:

```bash
pip install tensorflow==1.13.1
```

DEBAR has a dependency on TensorFlow v1 but is not compatible with TensorFlow v2. You may also notice that DEBAR has a dependency of z3-solver, it is due to some legacy during development which may be removed later.

#### Dataset

We share our two collected datasets and evaluation results [online](https://drive.google.com/uc?export=download&id=1GBHFd-fPIBWqJOpIC8ZO8g3F1LoIZYNn). You can manually download from the link, or

```bash
curl -L -o dataset.zip 'https://drive.google.com/uc?export=download&id=1GBHFd-fPIBWqJOpIC8ZO8g3F1LoIZYNn'
```

## Running DEBAR

```bash
python analysis_main.py PBTXT_FILE [unbounded_weight/unbounded_input]
```

The above command shows how to run DEBAR. The first argument to  `analysis_main.py` is the Protocol Buffer file describing the target computation graph. 

The second argument is a [optional] flag denoting whether to specify the range of the weights and the range of the inputs.

*  The default value (do not pass the second argument) means to specify the range of the weights and the range of the inputs.
* `unbounded_weight` means to specify the range of the inputs, but leave the weights unbounded, which means the ranges of weights will be set to `[-inf,+inf]`.
* `unbounded_input` means to specify the range of the weights, but leave the inputs unbounded, which means the ranges of inputs will be set to `[-inf,+inf]`.

The specification of ranges of weights/inputs can be given in two ways:

* Input to the console: During running, DEBAR will prompt the name of the node denoting weights/inputs, if the node name does not exist in `./parse/specified_ranges.py`. Then users can input the specified ranges into the console. 
* `./parse/specified_ranges.py`: Manually store the ranges in `./parse/specified_ranges.py` for future reproduction. Please see the documentation [Parse](./docs/parse.md) for more information. 

The recommended way of specifying ranges is first trying to input to the console and then manually store the ranges in `./parse/specified_ranges.py` if future reproduction is needed.

### Example

If you unzip the dataset in the DEBAR directory, you can type the following command to get the result of `Github-IPS-1`.

```bash
python analysis_main.py ./computation_graphs_and_TP_list/computation_graphs/Github-IPS-1.pbtxt
```

Our tool will report the following:

```
(1546, 5052902)
Log Log
warning
Github-IPS-1 , all:  5 	warnings:  1 	safe:  4
```

, which means there are 5 unsafe operations in total. Among them, 1 warning is generated on the operation `Log` with name `Log` and the other 4 unsafe operations are verified to be safe.
DEBAR will also output the basic information of the architecture: `(1546, 5052902)` means that there are 1546 operations and 5052902 parameters in the architecture.

## Reproduce Evaluation in our Paper

There are four tags showing the four configurations mentioned in our paper.

* `partition-affine`: We use the **tensor partitioning** as the abstraction for tensors and interval abstraction **with affine relations**.
* `smashing-affine`: We use the **tensor smashing** as the abstraction for tensors and interval abstraction **with affine relations**.
* `expansion-affine`: We use the **tensor expansion** as the abstraction for tensors and interval abstraction **with affine relations**.
* `partition-wo-affine`: We use the **tensor smashing** as the abstraction for tensors and interval abstraction **without affine relations**.

Please checkout to each tag and the following command, where `PATH_TO_DATASETS` is the path of the downloaded datasets, is supposed to reproduce the evaluation results in our paper. **In docker, `PATH_TO_DATASETS` is `./computation_graphs_and_TP_list/computation_graphs`.**

```bash
python main.py PATH_TO_DATASETS
```

The above command will only report one summary line for each architecture. For example, it will report the following summary line for the architecture Github-IPS-1:

    Github-IPS-1 , all:  5 	warnings:  1 	safe:  4

And the full output will be stored at ./results.txt.

Notice that we manually classify the warnings to true positives and false positives. The result and reason for each warning are reported in ./computation_graphs_and_TP_list/true_positives.csv (inside the collected datasets).


## Published Work

Yuhao Zhang, Luyao Ren, Liqian Chen, Yingfei Xiong, Shing-Chi Cheung, Tao Xie. Detecting Numerical Bugs in Neural Network Architectures.



For more information, please refer to the documentation under the `docs/` directory.

* [Overview](./docs/overview.md)
* [Analysis](./docs/analysis.md)
* [Parse](./docs/parse.md)

* [Troubleshoot](./docs/troubleshoot.md)
* [DynamicTool](./docs/dynamic_tool.md)

