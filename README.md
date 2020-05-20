# DEBAR: *DE*tecting Numerical *B*ugs in Neural Network *AR*chitectures


This repository contains the implementation and the evaluation of our upcoming ESEC/FSE 2020 paper:  Detecting Numerical Bugs in Neural Network Architectures.

DEBAR can detect numerical bugs in neural networks at the architecture level (without concrete weights and inputs, before the training session of the neural network).

## Environment 

We encourage users to use virtual environments such as virtualenv or conda.

```bash
pip install -r requirements.txt
```

The current implementation of DEBAR only supports detecting numerical bugs in static computation graphs in TensorFlow. 

DEBAR has a dependency on TensorFlow v1 but is not compatible with TensorFlow v2. You may also notice that DEBAR has a dependency of z3-solver, it is due to some legacy during development which may be removed later.

## Collected Datasets

We share our two collected datasets and evaluation results [online](https://drive.google.com/file/d/146UDCTFbjO3Wz_BcCVnRkyCo529dxmFk/view?usp=sharing). 

The first dataset is a set of 9 buggy architectures collected by existing studies. The buggy architectures come from two studies: eight architectures were collected by a previous [empirical study on TensorFlow bugs](https://github.com/ForeverZyh/TensorFlow-Program-Bugs) (Github/Stackoverflow-IPS-id.pbtxt) and one architecture was obtained from the study that proposes and evaluates [TensorFuzz](https://github.com/brain-research/tensorfuzz/blob/master/bugs/collection_bug.py) (TensorFuzz.pbtxt). 

The second dataset contains 48 architectures from a large collection of research projects in TensorFlow Models repository. Overall, our second dataset contains a great diversity of neural architectures like CNN, RNN, GAN, HMM, and so on. Please note that we have no knowledge about whether the architectures in this dataset contain numerical bugs when collecting the dataset.

For every architecture in two datasets, we extract the computation graph by using a TensorFlow API. Each extracted computation graph is represented by a Protocol Buffer file, which provides the operations (nodes) and the data flow relations (edges).

### Running DEBAR

```bash
python analysis_main.py PXTXT_FILE [unbounded_weight/unbounded_input]
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

### Reproduce Evaluation in our Paper

There are four tags showing the four configurations mentioned in our paper.

* `partition-affine`: We use the **tensor partitioning** as the abstraction for tensors and interval abstraction **with affine relations**.
* `smashing-affine`: We use the **tensor smashing** as the abstraction for tensors and interval abstraction **with affine relations**.
* `expansion-affine`: We use the **tensor expansion** as the abstraction for tensors and interval abstraction **with affine relations**.
* `partition-wo-affine`: We use the **tensor smashing** as the abstraction for tensors and interval abstraction **without affine relations**.

Please checkout to each tag and the following command is supposed to reproduce the results in our paper.

```bash
python main.py
```

## Published Work

Yuhao Zhang, Luyao Ren, Liqian Chen, Yingfei Xiong, Shing-Chi Cheung, Tao Xie. Detecting Numerical Bugs in Neural Network Architectures.



For more information, please refer to the documentation under the `docs/` directory.

* [Overview](./docs/overview.md)
* [Analysis](./docs/analysis.md)
* [Parse](./docs/parse.md)

* [Troubleshoot](./docs/troubleshoot.md)
* [DynamicTool](./docs/dynamic_tool.md)

