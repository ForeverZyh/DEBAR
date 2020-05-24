# Analysis

`analysis` folder contains the definition fo abstracted values and abstract interpretations for operations in computation graphs.

* `abstract_interpretation.py` contains the class type of abstracted values that we use for our tensor abstraction and interval abstraction with affine relations.
  The main component of `abstract_interpretation.py` is the `AbstractInterpretation` class, which is the data structure of abstracted values. It contains:

  * `size`: the shape of the tensor extracted from the protocol buffer format. The shape of the tensor may have an unknown dimension marked as $-1$ or ?. All shapes are inferred by TensorFlow.
  * `dtype`: the data type of the tensor extracted from the protocol buffer format. All data types are inferred by TensorFlow.
  * `value`: the interval abstraction stored in a `Range` object or a `numpy` concrete value.
  * `array`: the tensor partition stored in an `Array` object.
  * `constraints`: deprecated, used to store the z3 constraints generated alongside dataflow analysis.

  Notice that the value of `size`,  `dtype`, `value`, and `array` can be a list because the output value of a node can be a list. 

  * `index_of(self, i)` gets the $i$-th index of all the fields and returns a new `AbstractInterpretation` object. It returns `self` if `i` is `None`.
  * `has_none(self)` checks whether some of the fields are `None`, which indicates that dataflow analysis cannot infer this abstracted value due to unimplemented TensorFlow APIs. We do not necessarily need to throw an exception or to generate a warning to the unsafe operation under detection, because the input range of the unsafe operations may not depend on the unimplemented TensorFlow API.

* `inference.py` contains the abstract interpretations for operations in computation graphs. 

  * `real_size(a, b)` infers real size of `a` and `b` under the assumption that a = b, even though one of them might be unknown, i.e., equals to ?.
  * `dumy()`: returns an unbounded interval abstraction with [-inf, +inf].
  * `safeXXX(...)` are functions that calculate arithmetic function `XXX` in a numerical safe manner. 
  * `identity(args, node)`: the abstract interpretation of identity. It returns `None` if the input is a zero-size array with no concrete value. If the input is already an interval abstraction, then the input is returned. Otherwise, it converts the concrete `numpy` array (tensor) into its interval abstraction. `identity` will be called by some operations that only change the shape of the tensor, but will not change the values in the tensor or will only remove some values from the tensor so it is sound to abstract the operation by identity.
  * `packtorange(args, node)`: the abstract interpretation of joining ($\sqcup$) of a list of interval abstractions. It computes the lower bound of the output as the minimum of all lower bounds of abstracted inputs and concrete inputs. It computes the upper bound of the output as the maximum of all upper bounds of abstracted inputs and concrete inputs. `packtorange ` will be called by some operations like `pack`, `concatv2` that merge multiple tensors into one.
  * `InferValue` contains the abstract interpretations of TensorFlow APIs used in interval abstraction + tensor smashing.
    * All member methods in `InferValue` are static. 
    * Their names are the same as the lowercase operation names of the corresponding TensorFlow APIs in the protocol buffer format. This property should be preserved because the methods are being called by their names.
    * All methods accept two arguments, the first one is a list of abstracted values describing the inputs to the TensorFlow API, the second one is the node attribute in the protocol buffer format. The node attribute is used to extract additional information for DEBAR to understand the semantics of the API.
    * The methods return a `Range` object (or a concrete `numpy` array or a list of them).
  * `InferArray` contains the abstract interpretations of TensorFlow APIs used in the tensor partition and the linear affine relation.
    * All member methods have the same first three properties as described in `InferValue`.
    * The methods return an `Array` object (or a list of them).
    * Notice that ideally the `InferArray` shoule be split to `InferArray` and `InferLinear`, but the coupling of the tensor partition and the linear affine relation are so strong that we decide to implement them together in the `InferArray`.   

## Contributes to DEBAR by Implementing the Abstract Interpretations of TensorFlow APIs

We encourage the developers to contributes to DEBAR by implementing abstract interpretations of other TensorFlow APIs that are not handled by DEBAR. This Section is a guideline for implementing the abstract interpretations.

1. Please read the description of `InferValue` and `InferArray` in the previous Section. If you want to contribute to `InferValue`, please also read the `Range` class in [Overview](./overview.md). If you want to contribute to `InferArray`, please also read the `Array` and `Linear` class in  [Overview](./overview.md).
2. Contribute to `InferValue`: `InferValue` contains the abstract interpretations of TensorFlow APIs used in interval abstraction. Please make sure you add a method that:
   1. The method is static.
   2. The method name is the same as the lowercase operation name of the TensorFlow API in the protocol buffer format.
   3. The method accepts two arguments, the first one is a list of abstracted values describing the inputs to the TensorFlow API, the second one is the node attributes in the protocol buffer format.
   4. Please check the arity of the first argument if possible.
   5. Please make sure the abstract interpretation is sound. If you cannot handle some cases, it is always sound to return `None` or `dumy()` which instructs the dataflow analysis that part of the implementation is not available or the output range is unbounded.
   6. Returns a new `Range` object storing the results of the abstracted TensorFlow API.
3. Contribute to `InferArray`: `InferArray` contains the abstract interpretations of TensorFlow APIs used in the tensor partition and the linear affine relation. Please make sure you add a method that meets the 6 requirements above.
   Notice that it is not necessary to have an abstract interpretation in `InferArray` for any TensorFlow APIs. It is recommended to implement unhandled affine transformations. It is also recommended to implement some shape transformations whose input and output have the same interval abstraction. 

