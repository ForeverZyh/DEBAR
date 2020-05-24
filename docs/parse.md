# Parse

* `parse_graph.py` contains the parsing process of the Protocol Buffer format to the computation graph and the process of static dataflow analysis. 

  * `UnionSet` implements the [disjoint-set data structure](https://en.wikipedia.org/wiki/Disjoint-set_data_structure) for identifying the largest connected component in the parsed computation graph.

  * `Graph` mainly implements the parsing process of the Protocol Buffer format to the computation graph and the process of static dataflow analysis, as well as other functionalities that are related to the computation graph. We describe the main components of `Graph`.

    * `graph_backward` is a field storing the reversed edges of the computation graph. `graph_backward[0]` stores the dataflow edges and `graph_backward[1]` stores the control flow edges. The `graph_backward[0]` is seldomly used since we only care about the data flow.

    * `graph_forward` is a field storing the edges of the computation graph. `graph_forward[0]` stores the dataflow edges and `graph_forward[1]` stores the control flow edges. Similarly, the `graph_forward[0]` is seldomly used since we only care about the data flow.

    * `node_by_name` is a map mapping from the name of an operation (string) to the node attribute in protocol buffer format.

    * `node_output` is a map mapping from the name of an operation to an `AbstractInterpretation` object (or a list of `AbstractInterpretation` objects) denoting the abstracted output of the node computed by dataflow analysis. 
      One thing to mention is that if the output of a node "x" is a list of tensors, we will use an instrumented string "x|i" to denote the i-th element in the list.

    * `edge_index` is a map mapping from the name of an operation to a list. The list indicates which value is passed to the next node if the output of the current node is a list of `AbstractInterpretation` objects.

      For example, node `x` has three edges `x -> y0`, `x -> y1`,`x -> y2` in order and the output of `x` is a list of  `AbstractInterpretation` objects `[a0, a1, a2, a3]`. Suppose that `x` passes `a0` to `y0`, `a3` to `y2`, and `a2` to `y3`, then `self.edge_index[x] = [0, 3, 2]`. 

    * `node_visited` is a set storing which nodes have been visited by dataflow analysis and it is used for incremental dataflow analysis.

    * `tensor_to_op` is a map mapping from tensor name to the name of the operation (node) that generates this tensor. However, a small number of node inputs are named by the tensor names but not by the operation names. The purpose of this map is to unify the naming rule.

    * `nodes_in_main_clique_topology` is a map mapping from an operation name to its topological order, instructing the order of dataflow analysis. We first identify the DAG part of the graph using the topological traverse of the graph. Then we identify the  loops in the graph and mark the loop entries. At last, we specify the order of traversing the loop to get the topological order of loops as well.

    * `build(self)` parses the Protocol Buffer format, builds the computation graph, and the topological order of the nodes. The `size`, `dtype` of `AbstractInterpretation` in `node_output` will be extracted from protocol buffer format in `build` method.

    * `backward_slice(self, node, visited, non_control_only)` returns a list of nodes in the backward slice starting at `node`. `visited` is a set recording which nodes have already been visited to avoid potential loops. `non_control_only` is a flag instructing the method whether to visit control flow edges.

    * `summary_node(self, son, u, override_dict)` calculates the abstracted output of node `son` with its attribute `u` in protocol buffer format according to the abstractions of its inputs while the abstracted outputs of some nodes have been overridden in `override_dict`. `override_dict` is a map mapping the names to their overridden abstractions. It will only be used in **predicate splitting** (see [Overview](./overview.md)) and **handling element-wise `Select` operation **(see next section).
      This method mainly contains two parts:

      1. The logic of computing `value` and `array` of `AbstractInterpretation` in `node_output`. It first computes `value` and `array` using the abstract interpretations in `analysis/inference.py`. Then it further improves the precision of `value` (interval abstraction + tensor smashing) by the information in `array` computed by the tensor partition and the linear affine relation. Notice that the results of the tensor partition and the linear affine relation will be provably more precise than or equal to the results of interval abstraction + tensor smashing. Thus, as long as the result of `array` is available, we will use the results of the tensor partition and the linear affine relation as `value`. `get_left_right` method computes the results of the tensor partition and the linear affine relation. 
      2. Abstract interpretation of the element-wise `Select` operation. Ideally, this part should be located in `analysis/`. However, the coupling between `Select` operation and dataflow analysis is so strong that we decide to leave it in `parse_graph.py`. Considering to refactor it into `analysis/`. The detail of this part can be found in the next section.

    * `forward_analysis(self, node_interested, appended)` is the body of dataflow analysis. It computes the abstracted output of `node_interested`, and returns the ranges for **predicate splitting** (see [Overview](./overview.md)). `appended` is the node of the unsafe operation. `node_interested` is one input of  `appended`. In most of the cases, `node_interested` is the only input of `appended`. For operations like `RealDiv`, we only care about the denominator so `node_interested` will be the second input of `appended` (denoting the denominator). 

      1. First, `forward_analysis` computes the backward slice from `node_interested` by calling `backward_slice`, and sorts the nodes in the backward slice in the topological order `nodes_in_main_clique_topology`. 
      2. Second, `forward_analysis` calls `summary_node` for every node in the backward slice in the topological order iteratively to get the abstracted output. If the node has already been visited by dataflow analysis, we can skip this node because the abstracted output has been computed when verifying other unsafe operations. 
      3. Third, `forward_analysis` collects and returns the ranges for predicate splitting.

    * `reevaluate(self, nodes_interested, node_interested, changed, override_dict)` reevaluates the dataflow analysis for `nodes_interested` which contains the nodes in the backward slice of `node_interested`. The reevaluation is implemented in an incremental manner, which only reevaluates the nodes which will be affected by nodes in `changed`. The abstracted outputs of nodes in `changed` are overridden in `override_dict`. 

    * `get_value(self, name)` gets the corresponding abstracted output in `node_output`. It will also consider the specially instrumented name like "x|i" denoting the i-th element in the abstracted output.

    * `get_left_right(self, groups, node_name, override_dict)` computes the abstracted output of `node_name` using the tensor partition and the linear affine relation with values of some nodes overridden by `override_dict` . `groups` is the `block_to_symbol` field of the `Array` object.
      The abstracted output is the joining ($\sqcup$) of all the abstracted outputs in tensor partitions stored in `groups`. The joining ($\sqcup$) of interval abstractions can be easily defined: setting the lower bound as the minimum of all lower bounds and the upper bound as the maximum of all upper bounds.
      The key is to compute the abstracted output of every tensor partition from the linear affine relation stored in the `Linear` object. Considering the example in Overview:
      $$
      3x-relu(x)+4y+5.
      $$
      This expression depends on the abstracted outputs of $x$ and $y$. Since we compute the abstracted outputs in the topological order, the abstracted outputs of $x$ and $y$ must have been computed previously. Thus, the abstracted value of this expression can be computed in the interval arithmetic. Moreover, the cancellation like $(x+y)+(x-y)=2y$ has been handled in `Linear` class. However, the cancellation of $relu$ is handled in `get_left_right` by the following axiom of $relu$ to get a more precise result:
      $$
      x - relu(x) = -relu(-x).
      $$
      

      Thus,
      $$
      \alpha(x - relu(x)) = -_{\alpha}relu_{\alpha}(-_{\alpha}\alpha(x)),
      $$
      where $\alpha(t)$ means the interval abstraction of $t$, and $-_{\alpha}$, $relu_{\alpha}$ are negation and $relu$ functions in interval arithmetic.

      For example, we have an expression $x-relu(x)$, where $\alpha(x)=[-1,2]$. Naive calculation $\alpha(x)-_{\alpha}relu_{\alpha}(\alpha(x))$ leads to interval $[-3,2]$. However, using the above axiom of $relu$ leads to interval $[-1,0]$, which is more precise than $[-3,2]$ computed by naive calculation.

* `parse_format_text.py` contains the parsing process of constant values, variables, and placeholders.

  * `const(node)` parses the constant values from the `node` attribute.
  * `iteratorv2(node)`, `oneshotiterator(node)` parse the inputs obtained by the `iteratorv2` and `oneshotiterator` operations and return a list of Range objects. The `node` attribute is used to get to `size` and `dtype` of the inputs.
  * `variablev2(node)` parses the weights obtained by the `variablev2` operation, and returns a Range object. The `node` attribute is used to get to `size` and `dtype` of the weights.
  * `placeholder(node, weight)` parses the inputs obtained by the placeholder operation, and returns a Range object. The `node` attribute is used to get to `size` and `dtype` of the inputs. `placeholder` can also be called by `variablev2` when `weight=True`.

* `specified_ranges.py` contains the reusable weights/inputs ranges specified by users. It mainly contains class `SpecifiedRanges` which has two static fields:

  * `models` is a list containing all the architecture names collected in our datasets. Notice that we shortened some of the architecture names to fit into the table in our paper.
  * `specified_ranges` is a map storing the reusable weights/inputs ranges specified by users. The map  has keys denoting architecture names and values containing another map mapping from variable names to their ranges. A range is a 2-elements list denoting the lower bound and the upper bound. If the lower bound is `None`, it means `-inf` and if the upper bound is `None`, it means `+inf`. We show how we infer these specified ranges for all architectures in the comments. 

## Abstract Interpretation of the Element-wise `Select` Operation

We implement the abstract interpretation of the element-wise `select` operation in `summary_node`. The element-wise `select` operation takes three inputs `cond`, `b1`, `b2`, and the return value `ret = select(cond, b1, b2)`, where `cond` is a bool tensor, `b1` , `b2`, and `ret` are two tensors with the same type.  Moreover, `cond`, `b1`, `b2`, and `ret` have the same shape. The semantics of element-wise `select` operation over 1-dimension tensors (vectors) is defined as follow:
$$
ret[i] = b1[i] \text{ if } cond[i] \text{ else } b2[i].
$$
The `ret[i]` is equal to `b1[i]` if `cond[i]` evaluates to true, otherwise,  `ret[i]` is equal to `b2[i]`.

### Motivation

We get the abstracted values of `cond`, `b1`, and `b2` before analyzing the abstracted value of `ret`. Considering the results obtained by the tensor smashing with the interval abstraction, the abstracted value `cond` vector can be in the following three cases:

1. *All true*, then the abstracted value of `ret` is equal to `b1`
2. *All false*, then the abstracted value of `ret` is equal to `b2`
3. *Otherwise*, then the abstracted value of `ret` is equal to the joining ($\sqcup$) of `b1` and `b2`.

Consider the following concrete example: `cond = x > 0`, `b1 = x`, and  `b2 = -x`, where `x` is a numerical vector with interval abstraction $\alpha(b1)=\alpha(x)=[-1,2]$ and $\alpha(b2)=\alpha(-x)=[-2,1]$. Thus, the abstraction of `cond` is the case *otherwise*, leading to $\alpha(ret) = [-1,2] \sqcup [-2,1]=  [-2,2]$.

However, this abstraction of `ret` is an over-approximation. We can get a more precise result by considering the range of the numerical vector in `cond`. If a value in `b1` is chosen, it implies that the corresponding element in `x` is greater than $0$. For the same reason, if a value in `b2` is chosen, it implies that the corresponding value in `x` is less than or equal to $0$. Thus, the interval abstraction $\alpha(b1)$ and $\alpha(b2)$ can be improved to $[0,2]$ and $[0,1]$ respectively, leading to $\alpha(ret) = [0,2] \sqcup [0,1]=  [0,2]$.

### Details

Like **predicate splitting**, the abstract interpretation of element-wise `select` operation needs to "split" the numerical tensors used in `cond`, reevaluate the abstracted outputs of two branches, and finally compute the abstracted output of `select` operation by joining ($\sqcup$) abstracted outputs of two branches.

`cond` has the form of `arg0 cmp arg1`, where `arg0` and `arg1` are numerical tensors, and `cmp` is the compare operation. We require that one of (or both of) `arg0` and `arg1` depends on only one variable in the linear affine relation without $relu$. The one satisfies the above requirement, say node `x`,  will be split according to the comparison operation `cmp`. (If both of them satisfy the above requirement, we will choose the first one.) Without losing the generalizability, the `cond` can be written as `x cmp y`, where $\alpha(x)=[l_x,u_x]$ and $\alpha(y)=[l_y,u_y]$. We summarize the splits of `x` for different comparison operations `cmp`:

| `cmp`                     | $\alpha(x)$ for `b1`             | $\alpha(x)$ for `b2`             |
| ------------------------- | -------------------------------- | -------------------------------- |
| `GreaterEqual`, `Greater` | $[\max(l_x,l_y), \max(u_x,l_y)]$ | $[\min(l_x,u_y), \min(u_x,u_y)]$ |
| `LessEqual`, `Less`       | $[\min(l_x,u_y), \min(u_x,u_y)]$ | $[\max(l_x,l_y), \max(u_x,l_y)]$ |
| `NotEqual`                | $[l_x,u_x]$                      | $[\max(l_x,l_y), \min(u_x,u_y)]$ |
| `Equal`                   | $[\max(l_x,l_y), \min(u_x,u_y)]$ | $[l_x,u_x]$                      |

## Specify Ranges of Weights and Inputs

We provide all the specified ranges of weights and inputs in `specified_ranges.py`, not only for the reproduction of the evaluation results but also for users to specify ranges of weights and inputs by taking examples of provided cases. We find that specifying ranges of weights and inputs can eliminate unnecessary false positives. Thus, we hope these examples can help reduce false positives and users' manually inspection time.

Here is a short guideline for adding specified ranges in your setup:

1. Please read the description of `parse_format_text.py` and `specified_ranges.py` in the previous Section.
2. Add a data entry with the architecture name as the key and the mapping from variable names to their ranges as the value.
3. Make sure the types of ranges are matched. For `iteratorv2` and `oneshotiterator`, the types are lists of 2-elements lists. For `variablev2` and `placeholder`, the types are 2-elements lists.

