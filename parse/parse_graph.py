from google.protobuf import text_format
import tensorflow as tf
from analysis.inference import InferValue, InferArray, identity, dumy
from analysis.abstract_interpretation import AbstractInterpretation
import queue
from graphviz import Digraph
import warnings
import z3
from solver import meet, meet_relation_variable, magic
from solver import Range, Array, Solver
from utils import *
import numpy as np
import copy

turn_on_array = True


# implements the disjoint-set data structure https://en.wikipedia.org/wiki/Disjoint-set_data_structure
# for identifying the largest connected component in the parsed computation graph.
class UnionSet:
    def __init__(self, eles):
        self.f = {}
        self.rank = {}
        for ele in eles:
            self.f[ele] = ele
            self.rank[ele] = 1

    def find(self, x):
        if self.f[x] == x:
            return x
        self.f[x] = self.find(self.f[x])
        self.rank[x] = self.rank[self.f[x]]
        return self.f[x]

    def union(self, x, y):
        """merge y to x"""
        u = self.find(x)
        v = self.find(y)
        if u != v:
            if self.rank[u] < self.rank[v]:
                u, v = v, u
            self.f[v] = u
            self.rank[u] += self.rank[v]


# implements the parsing process of the Protocol Buffer file to the computation graph and the process of static
# dataflow analysis, as well as other functionalities that are related to the computation graph.
class Graph:
    def __init__(self, filename, verbose_file=None):
        with open(filename) as f:
            txt = f.read()
            self.graph_def = text_format.Parse(txt, tf.GraphDef())
            tf.import_graph_def(self.graph_def, name="")
            self.tf_graph = tf.get_default_graph()
        # storing the reversed edges of the computation graph
        self.graph_backward = [{}, {}]  # [0] for non_control; [1] for control
        # storing the edges of the computation graph
        self.graph_forward = [{}, {}]  # [0] for non_control; [1] for control
        # is a map mapping from the name of an operation (string) to the node attribute in protocol buffer format
        self.node_by_name = {}
        self.f = UnionSet([node.name for node in self.graph_def.node])
        # is a map mapping from the name of an operation (string) to an AbstractInterpretation object (or a list of
        # AbstractInterpretation objects) denoting the output of the node computed by dataflow analysis.
        self.node_output = {}
        # is a map mapping from the name of an operation to a list. The list indicates which value is passed to the
        # next node if the output of the current node is a list of AbstractInterpretation objects.
        self.edge_index = {}
        # is a set storing which nodes have been visited by data flow analysis and it is used for incremental
        # dataflow analysis.
        self.node_visited = set()
        self.unique_clique = []
        self.main_clique = None
        # is a map mapping from tensor name to the operation (node) name
        self.tensor_to_op = {}
        # is a map mapping from an operation name to its topological order, instructing the order of dataflow analysis
        self.nodes_in_main_clique_topology = {}
        self.file = None if verbose_file is None else open(verbose_file, "w")
        self.build()

    def write(self, x):
        if self.file is None:
            print(x)
        else:
            self.file.write(str(x) + "\n")

    # parses the Protocol Buffer format, builds the computation graph, and the topological order of the nodes.
    def build(self):
        for node in self.graph_def.node:
            self.node_by_name[node.name] = node
            self.node_output[node.name] = AbstractInterpretation()
        for op in self.tf_graph.get_operations():
            for tensor in op.values():
                self.tensor_to_op[tensor.name] = op.name

        # parse the protocol buffer format and builds the computation graph.
        for node in self.graph_def.node:
            self.graph_backward[0][node.name] = []
            self.graph_backward[1][node.name] = []
            self.edge_index[node.name] = []
            node_values = self.tf_graph.get_operation_by_name(node.name).values()

            if node_values is None or len(node_values) == 0:
                self.node_output[node.name] = AbstractInterpretation()
            elif len(node_values) > 1:
                self.node_output[node.name] = AbstractInterpretation(
                    size=[node_value.shape for node_value in node_values],
                    dtype=[node_value.dtype for node_value in node_values],
                    array=[Array(node.name + "|" + str(i), node_value.shape) for
                           (i, node_value) in enumerate(node_values)])
            else:
                self.node_output[node.name] = AbstractInterpretation(
                    size=node_values[0].shape, dtype=node_values[0].dtype,
                    array=Array(node.name, node_values[0].shape))
            for in_node_raw in node.input:
                is_control = False
                if in_node_raw[0] == '^':
                    in_node_raw = in_node_raw[1:]
                    is_control = True

                if in_node_raw in self.tensor_to_op:  # if the input is defined by the tensor's name
                    in_node = self.tensor_to_op[in_node_raw]

                    in_tensor_names = [tensor.name for tensor in self.tf_graph.get_operation_by_name(
                        self.tensor_to_op[in_node_raw]).values()]
                    if not is_control:
                        self.edge_index[node.name].append(None if len(
                            in_tensor_names) == 1 else in_tensor_names.index(in_node_raw))
                else:  # if the input is defined by the operation's name
                    in_node = in_node_raw

                    in_tensor_names = [tensor.name for tensor in self.tf_graph.get_operation_by_name(
                        in_node_raw).values()]
                    if not is_control:
                        self.edge_index[node.name].append(None if len(in_tensor_names) == 1 else 0)

                if in_node not in self.graph_forward[0]:
                    self.graph_forward[0][in_node] = []
                    self.graph_forward[1][in_node] = []
                self.graph_forward[is_control][in_node].append(node.name)
                self.graph_backward[is_control][node.name].append(in_node)
                self.f.union(in_node, node.name)

        max_rank = 0
        for node in self.f.f:
            if self.f.find(node) == node:
                self.unique_clique.append(node)
                max_rank = max(max_rank, self.f.rank[node])

        for node_name in self.unique_clique:
            if max_rank == self.f.rank[node_name]:
                self.main_clique = node_name

        node_inds = {}
        q = queue.Queue()
        nodes_in_main_clique = set()
        cnt = 0
        for node_name in self.f.f:
            node_inds[node_name] = 0 if node_name not in self.graph_backward[0] else len(
                self.graph_backward[0][node_name])  # is sufficient to only query in self.graph_backward[0]
            if self.f.find(node_name) == self.main_clique:
                nodes_in_main_clique.add(node_name)

        for node_name in nodes_in_main_clique:
            if node_inds[node_name] == 0:
                q.put(node_name)

        # build nodes_in_main_clique_topology instructing the order of dataflow analysis
        while True:
            while not q.empty():
                son = q.get()
                nodes_in_main_clique.remove(son)
                self.nodes_in_main_clique_topology[son] = cnt
                cnt += 1
                if son in self.graph_forward[0]:
                    for next_node_name in self.graph_forward[0][son]:
                        node_inds[next_node_name] -= 1
                        if node_inds[next_node_name] == 0 and next_node_name in nodes_in_main_clique:
                            q.put(next_node_name)

            if len(nodes_in_main_clique) == 0:
                break

            # identify loops
            min_ind = None
            for node_name in nodes_in_main_clique:
                if self.node_by_name[node_name].op == "Merge":
                    can_add = True
                    for in_node_name in self.graph_backward[0][node_name]:
                        if in_node_name in nodes_in_main_clique and self.node_by_name[
                            in_node_name].op != "NextIteration":
                            # if a Merge is not dominated by a NextIteration, then we cannot add it into the queue
                            can_add = False
                            break

                    if can_add and (min_ind is None or node_inds[node_name] < node_inds[min_ind]):
                        min_ind = node_name

            assert min_ind is not None
            q.put(min_ind)

    # returns a list of nodes in the backward slice starting at node
    def backward_slice(self, node, visited, non_control_only=True):  # return a list of nodes
        visited.add(node)
        ret = [node]
        for in_node in self.graph_backward[0][node]:
            if in_node not in visited:
                ret.extend(self.backward_slice(in_node, visited))
        if not non_control_only:
            for in_node in self.graph_backward[1][node]:
                if in_node not in visited:
                    ret.extend(self.backward_slice(in_node, visited))

        return ret

    def draw(self, clique, filename):
        dot = Digraph()
        clique = set(clique)
        for x in clique:
            dot.node(x, self.node_by_name[x].op)

        for node_name in clique:
            if node_name in self.graph_forward[0]:
                for is_contorl in range(2):
                    for next_node_name in self.graph_forward[is_contorl][node_name]:
                        if next_node_name in clique:
                            dot.edge(node_name, next_node_name, color="blue" if is_contorl == 0 else "red")

        dot.render("./%s.gv" % filename, view=False)

    # calculates the abstracted output of node son with its attribute u in protocol buffer format according to the
    # abstractions of its inputs while the abstracted outputs of some nodes have been overridden in override_dict.
    # override_dict is a map mapping the names to their overridden abstractions. It will only be used in predicate
    # splitting and handling element-wise Select operation.
    def summary_node(self, son, u, override_dict={}):
        self.write(son)
        parents_aps = []
        all_none = True
        for (i, in_node_name) in enumerate(self.graph_backward[0][son]):  # only care about non_control edges
            if in_node_name not in self.node_visited:
                # there is a loop, and the node is "Merge"
                assert self.node_by_name[in_node_name].op == "NextIteration"
                self.node_visited.add(in_node_name)
                self.node_output[in_node_name].value = dumy()

            parents_aps.append(self.node_output[in_node_name].index_of(self.edge_index[son][i]))
            all_none &= parents_aps[-1].has_none()

        temp = None
        temp_array = None
        if all_none and len(parents_aps) > 0:
            warnings.warn("fail to analysis %s due to None" % son, RuntimeWarning)
        else:
            try:
                temp = getattr(InferValue, u.op.lower())(parents_aps, u)
                if temp is not None and isinstance(temp, tuple):
                    raise AssertionError
            except AttributeError:
                if u.op.lower() in ["assert"]:
                    pass
                else:
                    temp = None
                    warnings.warn("fail to analysis %s due to NotImplemented" % son, RuntimeWarning)
            except AssertionError:
                raise AssertionError

            # TODO refactor the handling of Select operation to analysis/inference.py
            if u.op == "Select":  # special treatment for Select
                compare_node_name = self.graph_backward[0][son][0]
                compare_node = self.node_by_name[compare_node_name]
                branch_node_name = self.graph_backward[0][son][1:]
                branch_value = [self.node_output[branch_node_name[i - 1]].index_of(self.edge_index[son][i]).value for i
                                in range(1, 3)]
                branch_array = [self.node_output[branch_node_name[i - 1]].index_of(self.edge_index[son][i]).array for i
                                in range(1, 3)]
                if compare_node.op in ["GreaterEqual", "Greater", "LessEqual", "Less", "Equal", "NotEqual"]:
                    args = self.graph_backward[0][compare_node_name]  # args --> compare_node_name --> son
                    range_args = [identity([self.node_output[args[i]].index_of(self.edge_index[compare_node_name][i])])
                                  for i in range(2)]

                    # check whether the cond tensor can be determined to be all true or all false
                    def can_determine():
                        if compare_node.op == "GreaterEqual":
                            if range_args[0].left >= range_args[1].right:
                                return branch_value[0]
                            elif range_args[0].right < range_args[1].left:
                                return branch_value[1]
                        elif compare_node.op == "Greater":
                            if range_args[0].left > range_args[1].right:
                                return branch_value[0]
                            elif range_args[0].right <= range_args[1].left:
                                return branch_value[1]
                        elif compare_node.op == "LessEqual":
                            if range_args[0].right <= range_args[1].left:
                                return branch_value[0]
                            elif range_args[0].left > range_args[1].right:
                                return branch_value[1]
                        elif compare_node.op == "Less":
                            if range_args[0].right < range_args[1].left:
                                return branch_value[0]
                            elif range_args[0].left >= range_args[1].right:
                                return branch_value[1]
                        elif compare_node.op == "Equal":
                            if range_args[0].single() and range_args[1].single() and range_args[1].left == range_args[
                                0].left:
                                return branch_value[0]
                            elif range_args[0].left > range_args[1].right or range_args[0].right < range_args[1].left:
                                return branch_value[1]
                        elif compare_node.op == "NotEqual":
                            if range_args[0].single() and range_args[1].single() and range_args[1].left == range_args[
                                0].left:
                                return branch_value[1]
                            elif range_args[0].left > range_args[1].right or range_args[0].right < range_args[1].left:
                                return branch_value[0]
                        else:
                            raise NotImplementedError
                        return None

                    temp_ret = can_determine()
                    if temp_ret is not None:
                        temp = temp_ret
                    else:  # cannot determine the cond tensor
                        single_value_array_id = None
                        array = None
                        # the cond has the form of: arg0 cmp arg1
                        # we require one of two args to be single_value_array. If both are single_value_arrays, we
                        # will choose the first one
                        # single_value_array: partitions depend on only one variable in linear affine relation without
                        # relu.
                        for i in range(2):
                            array = self.node_output[args[i]].index_of(self.edge_index[compare_node_name][i]).array
                            single_value_array = True
                            for key in array.block_to_symbol:
                                group = array.block_to_symbol[key]
                                if len(group.value) > 1:
                                    single_value_array = False
                                    break
                                key = list(group.value.keys())[0]
                                if key[:len(magic)] == magic:  # we don't consider relu
                                    single_value_array = False
                                    break
                            if single_value_array:
                                single_value_array_id = i
                                break

                        if single_value_array_id is not None:
                            # compute the abstracted output of "GreaterEqual", "Greater", "LessEqual", "Less" operations
                            def compute(op, c):
                                values = []
                                # enumerate the branch id
                                for branch_id_select in range(2):
                                    override_dict = {}
                                    for key in array.block_to_symbol:
                                        group = array.block_to_symbol[key]
                                        if len(group.value) == 1:
                                            for (name, position) in group.value:
                                                factor = group.value[(name, position)]
                                                if factor == 0:
                                                    continue
                                                value = self.get_value(name)
                                                rhs = c * (1 / factor)
                                                if factor < 0:
                                                    rhs = Range(left=rhs.right, right=rhs.left)
                                                if (factor > 0) ^ (op in ["GreaterEqual", "Greater"]) ^ (
                                                        branch_id_select == 0):
                                                    # value >= rhs
                                                    override_dict[(name, position)] = Range(
                                                        left=max(value.left, rhs.left),
                                                        right=max(value.right, rhs.left))
                                                else:
                                                    # value <= rhs
                                                    override_dict[(name, position)] = Range(
                                                        left=min(value.left, rhs.right),
                                                        right=min(value.right, rhs.right))

                                    values.append(self.get_left_right(branch_array[branch_id_select].block_to_symbol,
                                                                      branch_node_name[branch_id_select], override_dict))
                                    if values[-1] is None:
                                        return None
                                return Range(left=min(values[0].left, values[1].left),
                                             right=max(values[0].right, values[1].right))

                            # compute the abstracted output of "Equal", "NotEqual"
                            def compute_equal(op, c):
                                values = []
                                # enumerate the branch id
                                for branch_id_select in range(2):
                                    override_dict = {}
                                    for key in array.block_to_symbol:
                                        group = array.block_to_symbol[key]
                                        if len(group.value) == 1:
                                            for (name, position) in group.value:
                                                factor = group.value[(name, position)]
                                                if factor == 0:
                                                    continue
                                                value = self.get_value(name)
                                                rhs = c * (1 / factor)
                                                if factor < 0:
                                                    rhs = Range(left=rhs.right, right=rhs.left)
                                                if (op == "NotEqual") ^ (branch_id_select == 0):
                                                    # value == rhs
                                                    override_dict[(name, position)] = Range(
                                                        left=max(value.left, rhs.left),
                                                        right=min(value.right, rhs.right))
                                                else:
                                                    # value != rhs
                                                    pass

                                    values.append(self.get_left_right(branch_array[branch_id_select].block_to_symbol,
                                                                      branch_node_name[branch_id_select], override_dict))
                                    if values[-1] is None:
                                        return None
                                return Range(left=min(values[0].left, values[1].left),
                                             right=max(values[0].right, values[1].right))

                            if compare_node.op in ["GreaterEqual", "Greater", "LessEqual", "Less"]:
                                if single_value_array_id == 1:
                                    temp_ret = compute(
                                        "Less" if compare_node.op in ["GreaterEqual", "Greater"] else "Greater",
                                        range_args[0])
                                else:
                                    temp_ret = compute(compare_node.op, range_args[1])
                            else:
                                if single_value_array_id == 1:
                                    temp_ret = compute_equal(compare_node.op, range_args[0])
                                else:
                                    temp_ret = compute_equal(compare_node.op, range_args[1])

                            if temp_ret is not None:
                                temp = temp_ret

            if turn_on_array:
                try:
                    for parents_ap in parents_aps:
                        assert parents_ap.array.index_slices is not None
                    temp_array = getattr(InferArray, u.op.lower())(parents_aps, u)
                    flag = True
                    if isinstance(self.node_output[son].dtype, list):
                        for x in self.node_output[son].dtype:
                            if int(x) == 10:
                                flag = False
                                break
                    else:
                        flag = int(self.node_output[son].dtype) != 10

                    if not flag:
                        temp_array = None
                except AttributeError:
                    pass
                except AssertionError:
                    pass

        self.node_output[son].value = temp

        if temp_array is not None and isinstance(temp, Range):
            self.node_output[son].array = temp_array
            if isinstance(temp_array, list):
                temp = []
                for (i, tmp_array) in enumerate(temp_array):
                    if temp_array[i].index_slices is None:
                        temp.append(self.node_output[son].value[i])
                        continue
                    value = self.get_left_right(tmp_array.block_to_symbol, son, override_dict)
                    if value is None:
                        temp.append(self.node_output[son].value[i])
                    else:
                        temp.append(value)
            elif temp_array.index_slices is not None:
                value = self.get_left_right(temp_array.block_to_symbol, son, override_dict)
                if value is not None:
                    temp = value

            self.node_output[son].value = temp

        self.node_output[son].constraints = None
        self.write(self.node_output[son])

    # is the body of dataflow analysis. It computes the abstracted output of node_interested, and returns the ranges
    # for predicate splitting. appended is the node of the unsafe operation.
    def forward_analysis(self, node_interested, appended=None):
        nodes_interested = self.backward_slice(node_interested.name, set(), True)  # only care about non_control edges
        # we do not consider operations related to gradient descent.
        for node in nodes_interested:
            if "gradient" in node.lower() and "stopgradient" not in node.lower():
                self.write("----------Gradients are not interested----------")
                return None

        nodes_interested.sort(key=lambda x: self.nodes_in_main_clique_topology[x])
        if appended is not None:
            if "gradient" in appended.name.lower() and "stopgradient" not in appended.name.lower():
                self.write("----------Gradients are not interested----------")
                return None
            nodes_interested.append(appended.name)

        pre_check = True
        for son in nodes_interested[:-1]:
            u = self.node_by_name[son]
            try:
                getattr(InferValue, u.op.lower())([], u)
            except AttributeError:
                if u.op.lower() not in ["assert", "nextiteration"]:
                    print(u.op, " not Implemented!")
            except:
                pass

        for son in nodes_interested[:-1]:
            u = self.node_by_name[son]
            if son in self.node_visited:
                continue

            self.node_visited.add(son)
            self.summary_node(son, u)

        range_to_split = set()
        for son in nodes_interested[:-1]:
            u = self.node_by_name[son]
            if u.op in ["Exp"]:  # if it is a non-linear function
                in_node_name = self.graph_backward[0][son][0]
                in_node_output = self.node_output[in_node_name].index_of(self.edge_index[son][0])
                non_self = True
                groups = in_node_output.array.block_to_symbol
                range_to_split_local = set()
                for key in groups:
                    group = groups[key]
                    for (name, position) in group.value:
                        if name == in_node_name:
                            non_self = False
                            break
                        factor = group.value[(name, position)]
                        if factor != 0:
                            if name[:len(magic)] == magic:
                                range_to_split_local.add(name[len(magic):])

                if non_self:
                    range_to_split.update(range_to_split_local)

        return range_to_split, nodes_interested[:-1]

    # reevaluates the dataflow analysis for nodes_interested which contains the nodes in the backward slice of
    # node_interested. The reevaluation is implemented in an incremental manner, which only reevaluates the nodes which
    # will be affected by nodes in changed. The abstracted outputs of nodes in changed are overridden in override_dict.
    def reevaluate(self, nodes_interested, node_interested, changed, override_dict):
        back_up = {}
        for son in nodes_interested:
            u = self.node_by_name[son]
            has_changed = False
            for in_node_name in self.graph_backward[0][son]:  # only care about non_control edges
                if in_node_name in changed:
                    has_changed = True
                    break

            if has_changed:
                back_up[son] = copy.deepcopy(self.node_output[son])
                self.summary_node(son, u, override_dict)
                changed.add(son)

        ret = copy.deepcopy(self.node_output[node_interested])
        # restore the back up
        for key in back_up:
            self.node_output[key] = back_up[key]
        return ret

    # gets the corresponding abstracted output in node_output. It will also consider the specially instrumented name
    # like "x|i" denoting the i-th element in the abstracted output.
    def get_value(self, name):
        if name.find("|") != -1:
            pos = name.find('|')
            index = int(name[pos + 1:])
            return identity([self.node_output[name[:pos]].index_of(index)])
        else:
            return identity([self.node_output[name].index_of(None)])

    # computes the abstracted output of node_name using the tensor partition and the linear affine relation with values
    # of some nodes overridden by override_dict . groups is the block_to_symbol field of the Array object.
    def get_left_right(self, groups: dict, node_name, override_dict):
        left = []
        right = []
        for key in groups:
            left_ele = 0
            right_ele = 0
            group = groups[key]
            new_relu = {}

            def get_value(name, position):
                if name in override_dict:
                    return override_dict[name]
                if (name, position) in override_dict:
                    return override_dict[(name, position)]
                return self.get_value(name)

            def update_ele(factor, value, is_relu):
                if is_relu:
                    value = Range(left=max(0, value.left), right=max(0, value.right))

                value = value * factor
                if factor < 0:
                    value.left, value.right = value.right, value.left
                return value

            for (name, position) in group.value:
                if name[:5] == magic:  # We first store relu_value
                    new_relu[(name, position)] = group.value[(name, position)]

            for (name, position) in group.value:
                if name[:5] == magic:  # We then skip relu_value
                    continue

                if name == node_name:
                    if name in override_dict or (name, position) in override_dict:
                        return get_value(name, position)
                    return None

                value = get_value(name, position)

                if value is None:  # only happens when self.node_output[name].index_of(index) is a zero-size array.
                    continue

                value = Range(left=value.left, right=value.right)

                non_relu_factor = group.value[(name, position)]
                relu_name = magic + name
                # we first del the key,value pair in the dict
                if (relu_name, position) in group.value:
                    relu_factor = group.value[(relu_name, position)]
                else:
                    relu_factor = 0

                # this will be encountered secondly
                # axiom: x - relu(x) = -relu(-x).
                if relu_factor < 0 and non_relu_factor > 0:
                    t = min(-relu_factor, non_relu_factor)
                    non_relu_factor -= t  # sub back the non_relu_factor
                    relu_factor += t  # add back the relu_factor
                    left_ele += min(0, value.left) * t
                    right_ele += min(0, value.right) * t

                if relu_factor > 0 and non_relu_factor < 0:
                    t = min(relu_factor, -non_relu_factor)
                    non_relu_factor += t  # add back the non_relu_factor
                    relu_factor -= t  # sub back the relu_factor
                    left_ele += max(0, -value.right) * t
                    right_ele += max(0, -value.left) * t

                # we add back non-zero relu_factor
                new_relu[(relu_name, position)] = relu_factor

                value = update_ele(non_relu_factor, value, False)
                left_ele = left_ele + value.left
                right_ele = right_ele + value.right

            for (name, position) in new_relu:
                non_relu = name[len(magic):]
                value = get_value(non_relu, position)

                if value is None:  # only happens when self.node_output[name].index_of(index) is a zero-size array.
                    continue

                value = Range(left=value.left, right=value.right)
                value = update_ele(new_relu[(name, position)], value, True)
                left_ele = left_ele + value.left
                right_ele = right_ele + value.right

            left.append(left_ele)
            right.append(right_ele)

        if len(left) == 0 or len(right) == 0:
            return None
        return Range(left=min(left), right=max(right))

    def get_info(self):
        variable_cnt = 0
        for op in self.node_by_name:
            if self.node_by_name[op].op.lower() in ["variablev2", "variable", "varhandleop"]:
                u = self.node_output[op].size
                if self.node_by_name[op].op.lower() == "varhandleop":
                    u = shape_from_proto(self.node_by_name[op].attr["shape"].shape)

                tmp = 1
                if str(u) == '<unknown>':
                    continue
                for x in u:
                    tmp *= int(x)
                variable_cnt += tmp

        return len(self.nodes_in_main_clique_topology), variable_cnt


def main():
    graph = Graph("./test.pbtxt")
    graph.backward_slice("Log", set())
