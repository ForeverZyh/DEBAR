from google.protobuf import text_format
import tensorflow as tf
from analysis.inference import InferValue
from analysis.inference import InferConstant
from analysis.inference import InferSize
from analysis.abstract_interpretation import AbstractInterpretation
import queue
from graphviz import Digraph
import warnings
import z3
from solver import meet, check_range_const
from itertools import product
from solver import Range


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


class Graph:
    def __init__(self, filename, verbose_file=None):
        with open(filename) as f:
            txt = f.read()
            self.graph_def = text_format.Parse(txt, tf.GraphDef())
            tf.import_graph_def(self.graph_def, name="")
            self.tf_graph = tf.get_default_graph()
            # for op in self.tf_graph.get_operations():
            #     print("==============")
            #     print(op.name, op.values())
            #     tensors = op.values()
            #     for tensor in tensors:
            #         print(tensor.shape, int(tensor.dtype))
            #     print("==============")
        self.graph_backward = {}
        self.graph_forward = {}
        self.node_by_name = {}
        self.f = UnionSet([node.name for node in self.graph_def.node])
        self.node_output = {}
        self.edge_index = {}
        self.node_visited = set()
        self.unique_clique = []
        self.main_clique = None
        self.tensor_to_op = {}
        self.nodes_in_main_clique_topology = {}
        self.build()
        self.file = None if verbose_file is None else open(verbose_file, "w")

    def write(self, x):
        if self.file is None:
            print(x)
        else:
            self.file.write(str(x) + "\n")

    def build(self):
        for node in self.graph_def.node:
            self.node_by_name[node.name] = node
            self.node_output[node.name] = AbstractInterpretation()
        for op in self.tf_graph.get_operations():
            for tensor in op.values():
                self.tensor_to_op[tensor.name] = op.name

        for node in self.graph_def.node:
            self.graph_backward[node.name] = []
            node_values = self.tf_graph.get_operation_by_name(node.name).values()

            if node_values is None or len(node_values) == 0:
                self.node_output[node.name] = AbstractInterpretation()
            elif len(node_values) > 1:
                self.node_output[node.name] = AbstractInterpretation(
                    size=[node_value.shape for node_value in node_values],
                    dtype=[node_value.dtype for node_value in node_values])
            else:
                self.node_output[node.name] = AbstractInterpretation(
                    size=node_values[0].shape, dtype=node_values[0].dtype)
            for in_node_raw in node.input:
                is_control = False
                if in_node_raw[0] == '^':
                    in_node_raw = in_node_raw[1:]
                    is_control = True
                # print(in_node_raw)
                if in_node_raw in self.tensor_to_op:  # if the input is defined by the tensor's name
                    in_node = self.tensor_to_op[in_node_raw]

                    in_tensor_names = [tensor.name for tensor in self.tf_graph.get_operation_by_name(
                        self.tensor_to_op[in_node_raw]).values()]
                    self.edge_index[(in_node, node.name)] = None if len(
                        in_tensor_names) == 1 else in_tensor_names.index(in_node_raw)
                else:  # if the input is defined by the operation's name
                    in_node = in_node_raw

                    in_tensor_names = [tensor.name for tensor in self.tf_graph.get_operation_by_name(
                        in_node_raw).values()]
                    self.edge_index[(in_node, node.name)] = None if len(in_tensor_names) == 1 else 0

                if in_node not in self.graph_forward:
                    self.graph_forward[in_node] = []
                self.graph_forward[in_node].append((node.name, is_control))
                self.graph_backward[node.name].append((in_node, is_control))
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
            node_inds[node_name] = 0 if node_name not in self.graph_backward else len(self.graph_backward[node_name])
            if self.f.find(node_name) == self.main_clique:
                nodes_in_main_clique.add(node_name)

        for node_name in nodes_in_main_clique:
            if node_inds[node_name] == 0:
                q.put(node_name)

        while True:
            while not q.empty():
                son = q.get()
                nodes_in_main_clique.remove(son)
                self.nodes_in_main_clique_topology[son] = cnt
                cnt += 1
                if son in self.graph_forward:
                    for (next_node_name, _) in self.graph_forward[son]:
                        node_inds[next_node_name] -= 1
                        if node_inds[next_node_name] == 0 and next_node_name in nodes_in_main_clique:
                            q.put(next_node_name)

            if len(nodes_in_main_clique) == 0:
                break

            min_ind = None
            for node_name in nodes_in_main_clique:
                flag = False
                if node_name in self.graph_backward:
                    for (in_node_name, _) in self.graph_backward[node_name]:
                        if self.edge_index[(in_node_name, node_name)] is not None:
                            flag = True
                            break
                if flag:
                    continue
                if self.node_by_name[node_name].op == "Merge":
                    if min_ind is None or node_inds[node_name] < node_inds[min_ind]:
                        min_ind = node_name

            assert min_ind is not None
            q.put(min_ind)

    def backward_slice(self, node, visited):  # return a list of nodes
        # print(self.node_by_name[node].op, " : ", self.graph_backward[node])
        visited.add(node)
        ret = [node]
        for (in_node, _) in self.graph_backward[node]:
            if in_node not in visited:
                ret.extend(self.backward_slice(in_node, visited))
        return ret

    def draw(self, clique, filename):
        dot = Digraph()
        clique = set(clique)
        # print(clique)
        for x in clique:
            dot.node(x, self.node_by_name[x].op)

        for node_name in clique:
            if node_name in self.graph_forward:
                for (next_node_name, is_control) in self.graph_forward[node_name]:
                    if next_node_name in clique:
                        dot.edge(node_name, next_node_name, color="blue" if is_control else "red")

        # print(dot.source)
        dot.render("./%s.gv" % filename, view=False)

    def forward_analysis(self, node_interested, appended=None):
        nodes_interested = self.backward_slice(node_interested.name, set())
        for node in nodes_interested:
            if "gradient" in node.lower() and "stopgradient" not in node.lower():
                print("----------Gradients are not interested----------")
                return None

        nodes_interested.sort(key=lambda x: self.nodes_in_main_clique_topology[x])
        if appended is not None:
            if "gradient" in appended.name.lower() and "stopgradient" not in appended.name.lower():
                print("----------Gradients are not interested----------")
                return None
            nodes_interested.append(appended.name)

        for son in nodes_interested[:-1]:
            u = self.node_by_name[son]
            if son in self.node_visited:
                self.write(str(son) + "passed")
                continue

            self.write(son)
            self.node_visited.add(son)
            parents_aps = []
            all_none = True
            for (in_node_name, is_control) in self.graph_backward[son]:
                if not is_control:
                    if in_node_name not in self.node_visited:
                        # there is a loop, and the node is "Merge"
                        assert self.node_by_name[in_node_name].op == "NextIteration"
                        self.node_visited.add(in_node_name)
                        self.node_output[in_node_name].value = Range(name="nextiteration",
                                                                     dtype=self.node_output[in_node_name].dtype)

                    parents_aps.append(self.node_output[in_node_name].index_of(self.edge_index[(in_node_name, son)]))
                    all_none &= parents_aps[-1].has_none()

            new_size = None
            temp = None
            if all_none and len(parents_aps) > 0:
                warnings.warn("fail to analysis %s due to None" % son, RuntimeWarning)
            else:
                try:
#                     if str(self.node_output[son].size) == "<unknown>":
#                         new_size = getattr(InferSize, u.op.lower())(parents_aps, u)
                    temp = getattr(InferValue, u.op.lower())(parents_aps, u)
                except AttributeError:
                    if u.op.lower() in ["assert"]:
                        pass
                    else:
                        raise AttributeError
                except AssertionError:
                    raise AssertionError
                # except:
                #     warnings.warn("fail to analysis %s due to None" % son, RuntimeWarning)
                #     temp = None

            if new_size is not None:
                self.node_output[son].size = new_size
            if isinstance(temp, tuple):
                self.node_output[son].value = temp[0]
                self.node_output[son].constraints = temp[1]
            else:
                self.node_output[son].value = temp

            self.write(self.node_output[son])

        ret_constraints = []
        for son in nodes_interested[:-1]:
            if self.node_output[son].constraints is not None:
                ret_constraints.append(self.node_output[son].constraints)
                self.write(self.node_output[son].constraints)
        return z3.And(ret_constraints)

    def backward_analysis_const(self, node, range_const):
        if self.node_output[node.name].value is not None:
            yield meet(self.node_output[node.name].value, range_const)
        else:
            raise NotImplementedError
            # in_node_values = []
            # for (in_node, is_control) in self.graph_backward[node.name]:
            #     if not is_control:
            #         in_node_values.append(self.node_output[in_node])
            # for in_node_value_ranges in getattr(InferConstant, node.op.lower())(in_node_values, range_const, node):
            #     idx = 0
            #     gens = []
            #     for (in_node, is_control) in self.graph_backward[node.name]:
            #         if not is_control:
            #             if in_node_value_ranges[idx] is not None:
            #                 if not check_range_const(in_node_value_ranges[idx]):
            #                     return False
            #                 gens.append(
            #                     self.backward_analysis_const(self.node_by_name[in_node], in_node_value_ranges[idx]))
            #             idx += 1
            #     for rets in product(*gens):
            #         if False in rets:
            #             yield False
            #         else:
            #             yield z3.And(rets)


def main():
    graph = Graph("./test.pbtxt")
    graph.backward_slice("Log", set())
