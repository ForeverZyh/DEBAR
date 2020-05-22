import z3
import math
import sys
import os

from parse.parse_graph import Graph
import parse.parse_format_text
from parse.specified_ranges import SpecifiedRanges
from solver import Range, meet
from utils import OVERFLOW_LIMIT, UNDERFLOW_LIMIT

if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    try:
        assert len(sys.argv) >= 2 and len(sys.argv) <= 3
        pbtxt = sys.argv[1]
        assert pbtxt[-6:] == ".pbtxt"
        if len(sys.argv) == 3:
            assert sys.argv[2] in ["unbounded_weight", "unbounded_input"]
            if sys.argv[2] == "unbounded_weight":
                parse.parse_format_text.unbounded_weight = True
            else:
                parse.parse_format_text.unbounded_input = True

    except:
        print(
            "Please run 'python analysis_main.py PBTXT_FILENAME'.\nAborted...")
        exit(1)

    rule = ["Log", "Exp", "RealDiv", "Sqrt", "Rsqrt", "Expm1", "Log1p", "Reciprocal"]

    network_name = os.path.basename(pbtxt)[:-6]
    if network_name in SpecifiedRanges.specified_ranges:
        SpecifiedRanges.ranges_looking_up = SpecifiedRanges.specified_ranges[network_name]

    graph = Graph(pbtxt, "verbose.txt")
    suspected_nodes = []
    for node in graph.graph_def.node:
        if node.op in rule and graph.f.find(node.name) == graph.main_clique:
            suspected_nodes.append(node)
    print(graph.get_info())

    cnt_all = 0
    cnt_sat = 0
    cnt_unknown = 0
    cnt_unsat = 0
    for suspected_node in suspected_nodes:
        # calculate the range of input of the unsafe operations
        if suspected_node.op in ["RealDiv", "Floormod"]:
            # special treatment for div because we only care about the denominator
            ret = graph.forward_analysis(graph.node_by_name[graph.graph_backward[0][suspected_node.name][1]],
                                         suspected_node)
        else:
            ret = graph.forward_analysis(suspected_node)
        if ret is None:
            continue
        # elif ret == "ni":
        #     cnt_all += 1
        #     print(suspected_node.op, suspected_node.name)
        #     print("unknown")
        #     cnt_unknown += 1
        #     continue

        if suspected_node.op in ["Exp", "Expm1"]:
            suspected_node_input = Range(left=math.log(OVERFLOW_LIMIT), right=None, const_type=0)
            backward_analysis_const_start = graph.graph_backward[0][suspected_node.name][0]
            index = graph.edge_index[suspected_node.name][0]
        elif suspected_node.op in ["RealDiv", "Floormod"]:
            suspected_node_input = Range(left=-UNDERFLOW_LIMIT, right=UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.graph_backward[0][suspected_node.name][1]
            index = graph.edge_index[suspected_node.name][1]
        elif suspected_node.op == "Log":
            suspected_node_input = Range(left=None, right=UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.graph_backward[0][suspected_node.name][0]
            index = graph.edge_index[suspected_node.name][0]
        elif suspected_node.op == "Sqrt":
            suspected_node_input = Range(left=None, right=-UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.graph_backward[0][suspected_node.name][0]
            index = graph.edge_index[suspected_node.name][0]
        elif suspected_node.op == "Rsqrt":
            suspected_node_input = Range(left=None, right=UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.graph_backward[0][suspected_node.name][0]
            index = graph.edge_index[suspected_node.name][0]
        elif suspected_node.op == "Log1p":
            suspected_node_input = Range(left=-UNDERFLOW_LIMIT - 1, right=UNDERFLOW_LIMIT - 1, const_type=0)
            backward_analysis_const_start = graph.graph_backward[0][suspected_node.name][0]
            index = graph.edge_index[suspected_node.name][0]
        elif suspected_node.op == "Reciprocal":
            suspected_node_input = Range(left=-UNDERFLOW_LIMIT, right=UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.graph_backward[0][suspected_node.name][0]
            index = graph.edge_index[suspected_node.name][0]
        else:
            raise NotImplementedError("No rule for ", suspected_node.op)


        # check whether the input_range intersects with its danger zone
        # return true if dose no intersect; otherwise, return false
        def is_valid(input_range):
            additional_constraint = meet(input_range, suspected_node_input)
            S = z3.Solver()
            S.add(additional_constraint)
            ans = S.check()
            assert ans != z3.unknown
            return ans == z3.unsat


        # check whether the unsafe operation's input is valid
        def is_valid_by_split():
            # if it is valid without predicate splitting
            if is_valid(graph.node_output[backward_analysis_const_start].index_of(index).value):
                return True
            else:
                # otherwise, try predicate splitting
                range_to_split, nodes_interested = ret
                range_to_split = list(range_to_split)
                for name in range_to_split:
                    override_dict = {}
                    # if the name has |, we have to remove it to get the name in the graph
                    changed = set()
                    if name.find('|') != -1:
                        changed.add(name[:name.find('|')])
                    else:
                        changed.add(name)
                    value = graph.get_value(name)
                    if value.left < 0 and value.right > 0:
                        spans = [Range(left=value.left, right=0), Range(left=0, right=value.right)]
                        is_span_valid = True
                        for span in spans:
                            override_dict[name] = span
                            # incrementally rerun the dataflow analysis on changed node set with the node output
                            # overridden to override_dict
                            node_out = graph.reevaluate(nodes_interested, backward_analysis_const_start, changed,
                                                        override_dict)
                            if not is_valid(node_out.index_of(index).value):
                                is_span_valid = False
                                break

                        if is_span_valid:
                            return True

                return False


        if not is_valid_by_split():
            print(suspected_node.op, suspected_node.name)
            print("warning")
            cnt_sat += 1
        else:
            cnt_unsat += 1
        cnt_all += 1
    print("all: ", cnt_all, "warnings: ", cnt_sat, "safe: ", cnt_unsat, "unknown because of API: ", cnt_unknown)
    print(graph.get_info())
