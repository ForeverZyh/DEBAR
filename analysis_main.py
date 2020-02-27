from parse.parse_graph import Graph
import parse.parse_format_text
import z3
from solver import Range, meet
from utils import OVERFLOW_LIMIT, UNDERFLOW_LIMIT
import math
import sys


sys.setrecursionlimit(100000)
try:
    assert len(sys.argv) >= 2 and len(sys.argv) <= 3
    pbtxt = sys.argv[1]
    if len(sys.argv) == 3:
        assert sys.argv[2] in ["unbounded_weight", "unbounded_input"]
        if sys.argv[2] == "unbounded_weight":
            parse.parse_format_text.unbounded_weight = True
        else:
            parse.parse_format_text.unbounded_input = True
            
except:
    print(
        "Please run 'python test_script PBTEXT_FILENAME'.\nAborted...")
    exit(1)

rule = ["Log", "Exp", "RealDiv", "Sqrt", "Rsqrt", "Expm1", "Log1p", "Reciprocal"]
range_to_split_len_limit = 10
# rule = ["RealDiv"]
if __name__ == "__main__":
    graph = Graph(pbtxt, "verbose.txt")
    # graph.backward_slice("Log", set())
    # graph.draw(graph.backward_slice("SpatialTransformer/_transform/_interpolate/truediv", set()), "interested1")
    # graph.draw(graph.nodes_in_main_clique_topology, "show_real")
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
        # graph.draw(graph.backward_slice(suspected_node.name, set()), "real_interested")
        if suspected_node.op in ["RealDiv", "Floormod"]:
            ret = graph.forward_analysis(graph.node_by_name[graph.graph_backward[0][suspected_node.name][1]],
                                                 suspected_node)
        else:
            ret = graph.forward_analysis(suspected_node)
        if ret is None:
            continue
        elif ret == "ni":
            cnt_all += 1
            print(suspected_node.op, suspected_node.name)
            print("sat")
            cnt_unknown += 1
            continue

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
            
        def is_valid(input_range):
            additional_constraint = meet(input_range, suspected_node_input)
            S = z3.Solver()
            S.add(additional_constraint)
            ans = S.check()
            assert ans != z3.unknown
            return ans == z3.unsat
        
        def is_valid_by_split():
            if is_valid(graph.node_output[backward_analysis_const_start].index_of(index).value):
                return True
            else:
                range_to_split, nodes_interested = ret
                range_to_split = list(range_to_split)
#                 if len(range_to_split) > range_to_split_len_limit:
#                     return False
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
                            node_out = graph.reevaluate(nodes_interested, backward_analysis_const_start, changed, override_dict)
                            if not is_valid(node_out.index_of(index).value):
                                is_span_valid = False
                                break
                        if is_span_valid:
                            return True
                
                return False
                
                
        if not is_valid_by_split():
            print(suspected_node.op, suspected_node.name)
            print("sat")
            cnt_sat += 1
        else:
            cnt_unsat += 1
        cnt_all += 1
    print("all: ", cnt_all, "sat: ", cnt_sat, "unsat: ", cnt_unsat, "unknown because of API: ", cnt_unknown)
    print(graph.get_info())
