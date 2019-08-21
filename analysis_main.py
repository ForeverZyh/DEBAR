from parse.parse_graph import Graph
import z3
from solver import Range, meet
from utils import OVERFLOW_LIMIT, UNDERFLOW_LIMIT
import math
import sys

sys.setrecursionlimit(100000)
try:
    assert len(sys.argv) == 2
    pbtxt = sys.argv[1]
except:
    print(
        "Please run 'python test_script PBTEXT_FILENAME'.\nAborted...")
    exit(1)

rule = ["Log", "Exp", "RealDiv", "Sqrt", "Rsqrt", "Expm1", "Log1p", "Reciprocal"]
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
            constraints = graph.forward_analysis(graph.node_by_name[graph.graph_backward[suspected_node.name][1][0]],
                                                 suspected_node)
        else:
            constraints = graph.forward_analysis(suspected_node)
        if constraints is None:
            continue

        if suspected_node.op in ["Exp", "Expm1"]:
            suspected_node_input = Range(left=math.log(OVERFLOW_LIMIT), right=None, const_type=0)
            backward_analysis_const_start = graph.graph_backward[suspected_node.name][0][0]
            index = graph.edge_index[suspected_node.name][0]
        elif suspected_node.op in ["RealDiv", "Floormod"]:
            suspected_node_input = Range(left=-UNDERFLOW_LIMIT, right=UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.graph_backward[suspected_node.name][1][0]
            index = graph.edge_index[suspected_node.name][1]
        elif suspected_node.op == "Log":
            suspected_node_input = Range(left=None, right=UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.graph_backward[suspected_node.name][0][0]
            index = graph.edge_index[suspected_node.name][0]
        elif suspected_node.op == "Sqrt":
            suspected_node_input = Range(left=None, right=-UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.graph_backward[suspected_node.name][0][0]
            index = graph.edge_index[suspected_node.name][0]
        elif suspected_node.op == "Rsqrt":
            suspected_node_input = Range(left=None, right=UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.graph_backward[suspected_node.name][0][0]
            index = graph.edge_index[suspected_node.name][0]
        elif suspected_node.op == "Log1p":
            suspected_node_input = Range(left=-UNDERFLOW_LIMIT - 1, right=UNDERFLOW_LIMIT - 1, const_type=0)
            backward_analysis_const_start = graph.graph_backward[suspected_node.name][0][0]
            index = graph.edge_index[suspected_node.name][0]
        elif suspected_node.op == "Reciprocal":
            suspected_node_input = Range(left=-UNDERFLOW_LIMIT, right=UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.graph_backward[suspected_node.name][0][0]
            index = graph.edge_index[suspected_node.name][0]
        else:
            raise NotImplementedError("No rule for ", suspected_node.op)

        try:
            input = graph.node_output[backward_analysis_const_start].value[index]
        except:
            input = graph.node_output[backward_analysis_const_start].value
        additional_constraint = meet(input, suspected_node_input)

        S = z3.Solver()
        all_constraints = [constraints, additional_constraint]
        S.add(all_constraints)
        ans = str(S.check())
        if ans == "sat":
            it = S.model()
            for x in it:
                graph.write(str(x) + ": " + str(it[x]))
            print(suspected_node.op, suspected_node.name)
            print("sat")
            cnt_sat += 1
        elif ans == "unknown":
            print(suspected_node.op, suspected_node.name)
            print("unknown")
            cnt_unknown += 1
        else:
            # print("unsat")
            cnt_unsat += 1
        cnt_all += 1
    print("all: ", cnt_all, "sat: ", cnt_sat, "unsat: ", cnt_unsat, "unknown: ", cnt_unknown)
    print(graph.get_info())
