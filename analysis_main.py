from parse.parse_graph import Graph
import z3
from solver import Range
from utils import OVERFLOW_LIMIT, UNDERFLOW_LIMIT
import math
import sys

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
    print(suspected_nodes)

    for suspected_node in suspected_nodes:
        # graph.draw(graph.backward_slice(suspected_node.name, set()), "real_interested")
        if suspected_node.op == "RealDiv":
            constraints = graph.forward_analysis(graph.node_by_name[graph.graph_backward[suspected_node.name][1][0]],
                                                 suspected_node)
        else:
            constraints = graph.forward_analysis(suspected_node)
        if constraints is None:
            continue

        if suspected_node.op in ["Exp", "Expm1"]:
            suspected_node_input = Range(left=math.log(OVERFLOW_LIMIT), right=None, const_type=0)
            backward_analysis_const_start = graph.node_by_name[graph.graph_backward[suspected_node.name][0][0]]
            additional_constraints_gen = graph.backward_analysis_const(backward_analysis_const_start,
                                                                       suspected_node_input)

        elif suspected_node.op == "RealDiv":
            suspected_node_input_y = Range(left=-UNDERFLOW_LIMIT, right=UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.node_by_name[graph.graph_backward[suspected_node.name][1][0]]
            additional_constraints_gen = graph.backward_analysis_const(backward_analysis_const_start,
                                                                       suspected_node_input_y)

        elif suspected_node.op == "Log":
            suspected_node_input = Range(left=None, right=UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.node_by_name[graph.graph_backward[suspected_node.name][0][0]]
            additional_constraints_gen = graph.backward_analysis_const(backward_analysis_const_start,
                                                                       suspected_node_input)
        elif suspected_node.op == "Sqrt":
            suspected_node_input = Range(left=None, right=-UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.node_by_name[graph.graph_backward[suspected_node.name][0][0]]
            additional_constraints_gen = graph.backward_analysis_const(backward_analysis_const_start,
                                                                       suspected_node_input)
        elif suspected_node.op == "Rsqrt":
            suspected_node_input = Range(left=-UNDERFLOW_LIMIT, right=UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.node_by_name[graph.graph_backward[suspected_node.name][0][0]]
            additional_constraints_gen = graph.backward_analysis_const(backward_analysis_const_start,
                                                                       suspected_node_input)
        elif suspected_node.op == "Log1p":
            suspected_node_input = Range(left=-UNDERFLOW_LIMIT - 1, right=UNDERFLOW_LIMIT - 1, const_type=0)
            backward_analysis_const_start = graph.node_by_name[graph.graph_backward[suspected_node.name][0][0]]
            additional_constraints_gen = graph.backward_analysis_const(backward_analysis_const_start,
                                                                       suspected_node_input)
        elif suspected_node.op == "Reciprocal":
            suspected_node_input_y = Range(left=-UNDERFLOW_LIMIT, right=UNDERFLOW_LIMIT, const_type=0)
            backward_analysis_const_start = graph.node_by_name[graph.graph_backward[suspected_node.name][0][0]]
            additional_constraints_gen = graph.backward_analysis_const(backward_analysis_const_start,
                                                                       suspected_node_input_y)
        else:
            raise NotImplementedError("No rule for ", suspected_node.op)

        is_sat = False
        has_unknown = False

        for additional_constraints in additional_constraints_gen:
            if additional_constraints == False:
                # print("failed")
                continue
            S = z3.Solver()
            all_constraints = [constraints, additional_constraints]
            S.add(all_constraints)
            # for x in all_constraints:
            #     print(x)
            # print(str(S.check()))
            if str(S.check()) == "sat":
                it = S.model()
#                 print(it)
                for x in it:
                    graph.write(str(x) + ": " + str(it[x]))
                is_sat = True
                break
            if str(S.check()) == "unknown":
                has_unknown = True

        if is_sat:
            print(suspected_node.op, suspected_node.name)
            print("sat")
        elif has_unknown:
            print(suspected_node.op, suspected_node.name)
            print("unknown")
#         else:
#             print("unsat")
