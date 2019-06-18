from parse.parse_graph import Graph
import z3
from solver import Range
from utils import OVERFLOW_LIMIT, UNDERFLOW_LIMIT
import math

# rule = ["Log", "Exp", "RealDiv", "Sqrt"]
rule = ["RealDiv"]
if __name__ == "__main__":
    graph = Graph("./real.pbtxt")
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

        if suspected_node.op == "Exp":
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
            for x in all_constraints:
                print(x)
            # print(str(S.check()))
            if str(S.check()) == "sat":
                print(S.model())
                is_sat = True
                break
            if str(S.check()) == "unknown":
                has_unknown = True

        if is_sat:
            print("sat")
        elif has_unknown:
            print("unknown")
        else:
            print("unsat")
