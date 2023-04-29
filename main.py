"""
    To create the QAOA for the Domatic Number, the QAOA for the MAX-CUT problem was analyzed.
        https://learn.qiskit.org/course/ch-applications/solving-combinatorial-optimization-problems-using-qaoa
"""
import os

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute, QuantumRegister, ClassicalRegister
from scipy.optimize import minimize
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
import networkx as nx
import numpy as np
import json
from collections import OrderedDict
from datetime import datetime
import random


class Graph:
    """
        Represents a graph data structure and provides methods to work with it.
    """
    def __init__(self, n, e):
        self.nodes = n
        self.edges = e

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    def get_adjacency_matrix(self):
        return np.array(nx.adjacency_matrix(self.graph).todense())

    def draw_graph(self):
        nx.draw(self.graph, with_labels=True, alpha=0.8, node_size=500)
        plt.show()

    def graph_drawing(self, **kwargs):
        figure, axes = plt.subplots(frameon=False)
        axes.axis('off')
        # Turn off tick labels
        req_dpi = 100
        figure.set_size_inches(500 / float(req_dpi), 500 / float(req_dpi))
        self.draw_raw_graph()

        plt.show()

    def draw_raw_graph(self):
        colors = ['cornflowerblue' for node in self.graph.nodes()]
        default_axes = plt.axes(frameon=False)
        random_pos = nx.random_layout(self.graph, seed=42)
        pos = nx.spring_layout(self.graph, pos=random_pos)

        nx.draw_networkx_nodes(self.graph, node_color=colors, node_size=900, pos=pos, alpha=1, ax=default_axes)
        nx.draw_networkx_edges(self.graph, pos=pos, width=8, alpha=1, ax=default_axes, edge_color="tab:gray")
        nx.draw_networkx_labels(self.graph, pos=pos, font_size=22, font_color="w")


class DomaticNumberQAOA:
    """
        Create QAOA circuit, calculate expectation value.
        Solve K-Domatic Number, where K is upper bound for count of dominating sets.
    """
    def __init__(self, graph, k, p):
        self.graph = graph

        self.work_qubit_count = len(graph.nodes) * k
        self.total_qubit_count = self.work_qubit_count + 1  # 1 ancilla qubit

        self.K = k  # Max count of dominating sets
        self.p = p  # Circuit depth

    def __prepare_circuit(self):
        qr = QuantumRegister(self.work_qubit_count, 'q')
        anc = QuantumRegister(1, 'ancilla')
        cr = ClassicalRegister(self.work_qubit_count, 'c')

        return QuantumCircuit(qr, anc, cr)

    def __get_neighbors(self, vertex):
        """
            Returns:
                 Given vertex and all corresponding neighbors in a graph.
        """
        adjacency_matrix = self.graph.get_adjacency_matrix()

        row = adjacency_matrix[vertex]  # row corresponding to vertex v
        neighbors = [vertex]

        for i in range(len(row)):
            if row[i] == 1:
                neighbors.append(i)

        return neighbors

    def __get_neighbors_qubits_by_set_vertex(self, vertex, set_number):
        """
            Returns:
                 Numbers of qubits in circuit for the given vertex and all corresponding neighbors in a graph.
        """
        work_qubits = []
        vertices = self.__get_neighbors(vertex)

        for qubit in vertices:
            work_qubits.append(set_number + qubit * self.K)

        return work_qubits

    def create_hadamard_circuit(self):
        qc_h = self.__prepare_circuit()
        return self.apply_hadamard(qc_h)

    def apply_hadamard(self, qc_h):
        """
            1-st QAOA step - Apply Hadamard for each qubit (except ancilla).
            Returns:
                Circuit with Hadamard gates.
        """
        for i in range(0, self.work_qubit_count):
            qc_h.h(i)

        return qc_h

    def create_problem_hamiltonian(self):
        qc_p = self.__prepare_circuit()

        return self.apply_problem_hamiltonian(qc_p, Parameter("$\\gamma$"))

    def apply_problem_hamiltonian(self, qc_p, gamma):
        """
            2-nd QAOA step - Problem Hamiltonian. Consists of two parts.
                1st part: Check that each vertex is "colored" in only one "color" (belongs to only one set).
                2nd part: Check that each set is dominating.
            Returns:
                Circuit with Problem Hamiltonian gates.
        """
        qc_p = self.vertex_in_one_set(qc_p, gamma)
        qc_p = self.each_vertices_set_dominating(qc_p, gamma)

        return qc_p

    def vertex_in_one_set(self, qc_p, gamma):
        """
            2-nd QAOA step.
            1st part for Problem Hamiltonian.
                Check that each vertex belongs to only one set.
        """
        if self.K > 2:
            for qubit in range(0, self.work_qubit_count):
                qc_p.rz(2 * gamma, qubit)

        for i in range(0, self.work_qubit_count, self.K):  # for each vertex
            for j in range(i, i + self.K):  # for each DOM-SET (color)
                for k in range(j + 1, i + self.K):
                    qc_p.rzz(2 * gamma, j, k)
                    qc_p.barrier()

        return qc_p

    def each_vertices_set_dominating(self, qc_p, gamma):
        """
            2-nd QAOA step.
            2nd part for Problem Hamiltonian.
                Check if each vertex set is DOM-SET.
        """
        ancilla_qubit = self.total_qubit_count - 1
        count_nodes = len(self.graph.nodes)

        for dom_set in range(0, self.K):  # for each DOM-SET (color)
            for vertex in range(0, count_nodes):  # for each vertex

                work_qubits = self.__get_neighbors_qubits_by_set_vertex(vertex, dom_set)

                for qubit in work_qubits:  # X gate for each DOM-SET
                    qc_p.x(qubit)

                qc_p.mcx(work_qubits, ancilla_qubit)  # C..C NOT, Controlled: each vertex, Target: ancilla

                #for i in work_qubits:  # Controlled RZ, Controlled: ancilla, Target: each vertex
                #    qc_p.crz(2 * gamma, ancilla_qubit, i)
                qc_p.rz(2 * gamma, ancilla_qubit)

                qc_p.mcx(work_qubits, ancilla_qubit)  # C..C NOT, Controlled: each vertex, Target: ancilla

                for qubit in work_qubits:  # X gate for each DOM-SET
                    qc_p.x(qubit)
                qc_p.barrier()

        return qc_p

    def create_mix_hamiltonian(self):
        qc_m = self.__prepare_circuit()

        return self.apply_mix_hamiltonian(qc_m, Parameter("$\\beta$"))

    def apply_mix_hamiltonian(self, qc_m, beta):
        """
            3-rd QAOA step - Apply Mix Hamiltonian for each qubit (except ancilla)
            Returns:
                Circuit with Mix Hamiltonian gates.
        """
        for i in range(0, self.work_qubit_count):
            qc_m.rx(2 * beta, i)

        return qc_m

    def create_qaoa_circuit_template(self):
        """
            Apply QAOA steps to solve K-Domatic Number.
            Returns:
                Circuit for K-Domatic Number.
        """
        qc = self.__prepare_circuit()

        qc = self.apply_hadamard(qc)
        qc = self.apply_problem_hamiltonian(qc, Parameter("$\\gamma$"))
        qc = self.apply_mix_hamiltonian(qc, Parameter("$\\beta$"))

        for qubit in range(self.total_qubit_count-1):
            qc.measure(qubit, qubit)

        return qc

    def compute_expectation(self, counts):
        """
            Computes expectation value based on measurement results
            Args:
                Dict (key,value)=(bitstring, count)
            Returns:
                Expectation value
        """
        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            obj = self.get_bitstring_weight(bitstring[::-1])
            avg += obj * count
            sum_count += count

        return avg / sum_count

    def get_bitstring_weight(self, bit_string):
        """
            Given a bitstring as a solution.
            Returns:
                Cost Function value
        """
        weight = 0
        weight -= self.cost_function_vertex_in_one_set(bit_string)
        weight -= self.cost_function_each_set_is_dominating(bit_string)

        return weight

    def cost_function_vertex_in_one_set(self, bit_string):
        """
            1st Cost Function part.
            Returns:
                Number of vertices which belongs to only one DOM-set.
        """
        weight = 0

        for i in range(0, len(bit_string), self.K):
            bit_vertex = bit_string[i:i + self.K]
            if bit_vertex.count('1') == 1:
                weight += 1

        return weight

    def cost_function_each_set_is_dominating(self, bit_string):
        """
            2nd Cost Function part.
            Returns:
                Number of dominating sets.
        """
        weight = 0
        count_nodes = len(self.graph.nodes)

        for k in range(0, self.K):  # for each DOM-SET
            is_dom_set = True

            for vertex in range(0, count_nodes):
                neighbors_qubits = self.__get_neighbors_qubits_by_set_vertex(vertex, k)

                if all(bit_string[i] == '0' for i in neighbors_qubits):
                    is_dom_set = False
                    break

            if is_dom_set:
                weight += 1

        return weight

    def get_expectation(self, shots=512):
        """
            Runs parametrized circuit. Executes the circuit on the chosen backend.
        """

        backend = Aer.get_backend('qasm_simulator')
        backend.shots = shots

        def execute_circuit(theta):
            qc = self.create_qaoa_circuit(theta)
            counts = execute(qc, backend, seed_simulator=10).result().get_counts()

            return self.compute_expectation(counts)

        return execute_circuit

    def create_qaoa_circuit(self, theta):
        """
            Creates a parametrized QAOA circuit
            Args:
                theta: list of unitary parameters
            Returns:
                Circuit for K-Domatic Number
        """

        p = len(theta) // 2  # number of alternating unitaries

        qc = self.__prepare_circuit()

        beta = theta[:p]
        gamma = theta[p:]

        self.apply_hadamard(qc)

        for irep in range(0, p):
            self.apply_problem_hamiltonian(qc, gamma[irep])
            self.apply_mix_hamiltonian(qc, beta[irep])

        for qubit in range(self.total_qubit_count - 1):
            qc.measure(qubit, qubit)

        return qc


class Testing:
    """
        Contains Testing methods for K-Domatic Number and methods for results output and save.
    """
    def __init__(self, test_case_path, k, p):
        self.testing_map = "Testing_results"
        self.test_case_path = os.path.join(self.testing_map, test_case_path)
        self.inner_path = "k" + str(k) + "_p" + str(p)
        self.path = os.path.join(self.testing_map, test_case_path, self.inner_path)

    def __save_params(self, params):
        with open(os.path.join(self.path, "params_beta_gamma.txt"), 'w') as file:
            file.write(str(params))

    def __save_counts(self, counts):
        sorted_counts = self.__get_sorted_counts(counts)

        with open(os.path.join(self.path, "all_data.json"), 'w') as file:
            file.write(json.dumps(OrderedDict(sorted_counts[:2 ** (K * len(nodes))]), indent=4))

        with open(os.path.join(self.path, "first_results.json"), 'w') as file:
            file.write(json.dumps(OrderedDict(sorted_counts[:64]), indent=4))

    def __save_run_time(self, run_time):
        with open(os.path.join(self.path, "run_time.txt"), 'w') as file:
            file.write(str(run_time))

    @staticmethod
    def __get_sorted_counts(counts):
        counts_prob = {k: v / 1024 for k, v in counts.items()}
        return sorted(counts_prob.items(), key=lambda x: x[1], reverse=True)  # OrderBy Value

    def save_results(self, params, counts, run_time):
        if not os.path.exists(self.testing_map):
            os.mkdir(self.testing_map)

        if not os.path.exists(self.test_case_path):
            os.mkdir(self.test_case_path)

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.__save_params(params)
        self.__save_counts(counts)
        self.__save_run_time(run_time)

    def plot_counts(self, counts):
        plot_histogram(counts)

    def plot_sorted_counts(self, counts):
        plot_histogram(OrderedDict(self.__get_sorted_counts(counts)))

    @staticmethod
    def g1_n2(k, p):
        return k, [0, 1], [(0, 1)], p

    @staticmethod
    def g2_n3(k, p):
        return k, [0, 1, 2], [(1, 2), (0, 1)], p

    @staticmethod
    def g3_n3(k, p):
        return k, [0, 1, 2], [(0, 1), (1, 2), (0, 2)], p

    @staticmethod
    def g4_n4(k, p):
        return k, [0, 1, 2, 3], [(0, 1), (1, 3), (3, 2), (2, 0), (0, 3), (1, 2)], p

    @staticmethod
    def g5_n4(k, p):
        return k, [0, 1, 2, 3], [(0, 1), (1, 3), (3, 2), (2, 0), (0, 3)], p

    @staticmethod
    def g6_n4(k, p):
        return k, [0, 1, 2, 3], [(0, 1), (1, 3), (3, 2), (2, 0)], p

    @staticmethod
    def g7_n4(k, p):
        return k, [0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)], p

    @staticmethod
    def g8_n4(k, p):
        return k, [0, 1, 2, 3], [(0, 1), (1, 2), (2, 0), (2, 3)], p

    @staticmethod
    def g9_n4(k, p):
        return k, [0, 1, 2, 3], [(0, 1), (2, 3)], p

    @staticmethod
    def g10_n5(k, p):
        return k, [0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (0, 3), (1, 3), (1, 4), (2, 4)], p

    @staticmethod
    def g11_n5(k, p):
        return k, [0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (0, 3)], p

    @staticmethod
    def g12_n5(k, p):
        return k, [0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 0), (3, 4)], p

    @staticmethod
    def g13_n5(k, p):
        return k, [0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 0), (1, 3), (2, 3), (3, 4)], p

    @staticmethod
    def g14_n6(k, p):
        return k, [0, 1, 2, 3, 4, 5], [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], p

    @staticmethod
    def g15_n6(k, p):
        return k, [0, 1, 2, 3, 4, 5], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)], p

    @staticmethod
    def g16_n6(k, p):
        return k, [0, 1, 2, 3, 4, 5], [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)], p


if __name__ == '__main__':
    start_time = datetime.now()
    """
    with open('first_results.json', 'r') as f:
        data = json.loads(f.read())

    correct_answers = ['011001', '100110', '010110', '101001', '100101', '011010']
    data_top = {key: val for key, val in data.items() if key in correct_answers}
    data_rest = {key: val for key, val in data.items() if key not in correct_answers}

    keys = list(data_top.keys()) + list(data_rest.keys())

    values = list(data_top.values()) + list(data_rest.values())



    colors = ['darkseagreen' if key in correct_answers else 'indianred' for key in keys]

   # plt.bar(keys, values, color=colors )


    fig, ax = plt.subplots()
    ax.bar(keys, values, color=colors)

    ax.set_xticklabels(keys, rotation=70)
    plt.xlabel( 'Answers')
    #ax.xaxis.set_label_coords(1.05, -0.1)

    plt.ylabel ('Probabilities')
    plt.title ('n=3, k=2, p=10, Time=0:00:57')

    plt.show()

    #plot_histogram(data_sorted, sort='value_desc')
    plt.show()
    plot_histogram(data)
    """
    k = 2
    p = 5
    T = Testing("g12_n5", k, p)
    K, nodes, edges, p = T.g12_n5(k, p)

    G = Graph(nodes, edges)

    #G.graph_drawing(filename='Complex_model11.png')

    dom_number = DomaticNumberQAOA(G, K, p)

    """
        # 0. Step - Create Template
        # 0.1. Step - Apply Hadamard
    
        qc_0 = dom_number.create_hadamard_circuit()
    
        # 0.2. Step - Create Problem and Mix Hamiltonian
        qc_problem = dom_number.create_problem_hamiltonian()
        qc_mix = dom_number.create_mix_hamiltonian()
    
        # Demonstrate QAOA circuit template
        qc_qaoa = dom_number.create_qaoa_circuit_template()
        qc_qaoa.draw(output='mpl')
    """

    # 1. Step - Calculate expectation
    expectation = dom_number.get_expectation()
    res = minimize(expectation, [1.0, 1.0]*p, method='COBYLA')

    # 2. Step - Analyzing the results
    backend = Aer.get_backend('aer_simulator')
    backend.shots = 512

    qc_res = dom_number.create_qaoa_circuit(res.x)  # res.x == theta
    counts = execute(qc_res, backend, seed_simulator=10, shots=1024).result().get_counts()

    # Save results
    end_time = datetime.now()
    T.save_results(res.x, counts, end_time-start_time)
    T.plot_counts(counts)
    T.plot_sorted_counts(counts)

    print("The problem results are ready!")
