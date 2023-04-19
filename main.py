"""
    Based on:
    https://learn.qiskit.org/course/ch-applications/solving-combinatorial-optimization-problems-using-qaoa
"""
import json

from qiskit import QuantumCircuit
from qiskit import Aer, execute, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from scipy.optimize import minimize
from qiskit.visualization import plot_histogram
import networkx as nx
from collections import OrderedDict


class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    def get_graph(self):
        return self.graph

    def get_adjacency_matrix(self):
        return nx.adjacency_matrix(self.graph).todense()

    def get_vertices(self):
        return self.nodes

    def draw_graph(self):
        nx.draw(self.graph, with_labels=True, alpha=0.8, node_size=500)


class DomaticNumberQAOA:
    def __init__(self, graph, k, p):
        self.graph = graph

        self.work_qubit_count = len(graph.nodes) * k
        self.ancilla_qubit_count = 1
        self.total_qubit_count = self.work_qubit_count + self.ancilla_qubit_count

        self.K = k  # K
        self.p = p

    def _prepare_circuit(self):
        qr = QuantumRegister(self.work_qubit_count, 'q')
        anc = QuantumRegister(1, 'ancilla')
        cr = ClassicalRegister(self.work_qubit_count, 'c')

        return QuantumCircuit(qr, anc, cr)

    def create_hadamard_circuit(self):
        qc_h = self._prepare_circuit()
        return self.apply_hadamard(qc_h)

    def apply_hadamard(self, qc_h):
        for i in range(0, self.work_qubit_count):
            qc_h.h(i)

        return qc_h

    def create_problem_hamiltonian(self):
        qc_p = self._prepare_circuit()
        gamma = Parameter("$\\gamma$")

        return self.apply_problem_hamiltonian(qc_p, gamma)

    def apply_problem_hamiltonian(self, qc_p, gamma):
        qc_p = self.vertex_in_one_set(qc_p, gamma)
        qc_p = self.each_vertex_set_dominating(qc_p, gamma)
        qc_p = self.count_dominating_set(qc_p, gamma)

        return qc_p

    def vertex_in_one_set(self, qc_p, gamma):
        """
            Apply check if each vertex is exactly in one DOM set
        """
        if self.K > 2:
            for qubit in range(0, self.work_qubit_count, self.K):
                qc_p.rz(2 * gamma, qubit)

        for i in range(0, self.work_qubit_count, self.K):  # for each vertex
            for j in range(i, i + self.K):  # for each DOM-SET (color)
                for k in range(j + 1, i + self.K):
                    qc_p.rzz(2 * gamma, j, k)
                    qc_p.barrier()

        return qc_p

    def each_vertex_set_dominating(self, qc_p, gamma):
        """
            Apply check if each vertex (color) set is DOM-SET
        """
        ancilla_qubit = self.total_qubit_count - 1
        count_nodes = len(self.graph.nodes)

        for dom_set in range(0, self.K):  # for each DOM-SET (color)
            for vertex in range(0, count_nodes):  # for each vertex

                work_qubits = self._get_neighbors_qubits_by_set(vertex, dom_set)

                for qubit in work_qubits:  # X gate for each DOM-SET (color)
                    qc_p.x(qubit)

                qc_p.mcx(work_qubits, ancilla_qubit)  # C..C NOT, Controlled: each vertex, Target: ancilla

                for i in work_qubits:  # Controlled RZ, Controlled: ancilla, Target: each vertex
                    qc_p.crz(2 * gamma, ancilla_qubit, i)

                qc_p.mcx(work_qubits, ancilla_qubit)  # C..C NOT, Controlled: each vertex, Target: ancilla

                for qubit in work_qubits:  # X gate for each DOM-SET (color)
                    qc_p.x(qubit)
                qc_p.barrier()

        return qc_p

    def count_dominating_set(self, qc_p, gamma):
        """
            Check count of DOM sets
        """
        ancilla_qubit = self.total_qubit_count - 1

        for dom_set in range(0, self.K):  # for each DOM-SET (color)

            # find ALL vertices with color K!!!
            work_qubits = self._get_qubits_by_dominating_set(dom_set)

            for qubit in work_qubits:  # X gate for each DOM-SET (color)
                qc_p.x(qubit)

            qc_p.mcx(work_qubits, ancilla_qubit)  # C..C NOT, Controlled: each vertex, Target: ancilla

            for i in work_qubits:  # Controlled RZ, Controlled: ancilla, Target: each vertex
                qc_p.crz(2 * gamma, ancilla_qubit, i)

            qc_p.mcx(work_qubits, ancilla_qubit)  # C..C NOT, Controlled: each vertex, Target: ancilla

            for qubit in work_qubits:  # X gate for each DOM-SET (color)
                qc_p.x(qubit)
            qc_p.barrier()

        return qc_p

    def _get_neighbors(self, vertex):
        adjacency_matrix = self.graph.get_adjacency_matrix()

        row = adjacency_matrix[vertex]  # row corresponding to vertex v
        neighbors = [vertex]

        for i in range(row.shape[0]):
            for j in range(row.shape[1]):
                if row[i, j] == 1:
                    neighbors.append(j)  # neighbors

        return neighbors

    def _get_neighbors_qubits_by_set(self, vertex, set_number):
        work_qubits = []
        vertices = self._get_neighbors(vertex)

        for qubit in vertices:
            work_qubits.append(set_number + qubit * self.K)

        return work_qubits

    def _get_all_neighbors_qubits_by_set(self, set_number):
        work_qubits = []
        vertices = self.graph.get_vertices()

        for vertex in vertices:
            work_qubits.append(self._get_neighbors_qubits_by_set(vertex, set_number))

        return work_qubits

    def _get_qubits_by_dominating_set(self, dom_set):
        work_qubits = []
        vertices_count = self.work_qubit_count
        for qubit in range(dom_set, vertices_count, self.K):
            work_qubits.append(qubit)

        return work_qubits

    def create_mix_hamiltonian(self):
        qc_m = self._prepare_circuit()
        beta = Parameter("$\\beta$")

        return self.apply_mix_hamiltonian(qc_m, beta)

    def apply_mix_hamiltonian(self, qc_m, beta):
        for i in range(0, self.work_qubit_count):
            qc_m.rx(2 * beta, i)

        return qc_m

    def create_qaoa_circuit_template(self):
        qc = self._prepare_circuit()

        qc = self.apply_hadamard(qc)
        qc = self.apply_problem_hamiltonian(qc, Parameter("$\\gamma$"))
        qc = self.apply_mix_hamiltonian(qc, Parameter("$\\beta$"))

        for qubit in range(self.total_qubit_count-1):
            qc.measure(qubit, qubit)

        return qc

    def C_in_one_set(self, bit_string):
        #  Each Vertex only in one Dominating Set
        weight = 0

        for i in range(0, len(bit_string), self.K):
            bit_vertex = bit_string[i:i + self.K]
            if bit_vertex.count('1') == 1:
                weight += 1

        return weight

    def C_each_set_is_dominating(self, bit_string):
        #  Each Set is Dominating Set

        weight = 0
        count_nodes = len(self.graph.nodes)

        for k in range(0, self.K):  # for each DOM-SET
            if self._is_dominating_set_in_vertices(bit_string, k):
                for vertex in range(0, count_nodes):

                    neighbors_qubits = self._get_neighbors_qubits_by_set(vertex, k)
                    if all(bit_string[i] == '0' for i in neighbors_qubits):
                        break
                weight += 1

        return weight

    def C_count_dominating(self, bit_string):
        # Count(DOM-SETS) = K

        weight = 0

        for i in range(self.K):
            for j in range(i, len(bit_string), self.K):
                if bit_string[j] == '1':
                    weight += 1
                    break

        return weight

    def get_bitstring_weight(self, bit_string):
        """
            Given a bitstring as a solution, this function returns
            the number of edges shared between the two partitions of the graph.
        """
        weight = 0
        weight -= self.C_in_one_set(bit_string)
        weight -= self.C_each_set_is_dominating(bit_string)
        weight -= self.C_count_dominating(bit_string)

        return weight

    def _is_dominating_set_in_vertices(self, bit_string, dom_set):
        dom_set_qubits = self._get_all_neighbors_qubits_by_set(dom_set)
        dom_set_qubits = list(set([item for sublist in dom_set_qubits for item in sublist]))

        for i in dom_set_qubits:
            if bit_string[i] == '1':
                return True

        return False

    def compute_expectation(self, counts):
        """
        Computes expectation value based on measurement results
        Args:
            counts: dict => (key,value)=(bitstring, count)
        Returns:
            avg: float => expectation value
        """

        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            obj = self.get_bitstring_weight(bitstring[::-1])
            avg += obj * count
            sum_count += count

        return avg / sum_count

    def create_qaoa_circuit(self, theta):
        """
        Creates a parametrized QAOA circuit
        Args:
            theta: list of unitary parameters
        Returns:
            qc: qiskit circuit
        """

        p = len(theta) // 2  # number of alternating unitaries

        qc = self._prepare_circuit()

        beta = theta[:p]
        gamma = theta[p:]

        self.apply_hadamard(qc)

        for irep in range(0, p):
            self.apply_problem_hamiltonian(qc, gamma[irep])
            self.apply_mix_hamiltonian(qc, beta[irep])

        for qubit in range(self.total_qubit_count - 1):
            qc.measure(qubit, qubit)

        return qc

    # A function that executes the circuit on the chosen backend
    def get_expectation(self, shots=512):
        """
        Runs parametrized circuit
        Args:
            p: int,
               Number of repetitions of unitaries
               :param shots: number of repetitions
        """

        backend = Aer.get_backend('qasm_simulator')
        backend.shots = shots

        def execute_circuit(theta):
            qc = self.create_qaoa_circuit(theta)
            counts = execute(qc, backend, seed_simulator=10).result().get_counts()

            return self.compute_expectation(counts)

        return execute_circuit


class Testing:
    @staticmethod
    def k_2_n_1_p_1():
        return 2, [0], [], 1

    @staticmethod
    def k_2_n_2_p_1():
        return 2, [0, 1], [], 1

    @staticmethod
    def k_2_n_3_p_1():
        return 2, [0, 1, 2], [], 1

    @staticmethod
    def k_2_n_6_p_1():
        return 2, [0, 1, 2, 3, 4, 5], [], 1

    @staticmethod
    def k_2_n_6_p_5():
        return 2, [0, 1, 2, 3, 4, 5], [], 5

    @staticmethod
    def k_3_n_2_p_1():
        return 3, [0, 1], [], 1

    @staticmethod
    def k_3_n_2_p_5():
        return 3, [0, 1], [], 5

    @staticmethod
    def k_3_n_3_p_1():
        return 3, [0, 1, 2], [], 1

    @staticmethod
    def k_3_n_3_p_7():
        return 3, [0, 1, 2], [], 7

    @staticmethod
    def k_3_n_6_p_7():
        return 3, [0, 1, 2, 3, 4, 5], [], 7

    @staticmethod
    def k_2_n_3_p_1_second():
        return 2, [0, 1, 2], [(0, 1), (1, 2)], 1

    @staticmethod
    def k_2_n_6_p_50_second():
        return 2, [0, 1, 2, 3, 4, 5], [(0, 1), (1, 2), (3, 4), (4, 5)], 50

    @staticmethod
    def k_3_n_3_p_1_total():
        return 3, [0, 1, 2], [(0, 1), (1, 2), (2, 0)], 1


if __name__ == '__main__':

    K, nodes, edges, p = Testing.k_3_n_3_p_1_total()

    # 1. Step - Create Template
    graph = Graph(nodes, edges)

    # 1.1. Step - Apply Hadamard
    dom_number = DomaticNumberQAOA(graph, K, p)
    qc_0 = dom_number.create_hadamard_circuit()

    # 1.2. Step - Create Problem and Mix Hamiltonian
    qc_problem = dom_number.create_problem_hamiltonian()
    qc_mix = dom_number.create_mix_hamiltonian()
    qc_problem.draw(output='mpl')

    # Demonstrate QAOA circuit template
    qc_qaoa = dom_number.create_qaoa_circuit_template()
    qc_qaoa.draw(output='mpl')

    # 2. Step - Calculate expectation
    expectation = dom_number.get_expectation(2048)
    res = minimize(expectation, [1.0, 1.0]*p, method='COBYLA')

    # 3. Step - Analyzing the results
    backend = Aer.get_backend('aer_simulator')
    backend.shots = 512

    qc_res = dom_number.create_qaoa_circuit(res.x)  # res.x == theta
    #qc_res.decompose().decompose().draw(output='mpl')

    counts = execute(qc_res, backend, seed_simulator=10).result().get_counts()

    counts_prob = {k: v / 1024 for k, v in counts.items()}
    sorted_counts = sorted(counts_prob.items(), key=lambda x: x[1], reverse=True)  # .OrderBy(x => x.Value)

    ordered_counts = OrderedDict(sorted_counts[:K**len(nodes)+8])

    json_data = json.dumps(ordered_counts)

    # write JSON string to file
    with open('data0', 'w') as file:
        file.write(str(res.x))

    # write JSON string to file
    with open('data.json', 'w') as file:
        file.write(json_data)

    with open('data2.json', 'w') as file:
        file.write(json.dumps(OrderedDict(sorted_counts[:64])))

    with open('data3.json', 'w') as file:
        file.write(json.dumps(sorted(counts.items(), key=lambda x: x[1], reverse=True)))

    plot_histogram(counts)
    plot_histogram(ordered_counts)

    print("Problem has been solved successfully")