"""
    Based on:
    https://learn.qiskit.org/course/ch-applications/solving-combinatorial-optimization-problems-using-qaoa
"""


from qiskit import QuantumCircuit
from qiskit import Aer, execute, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import C3XGate
from scipy.optimize import minimize
from qiskit.visualization import plot_histogram
import networkx as nx
import matplotlib.pyplot as plt
from collections import OrderedDict
from qiskit import transpile


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

    def get_nodes_count(self):
        return len(self.nodes)

    def draw_graph(self):
        nx.draw(self.graph, with_labels=True, alpha=0.8, node_size=500)


class DomaticNumberQAOA:
    def __init__(self, graph, num):
        self.graph = graph

        self.work_qubit_count = len(graph.nodes * num)
        self.ancilla_qubit_count = 1
        self.total_qubit_count = self.work_qubit_count + self.ancilla_qubit_count

        self.num = num  # K

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

        return qc_p

    def vertex_in_one_set(self, qc_p, gamma):
        """
            Apply check if each vertex is exactly in one DOM set
        """
        if self.num > 2:
            for qubit in range(0, self.work_qubit_count, self.num):
                qc_p.rz(2 * gamma, qubit)

        for i in range(0, self.work_qubit_count, self.num):  # for each vertex
            for j in range(i, i + self.num):  # for each DOM-SET (color)
                for k in range(j + 1, i + self.num):
                    qc_p.rzz(2 * gamma, j, k)
                    qc_p.barrier()

        return qc_p

    def each_vertex_set_dominating(self, qc_p, gamma):
        """
            Apply check if each vertex (color) set is DOM-SET
        """
        ancilla_qubit = self.total_qubit_count - 1
        count_nodes = self.graph.get_nodes_count()

        for dom_set in range(0, self.num):  # for each DOM-SET (color)
            for vertex in range(0, count_nodes):  # for each vertex

                work_vertices = self._get_dominating_set_vertices(vertex)
                work_qubits = self._get_dominating_set_qubits_by_set(work_vertices, dom_set)

                for qubit in work_qubits:  # X gate for each DOM-SET (color)
                    qc_p.x(qubit)
                qc_p.barrier()

                qc_p.mcx(work_qubits, ancilla_qubit)  # C..C NOT, Controlled: each vertex, Target: ancilla
                qc_p.barrier()

                for i in work_qubits:  # Controlled RZ, Controlled: ancilla, Target: each vertex
                    qc_p.crz(2 * gamma, ancilla_qubit, i)
                qc_p.barrier()

                qc_p.mcx(work_qubits, ancilla_qubit)  # C..C NOT, Controlled: each vertex, Target: ancilla
                qc_p.barrier()

                for qubit in work_qubits:  # X gate for each DOM-SET (color)
                    qc_p.x(qubit)
                qc_p.barrier()

        return qc_p

    def _get_dominating_set_vertices(self, vertex):
        adjacency_matrix = self.graph.get_adjacency_matrix()

        row = adjacency_matrix[vertex]  # row corresponding to vertex v
        work_vertices = [vertex]

        for i in range(row.shape[0]):
            for j in range(row.shape[1]):
                if row[i, j] == 1:
                    work_vertices.append(j)  # neighbors

        return work_vertices

    def _get_dominating_set_qubits_by_set(self, vertices, set_number):
        work_qubits = []

        for qubit in vertices:
            work_qubits.append(set_number + qubit * self.num)

        return work_qubits

    def create_mix_hamiltonian(self):
        qc_m = self._prepare_circuit()
        beta = Parameter("$\\beta$")

        return self.apply_mix_hamiltonian(qc_m, beta)

    def apply_mix_hamiltonian(self, qc_m, beta):
        for i in range(0, self.work_qubit_count):
            qc_m.rx(2 * beta, i)

        return qc_m

    def create_qaoa_circuit_template(self, qc_h, qc_p, qc_m):
        qc = self._prepare_circuit()

        qc = self.apply_hadamard(qc)
        qc = self.apply_problem_hamiltonian(qc, Parameter("$\\gamma$"))
        qc = self.apply_mix_hamiltonian(qc, Parameter("$\\beta$"))

        for qubit in range(self.total_qubit_count-1):
            qc.measure(qubit, qubit)

        return qc

    def get_bitstring_weight(self, bit_string):
        """
            Given a bitstring as a solution, this function returns
            the number of edges shared between the two partitions of the graph.
        """
        weight = 0
        for i in range(0, len(bit_string), self.num):
            bit_vertex = bit_string[i:i+self.num]
            penalty = bit_vertex.count('1') - 1
            if penalty == 0:
                weight = -1
            else:
                return 0

        count_nodes = self.graph.get_nodes_count()

        for dom_set in range(0, self.num):  # for each DOM-SET (color)
            for vertex in range(0, count_nodes):
                work_vertices = self._get_dominating_set_vertices(vertex)
                work_qubits = self._get_dominating_set_qubits_by_set(work_vertices, dom_set)

                is_visited = 0

                for i in work_qubits:
                    is_visited += int(bit_string[i])

                if is_visited == 0:
                    return 0

        return -1

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
            counts = execute(qc, backend, seed_simulator=10, nshots=512).result().get_counts()
               # backend.run(qc,seed_simulator=10, nshots=512).result().get_counts()

            return self.compute_expectation(counts)

        return execute_circuit


if __name__ == '__main__':
    # 0. Step - Prepare Input: Graph, qubit count
    #nodes = [0, 1, 2, 3, 4]
    #edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]

    #G.add_nodes_from([0, 1, 2, 3, 4, 5])
    #G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 0)])

    #nodes = [0, 1, 2, 3]
    #edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2)]

    K = 2

    # 1. Step - Create Template
    graph = Graph(nodes, edges)
    #graph.draw_graph()

    # 1.1. Step - Apply Hadamard
    dom_number = DomaticNumberQAOA(graph, K)
    qc_0 = dom_number.create_hadamard_circuit()
    #qc_0.draw(output='mpl')

    # 1.2. Step - Create Problem Hamiltonian
    qc_problem = dom_number.create_problem_hamiltonian()
    #qc_problem.draw(output='mpl')

    # 1.3. Step - Create Mix Hamiltonian
    qc_mix = dom_number.create_mix_hamiltonian()
    #qc_mix.draw(output='mpl')

    # Demonstrate QAOA circuit template
    qc_qaoa = dom_number.create_qaoa_circuit_template(qc_0, qc_problem, qc_mix)
    qc_qaoa.decompose().draw(output='mpl')

    # 2. Step - Calculate expectation
    expectation = dom_number.get_expectation(2048)
    res = minimize(expectation, [1.0, 1.0], method='COBYLA')

    # 3. Step - Analyzing the results
    backend = Aer.get_backend('aer_simulator')
    backend.shots = 512

    qc_res = dom_number.create_qaoa_circuit(res.x)  # res.x == theta
    #qc_res.decompose().decompose().draw(output='mpl')

    counts = execute(qc_res, backend, seed_simulator=10).result().get_counts()
        #backend.run(qc_res, seed_simulator=10).result().get_counts()

    counts_prob = {k: v / 1024 for k, v in counts.items()}
    sorted_counts = sorted(counts_prob.items(), key=lambda x: x[1], reverse=True)  # .OrderBy(x => x.Value)

    ordered_counts = OrderedDict(sorted_counts[:K**len(nodes)+8])

    plot_histogram(counts)
    plot_histogram(ordered_counts)

    print("Problem has been solved successfully")

{'10': 530, '01': 494}