"""
    Based on:
    https://learn.qiskit.org/course/ch-applications/solving-combinatorial-optimization-problems-using-qaoa
"""


from qiskit import QuantumCircuit
from qiskit import Aer
from qiskit.circuit import Parameter
from scipy.optimize import minimize
from qiskit.visualization import plot_histogram
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    def get_graph(self):
        return self.graph

    def draw_graph(self):
        nx.draw(self.graph, with_labels=True, alpha=0.8, node_size=500)


class DomaticNumberQAOA:
    def __init__(self, graph, num):
        self.graph = graph
        self.qubit_count = len(graph.nodes * num)
        self.num = num

    def create_hadamard_circuit(self):
        qc_h = QuantumCircuit(self.qubit_count)
        return self.apply_hadamard(qc_h)

    def apply_hadamard(self, qc_h):
        for i in range(0, self.qubit_count):
            qc_h.h(i)

        return qc_h

    def create_problem_hamiltonian(self):
        qc_p = QuantumCircuit(self.qubit_count)
        gamma = Parameter("$\\gamma$")

        return self.apply_problem_hamiltonian(qc_p, gamma)

    def apply_problem_hamiltonian(self, qc_p, gamma):
        qc_p = self.vertex_in_one_set(qc_p, gamma)

        return qc_p

    def vertex_in_one_set(self, qc_p, gamma):
        """
            Apply check if each vertex is exactly in one DOM set
        """
        if self.num > 2:
            for qubit in range(0, self.qubit_count, self.num):
                qc_p.rz(2 * gamma, qubit)

        for i in range(0, self.qubit_count, self.num):  # for each vertex
            for j in range(i, i + self.num):  # for each DOM-SET (color)
                for k in range(j + 1, i + self.num):
                    qc_p.rzz(2 * gamma, j, k)
                    qc_p.barrier()

        return qc_p

    def create_mix_hamiltonian(self):
        qc_m = QuantumCircuit(self.qubit_count)
        beta = Parameter("$\\beta$")

        return self.apply_mix_hamiltonian(qc_m, beta)

    def apply_mix_hamiltonian(self, qc_m, beta):
        for i in range(0, self.qubit_count):
            qc_m.rx(2 * beta, i)

        return qc_m

    def create_qaoa_circuit_template(self, qc_h, qc_p, qc_m):
        qc = QuantumCircuit(self.qubit_count)

        qc.append(qc_h, [i for i in range(0, self.qubit_count)])
        qc.append(qc_p, [i for i in range(0, self.qubit_count)])
        qc.append(qc_m, [i for i in range(0, self.qubit_count)])

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
                weight = 0
                break

        return weight

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
        qc = QuantumCircuit(self.qubit_count)

        beta = theta[:p]
        gamma = theta[p:]

        self.apply_hadamard(qc)

        for irep in range(0, p):
            self.apply_problem_hamiltonian(qc, gamma[irep])
            self.apply_mix_hamiltonian(qc, beta[irep])

        qc.measure_all()

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
            counts = backend.run(qc, seed_simulator=10, nshots=512).result().get_counts()

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

    nodes = [0, 1]
    edges = [(0, 1)]

    # 1. Step - Create Template
    graph = Graph(nodes, edges)
    graph.draw_graph()

    # 1.1. Step - Apply Hadamard
    dom_number = DomaticNumberQAOA(graph, 3)
    qc_0 = dom_number.create_hadamard_circuit()
    qc_0.draw()

    # 1.2. Step - Create Problem Hamiltonian
    qc_problem = dom_number.create_problem_hamiltonian()
    qc_problem.decompose().draw(output='mpl')

    # 1.3. Step - Create Mix Hamiltonian
    qc_mix = dom_number.create_mix_hamiltonian()
    qc_mix.draw()

    # Demonstrate QAOA circuit template
    qc_qaoa = dom_number.create_qaoa_circuit_template(qc_0, qc_problem, qc_mix)
    qc_qaoa.decompose().decompose().draw(output='mpl')

    # 2. Step - Calculate expectation
    expectation = dom_number.get_expectation(2048)
    res = minimize(expectation, [1.0, 1.0], method='COBYLA')

    # 3. Step - Analyzing the results
    backend = Aer.get_backend('aer_simulator')
    backend.shots = 512

    qc_res = dom_number.create_qaoa_circuit(res.x)  # res.x == theta
    qc_res.decompose().decompose().draw(output='mpl')

    counts = backend.run(qc_res, seed_simulator=10).result().get_counts()
    plot_histogram(counts)

    print("Problem has been solved successfully")

