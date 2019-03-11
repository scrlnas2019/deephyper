import networkx as nx
from tensorflow import keras

from deephyper.search.nas.model.space.cell import Cell
from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.node import Node, ConstantNode
from deephyper.search.nas.model.space.op.basic import Connect, Tensor
from deephyper.search.nas.model.space.op.op1d import Concatenate


class Structure:
    graph = None
    struct = None

    def draw_graphviz(self, path):
        with open(path, 'w') as f:
            try:
                nx.nx_agraph.write_dot(self.graph, f)
            except:
                print('Error: can\'t create graphviz file...')

class KerasStructure(Structure):
    def __init__(self, input_shape, output_shape):
        """A KerasSructure represent a search space of networks.

        Arguments:
            input_shape {list(tuple(int))} -- list of shapes of all inputs
            output_shape {tuple(int)} -- shape of output
        """

        self.graph = nx.DiGraph()

        if type(input_shape) is tuple:
            # we have only one input tensor here
            op = Tensor(keras.layers.Input(input_shape, name="input_0"))
            self.input_nodes = [ConstantNode(op=op, name='Input_0')]

        elif type(input_shape) is list and all(map(lambda x: type(x) is tuple, input_shape)):
            # we have a list of input tensors here
            self.input_nodes = list()
            for i in range(len(input_shape)):
                op = Tensor(keras.layers.Input(input_shape[i], name=f"input_{i}"))
                inode = ConstantNode(op=op, name=f'Input_{i}')
                self.input_nodes.append(inode)
        else:
            raise RuntimeError(f"input_shape must be either of type 'tuple' or 'list(tuple)' but is of type '{type(input_shape)}'!")

        self.__output_shape = output_shape
        self.output_node = None

        self.struct = []

        self.map_sh2int = {}

    def __len__(self):
        return len(self.struct)

    def __getitem__(self, sliced):
        return self.struct[sliced]

    @property
    def max_num_ops(self):
        """Returns the maximum number of operations of nodes of blocks of cells of struct.

        Returns:
            int: maximum number of Operations for a VariableNode in the current Structure.
        """

        return max([c.max_num_ops() for c in self.struct] + [0])

    @property
    def num_nodes(self):
        """Returns the number of VariableNodes in the current Structure.

        Returns:
            int: number of VariableNodes in the current Structure.
        """

        return sum([c.num_nodes for c in self.struct] + [0])


    def num_nodes_cell(self, i=None):
        """Returns the number of VariableNodes in Cells.
        Args:
            i (int): index of cell i in self.struct

        Returns:
            list(int): the number of nodes in cell i or the list of number of nodes for each cell.
        """
        if i != None:
            return [self.struct[i].num_nodes()]
        else:
            return [self.struct[i].num_nodes() for i in range(len(self.struct))]

    @property
    def num_cells(self):
        """Returns the number of Cells in the current Structure.

        Returns:
            int: number of Cells in the current Structure.
        """

        return len(self.struct)

    def add_cell_f(self, func, num=1):
        """Add a new cell in the structure.

        Args:
            func (function): a function that return a cell with one argument list of input nodes.
            num (int): number of hidden state with which the new cell can connect or None means all previous hidden states
        """
        possible_inputs = self.input_nodes[:] # it's very important to use a copy of the list
        possible_inputs.extend([c.output for c in self.struct])
        if len(self.struct) > 0:
            if num is None:
                cell = func(possible_inputs[:])
            else:
                cell = func(possible_inputs[len(possible_inputs)-num:])
        else:
            cell = func()

        self.add_cell(cell)

    def add_cell(self, cell):
        self.struct.append(cell)

        # hash
        action_nodes = cell.action_nodes
        for an in action_nodes:
            ops = an.ops
            for o in ops:
                str_hash = str(o)
                if not (str_hash in self.map_sh2int):
                    self.map_sh2int[str_hash] = len(self.map_sh2int)+1


    def set_ops(self, indexes, output_node=None):
        """
        Set the operations for each node of each cell of the structure.

        Args:
            indexes (list): element of list can be float in [0, 1] or int.
            output_node (ConstantNode): the output node of the Structure.
        """
        cursor = 0
        for c in self.struct:
            num_nodes = c.num_nodes
            c.set_ops(indexes[cursor:cursor+num_nodes])
            cursor += num_nodes

            self.graph.add_nodes_from(c.graph.nodes())
            self.graph.add_edges_from(c.graph.edges())

        output_nodes = get_output_nodes(self.graph)
        node = ConstantNode(name='Structure_Output')
        node.set_op(Concatenate(self.graph, node, output_nodes))
        self.output_node = node


    def create_model(self, activation=None):
        """Create the tensors corresponding to the structure.

        Args:
            train (bool): True if the network is built for training, False if the network is built for validation/testing (for example False will deactivate Dropout).

        Returns:
            The output tensor.
        """

        output_tensor = create_tensor_aux(self.graph, self.output_node)
        print('input of output layer shape: ', output_tensor.get_shape())
        print('input of output layer tensor: ', output_tensor)
        print('self.output_node: ', self.output_node)
        output_tensor = keras.layers.Flatten()(output_tensor)
        output_tensor = keras.layers.Dense(self.__output_shape[0], activation=activation)(output_tensor)
        print('output of output layer shape: ', output_tensor.get_shape())
        # input_tensor = self.input_node._tensor
        print('-> Input Nodes: ', self.input_nodes)
        input_tensors = [inode._tensor for inode in self.input_nodes]
        print('-> Input Tensors: ', input_tensors)
        return keras.Model(inputs=input_tensors, outputs=output_tensor)

    def get_hash(self, node_index, index):
        """Get the hash representation of a given operation for this structure.

        Args:
            node_index (int): index of the nodes in the structure.
            index (int,float): index of the operation in the node.

        Returns:
            list(int): the hash of the operation as a list of int.
        """

        cursor = 0
        for c in self.struct:
            for n in c.action_nodes:
                if cursor == node_index:
                    str_hash = str(n.get_op(index))
                    int_hash = self.map_sh2int[str_hash]
                    b = bin(int_hash)[2:]
                    b = '0'*(len(bin(len(self.map_sh2int))[2:]) - len(b)) + b
                    b = [int(e) for e in b]
                    return b
                cursor += 1


def get_output_nodes(graph):
    """
    Args:
        graph: (nx.Digraph)

    Return: the nodes without successors of a DiGraph.
    """
    nodes = list(graph.nodes())
    output_nodes = []
    for n in nodes:
        if len(list(graph.successors(n))) == 0:
            output_nodes.append(n)
    return output_nodes

def create_tensor_aux(g, n, train=None):
    """Recursive function to create the tensors from the graph.

    Args:
        g (nx.DiGraph): a graph
        n (nx.Node): a node
        train (bool): True if the network is built for training, False if the network is built for validation/testing (for example False will deactivate Dropout).

    Return:
        the tensor represented by n.
    """

    if n._tensor != None:
        output_tensor = n._tensor
    else:
        pred = list(g.predecessors(n))
        if len(pred) == 0:
            output_tensor = n.create_tensor(train=train)
        else:
            tensor_list = list()
            for s_i in pred:
                tmp = create_tensor_aux(g, s_i, train=train)
                if type(tmp) is list:
                    tensor_list.extend(tmp)
                else:
                    tensor_list.append(tmp)
            output_tensor = n.create_tensor(tensor_list, train=train)
    return output_tensor
