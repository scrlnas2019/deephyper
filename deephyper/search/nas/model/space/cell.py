import networkx as nx

from deephyper.search.nas.model.space.node import Node, ConstantNode
from deephyper.search.nas.model.space.block import Block
from deephyper.search.nas.model.space.op.op1d import Concatenate


class Cell:
    """Create a new Cell object.

    Args:
        inputs (list(Node)): possible inputs of the cell
    """
    num = 0

    def __init__(self, inputs=None):
        Cell.num += 1
        self.num = Cell.num
        self.inputs = inputs if not inputs is None else []
        self.output = None
        self.blocks = []
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(inputs)

    @property
    def num_nodes(self):
        """Number of VariableNodes current cell.

        Return:
            int: the number of VariableNodes of current Cell.
        """
        return sum([b.num_nodes() for b in self.blocks]+[0])

    @property
    def action_nodes(self):
        """Return the list of VariableNodes of current Cell.

        Returns:
            list(VariableNode): list of VariableNodes of current Cell.
        """

        var_nodes = []
        for b in self.blocks:
            var_nodes.extend(b.action_nodes)
        return var_nodes

    def set_outputs(self, node=None):
        """Set output node of the current cell.
            node (Node, optional): Defaults to None will create a Concatenation node for the last axis.
        """

        if node is None:
            stacked_nodes = self.get_blocks_output()

            output_node = ConstantNode(name=f'Cell_{self.num}_Output')
            output_node.set_op(Concatenate(self.graph, output_node, stacked_nodes))
        else:
            output_node = node
        self.output = output_node

    def get_blocks_output(self):
        """Get outputs of all blocks of current cell.

        Returns:
            list(Node): outputs of blocks of the current cell.
        """

        b_outputs = []
        for b in self.blocks:
           b_outputs.extend(b.outputs)
        return b_outputs

    def add_block(self, block):
        """Add a new Block object to the Cell.

        Args:
            block (Block): Block the new block to add to the current Cell.
        """
        if block in self.blocks:
            raise RuntimeError(f"The block has already been added to the current Cell(id={self.num}).")

        self.blocks.append(block)

    def max_num_ops(self):
        """Return the maximum number of operations for a VariableNode in current Cell.

        Returns:
            int: maximum number of operations for a VariableNode in current Cell.
        """
        return max(map(lambda b: b.max_num_ops(), self.blocks))

    def set_ops(self, indexes):
        cursor = 0
        for b in self.blocks:
            num_nodes = b.num_nodes()
            b.set_ops(indexes[cursor:cursor+num_nodes])
            cursor += num_nodes

            self.graph.add_nodes_from(b.graph.nodes())
            self.graph.add_edges_from(b.graph.edges())
