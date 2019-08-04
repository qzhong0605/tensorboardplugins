# The base class for converting other-format graph to TensorBoard-format graph
# network
#
# The details for TensorBoard-format graph is on the file
# `tensorboard/compat/proto/graph.proto`
#
################################################################################

from tensorboard.compat.proto import graph_pb2

class TBGraph(object):
    def __init__(self):
        self._tb_graph = graph_pb2.GraphDef()

    def GetTBGraph(self):
        return self._tb_graph

    def convert_to_nodes(self, op):
        raise NotImplementedError("You must provide 'convert_to_nodes' method")

    def ConvertNet(self):
        raise NotImplementedError("You must provide 'ConvertNet' method")

    def SaveNet(self, *kargs):
        """ kwargs includes file path where to hold the disk network """
        raise NotImplementedError("You must provide 'SaveNet' method")
