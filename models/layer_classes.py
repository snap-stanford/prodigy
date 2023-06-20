'''

Classes to mark different components of the model. Used to tell the code what arguments to pass to a function
and what to do with the output of a function.
The idea is that we could then stack different layer¸s together in a simple list.
(e. g. GAT+GAT+SupernodeAggr+Metagraph+SupernodeToBgGraph+GAT+GAT+SupernodeAggr+Metagraph or any other combination
- to see what order works best)

'''


class BackgroundGNNLayer:
    '''
    Layer for the background graph (original (sub)graphs without supernodes).
    '''
    def forward(self, x, edge_index, edge_attr, supernode_edge_index=None):
        raise NotImplementedError
    #  returns x matrix of the original graph


class SupernodeAggrLayer:
    '''
    Layer for aggregating the supernode embeddings from the background graph.
    '''
    def forward(self, x, supernode_edge_index, supernode_idx, graph_batch):
        raise NotImplementedError
    # return a matrix of supernode embeddings


class SupernodeToBgGraphLayer:
    '''
    Layer for propagating the supernode embeddings back to the background graph.
    '''
    def forward(self, x, new_supernode_x, supernode_edge_index, supernode_idx, graph_batch):
        raise NotImplementedError
    # return updated x


class MetagraphLayer:
    '''
    A metagraph (e. g. GAT) layer or similar.
    '''
    def forward(self, x, edge_index, edge_attr, start_right):
        # forward pass on the metagraph
        raise NotImplementedError
    #  return updated metagraph x

