from FACE.structure.Base import Node


class Sum(Node):
    def __init__(self, weights=None, children=None, cluster_centers=None, cardinality=None):
        Node.__init__(self)

        if weights is None:
            weights = []
        self.weights = weights

        if children is None:
            children = []
        self.children = children

        if cluster_centers is None:
            cluster_centers = []
        self.cluster_centers = cluster_centers

        if cardinality is None:
            cardinality = 0
        self.cardinality = cardinality

    @property
    def parameters(self):
        sorted_children = sorted(self.children, key=lambda c: c.id)
        params = [(n.id, self.weights[i]) for i, n in enumerate(sorted_children)]
        return tuple(params)


def gemv(F, x, y, beta):
    """
    Compute y = beta * y + x
    """
    if beta == 0:
        F.assign(y, x)
    else:
        F.elemwise_add(F.negative(y), x, out=y)
        F.elemwise_add(y, beta, out=y)

