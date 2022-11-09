import numpy as np
from dataclasses import dataclass
import torch as th
from bastelfv.util import DATA, INDEX

DEFAULT_N_MAX_CORNERS: INDEX = 4
DEFAULT_N_MAX_ELEMS_PER_NODE: INDEX = 4
DEFAULT_N_MAX_NEIGHB_ELEMS: INDEX = 4
DEFAULT_N_MAX_NEIGHB_NODES: INDEX = 4


################################################################################
################################################################################
@dataclass
class Mesh:
    # Base data
    x_node: th.DoubleTensor = None
    i_corners: th.IntTensor = None
    n_corners: th.IntTensor = None
    n_max_corners: INDEX = DEFAULT_N_MAX_CORNERS

    # Face data
    i_nodes_per_face: th.IntTensor = None

    # Boundary data
    bdry_marker: th.IntTensor = None
    i_bdry_face: th.IntTensor = None

    # Connectivity data
    n_max_elems_per_node: INDEX = DEFAULT_N_MAX_ELEMS_PER_NODE
    i_elems_per_node: th.IntTensor = None
    n_elems_per_node: th.IntTensor = None

    n_max_neighb_elems: INDEX = DEFAULT_N_MAX_NEIGHB_ELEMS
    i_neighb_elems: th.IntTensor = None
    n_neighb_elems: th.IntTensor = None

    n_max_neighb_nodes: INDEX = DEFAULT_N_MAX_NEIGHB_NODES
    i_neighb_nodes: th.IntTensor = None
    n_neighb_nodes: th.IntTensor = None

    ############################################################################
    @property
    def nnodes(self):
        return 0 if self.x_node is None else self.x_node.size(-2)

    ############################################################################
    @property
    def nelems(self):
        return 0 if self.i_corners is None else self.i_corners.size(-2)


################################################################################
################################################################################
def calc_elem_centroids(mesh: Mesh) -> (th.DoubleTensor, th.DoubleTensor):
    centroids_tri = th.mean(mesh.i_corners, dim=-1)
    centroids_quad = th.mean(mesh.iquad, dim=-1)
    return centroids_tri, centroids_quad


################################################################################
################################################################################
def calc_face_centroids(mesh: Mesh):
    pass


################################################################################
################################################################################
def calc_n_corners(i_corners: th.IntTensor):
    return th.count_nonzero(i_corners >= 0, dim=1).to(INDEX)


################################################################################
################################################################################
def calc_elems_per_node(nnodes, i_corners, n_corners, n_max_elems_per_node=DEFAULT_N_MAX_ELEMS_PER_NODE):
    nelems = i_corners.size(-2)

    i_elems_per_node = th.full((nnodes, n_max_elems_per_node), -1, dtype=INDEX)
    n_elems_per_node = th.zeros((nnodes,), dtype=INDEX)

    for ielem in range(nelems):
        for icorner in range(n_corners[ielem]):
            inode = i_corners[ielem, icorner]
            i_elems_per_node[inode, n_elems_per_node[inode]] = ielem
            n_elems_per_node[inode] += 1

    return i_elems_per_node, n_elems_per_node
