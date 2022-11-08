from dataclasses import dataclass
import torch as th


DEFAULT_N_MAX_CORNERS: th.int = 4
DEFAULT_N_MAX_ELEMS_PER_NODE: th.int = 4
DEFAULT_N_MAX_NEIGHB_ELEMS: th.int = 4
DEFAULT_N_MAX_NEIGHB_NODES: th.int = 4


################################################################################
################################################################################
@dataclass
class Mesh:

    # Base data
    x_node: th.DoubleTensor = None
    i_corners: th.IntTensor = None
    n_corners: th.IntTensor = None
    n_max_corners: th.int = DEFAULT_N_MAX_CORNERS

    # Face data
    i_nodes_per_face: th.IntTensor = None

    # Boundary data
    bdry_marker: th.IntTensor = None
    i_bdry_face: th.IntTensor = None

    # Connectivity data
    n_max_elems_per_node: th.int = DEFAULT_N_MAX_ELEMS_PER_NODE
    i_elems_per_node: th.IntTensor = None
    n_elems_per_node: th.IntTensor = None

    n_max_neighb_elems: th.int = DEFAULT_N_MAX_NEIGHB_ELEMS
    i_neighb_elems: th.IntTensor = None
    n_neighb_elems: th.IntTensor = None

    n_max_neighb_nodes: th.int = DEFAULT_N_MAX_NEIGHB_NODES
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
def calc_n_corners(mesh: Mesh):
    return th.count_nonzero(mesh.i_corners,dim=1)