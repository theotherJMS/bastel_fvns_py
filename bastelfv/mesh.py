from dataclasses import dataclass
import torch as th
from bastelfv.util import DATA, INDEX

DEFAULT_N_MAX_CORNERS: INDEX = 4
DEFAULT_N_MAX_ELEMS_PER_NODE: INDEX = 4
DEFAULT_N_MAX_NEIGHB_ELEMS: INDEX = 4
DEFAULT_N_MAX_NEIGHB_NODES: INDEX = 8


################################################################################
################################################################################
@dataclass
class Mesh:
    # Base data
    x_node: th.DoubleTensor = None
    corners: th.IntTensor = None
    n_corners: th.IntTensor = None
    n_max_corners: INDEX = DEFAULT_N_MAX_CORNERS

    # Face data
    i_nodes_per_face: th.IntTensor = None

    # Boundary data
    bdry_id: th.IntTensor = None
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
        return 0 if self.corners is None else self.corners.size(-2)

    ################################################################################
    ################################################################################
    @property
    def n_bdry_faces(self):
        return 0 if self.i_bdry_face is None or self.bdry_id is None else self.i_bdry_face.size(0)


################################################################################
################################################################################
def calc_elem_centroids(mesh: Mesh) -> (th.DoubleTensor, th.DoubleTensor):
    centroids_tri = th.mean(mesh.corners, dim=-1)
    centroids_quad = th.mean(mesh.iquad, dim=-1)
    return centroids_tri, centroids_quad


################################################################################
################################################################################
def calc_n_corners(corners: th.IntTensor):
    return th.count_nonzero(corners >= 0, dim=1).to(INDEX)


################################################################################
################################################################################
def calc_elems_per_node(nnodes, corners, n_corners, n_max_elems_per_node=DEFAULT_N_MAX_ELEMS_PER_NODE,
                        out_i_elems_per_node=None, out_n_elems_per_node=None):
    if out_i_elems_per_node is None:
        out_i_elems_per_node = th.full((nnodes, n_max_elems_per_node), -1, dtype=INDEX)
    if out_n_elems_per_node is None:
        out_n_elems_per_node = th.zeros((nnodes,), dtype=INDEX)

    for ielem in range(corners.size(-2)):
        for icorner in range(n_corners[ielem]):
            inode = corners[ielem, icorner]
            out_i_elems_per_node[inode, out_n_elems_per_node[inode]] = ielem
            out_n_elems_per_node[inode] += 1

    return out_i_elems_per_node, out_n_elems_per_node


################################################################################
################################################################################
def calc_neighb_nodes(nnodes, corners, n_corners, n_max_neighb_nodes=DEFAULT_N_MAX_NEIGHB_NODES,
                      out_i_neighb_nodes=None, out_n_neighb_nodes=None):
    if out_i_neighb_nodes is None:
        out_i_neighb_nodes = th.full((nnodes, n_max_neighb_nodes), -1, dtype=INDEX)
    if out_n_neighb_nodes is None:
        out_n_neighb_nodes = th.zeros(nnodes, dtype=INDEX)

    for ielem in range(corners.size(0)):
        for icorner in range(n_corners[ielem]):
            inode = corners[ielem, icorner]
            iprev = corners[ielem, (icorner - 1) % n_corners[ielem]]
            inext = corners[ielem, (icorner + 1) % n_corners[ielem]]
            if iprev not in out_i_neighb_nodes[inode]:
                out_i_neighb_nodes[inode, out_n_neighb_nodes[inode]] = iprev
                out_n_neighb_nodes[inode] += 1
            if inext not in out_i_neighb_nodes[inode]:
                out_i_neighb_nodes[inode, out_n_neighb_nodes[inode]] = inext
                out_n_neighb_nodes[inode] += 1

    # Temporarily shift -1 entries so they are not sorted to the left.
    mask = out_i_neighb_nodes < 0
    out_i_neighb_nodes[mask] += nnodes + 1
    out_i_neighb_nodes, _ = th.sort(out_i_neighb_nodes, dim=-1)
    out_i_neighb_nodes[mask] -= nnodes + 1

    return out_i_neighb_nodes, out_n_neighb_nodes


################################################################################
################################################################################
def create_neighb_nodes_csr(n_neighb_nodes, i_neighb_nodes, csr_neighb_nodes=None):
    nnodes = n_neighb_nodes.size(0)
    n_neighbs = th.sum(n_neighb_nodes)
    if csr_neighb_nodes is not None:
        icrow = csr_neighb_nodes.crow_indices()
        icol = csr_neighb_nodes.col_indices()
        dummy = csr_neighb_nodes.values()
    else:
        icrow = th.zeros(nnodes + 1, dtype=INDEX)
        icol = th.zeros(n_neighbs, dtype=INDEX)
        dummy = th.ones(n_neighbs, dtype=th.bool)

    icrow[0] = 0
    icrow[1:] = th.cumsum(n_neighb_nodes, 0)

    for inode in range(nnodes):
        icol[icrow[inode]:icrow[inode + 1]] = i_neighb_nodes[inode, :n_neighb_nodes[inode]]

    if csr_neighb_nodes is None:
        csr_neighb_nodes = th.sparse_csr_tensor(icrow, icol, dummy, size=(nnodes, nnodes))

    return csr_neighb_nodes


################################################################################
################################################################################
def calc_neighb_elems(corners, n_corners, i_elems_per_node, n_elems_per_node,
                      n_max_neighb_elems=DEFAULT_N_MAX_NEIGHB_ELEMS, out_i_neighb_elems=None,
                      out_n_neighb_elems=None):
    nelems = corners.size(-2)
    if out_i_neighb_elems is None:
        out_i_neighb_elems = th.full((nelems, n_max_neighb_elems), -1, dtype=INDEX)
    if out_n_neighb_elems is None:
        out_n_neighb_elems = th.zeros(nelems, dtype=INDEX)

    for ielem in range(nelems):
        for icorner in range(n_corners[ielem]):

            inode = corners[ielem, icorner]
            inext = corners[ielem, (icorner + 1) % n_corners[ielem]]
            neighb_candidates = i_elems_per_node[inode, :n_elems_per_node[inode]]

            for ineighb in range(n_elems_per_node[inode]):
                ielem_neighb = neighb_candidates[ineighb]
                if ielem_neighb == ielem:
                    continue
                for icorner_neighb in range(n_corners[ielem_neighb]):
                    if corners[ielem_neighb, icorner_neighb] == inext:
                        out_i_neighb_elems[ielem, out_n_neighb_elems[ielem]] = ielem_neighb
                        out_n_neighb_elems[ielem] += 1
                        break

    return out_i_neighb_elems, out_n_neighb_elems
