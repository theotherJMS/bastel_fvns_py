from dataclasses import dataclass
import torch as th


################################################################################
################################################################################
@dataclass
class PrimaryMesh:

    xnode: th.DoubleTensor = None
    itri: th.IntTensor = None
    iquad: th.IntTensor = None
    bdry_marker: th.IntTensor = None
    i_bdry_face: th.IntTensor = None
    iface: th.IntTensor = None

    ############################################################################
    @property
    def nnodes(self):
        return self.xnode.size(-2)
        #-----------------------------------------------------------------------
    ############################################################################
    @property
    def ntris(self):
        return self.itri.size(-2)

    ############################################################################
    @property
    def nquads(self):
        return self.iquad.size(-2)

    ############################################################################
    @property
    def nelems(self):
        return self.ntris + self.nquads


################################################################################
################################################################################
@dataclass
class DualMesh:
    xVertex: th.DoubleTensor = None


################################################################################
################################################################################
def calc_elem_centroids(mesh: PrimaryMesh) -> (th.DoubleTensor, th.DoubleTensor):
    centroids_tri = th.mean(mesh.itri, dim=-1)
    centroids_quad = th.mean(mesh.iquad, dim=-1)
    return centroids_tri, centroids_quad


################################################################################
################################################################################
def calc_face_centroids(mesh: PrimaryMesh):
    pass