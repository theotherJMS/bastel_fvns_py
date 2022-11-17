import unittest
from dataclasses import dataclass
import torch as th
import mesh as msh
from torch.testing import assert_close
from bastelfv.util import DATA, INDEX


################################################################################
################################################################################
@dataclass(frozen=True)
class MockAirfoil:
    xNode: th.DoubleTensor = th.tensor([
        [1.0, 0.0],
        [0.7, -0.1],
        [0.4, -0.11],
        [0.2, -0.1],
        [0.0, 0.0],
        [0.2, 0.1],
        [0.4, 0.11],
        [0.7, 0.1],
        [1.0, 0.0],
    ], dtype=DATA)


################################################################################
################################################################################
@dataclass(frozen=True)
class MockQuadMesh:
    x_node: th.DoubleTensor = th.tensor([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0],
        [0.0, 3.0], [1.0, 3.0], [2.0, 3.0], [3.0, 3.0], [4.0, 3.0],
    ], dtype=DATA)

    i_corners: th.IntTensor = th.tensor([
        [0, 1, 6, 5], [1, 2, 7, 6], [2, 3, 8, 7], [3, 4, 9, 8],
        [5, 6, 11, 10], [6, 7, 12, 11], [7, 8, 13, 12], [8, 9, 14, 13],
        [10, 11, 16, 15], [11, 12, 17, 16], [12, 13, 18, 17], [13, 14, 19, 18],
    ], dtype=INDEX)
    n_corners: th.IntTensor = th.tensor([
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    ], dtype=INDEX)

    x_centroid: th.DoubleTensor = th.tensor([
        [0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5],
        [0.5, 1.5], [1.5, 1.5], [2.5, 1.5], [3.5, 1.5],
        [0.5, 2.5], [1.5, 2.5], [2.5, 2.5], [3.5, 2.5],
    ], dtype=DATA)

    elem_area: th.DoubleTensor = th.tensor([
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ], dtype=DATA)

    i_neighb_nodes: th.IntTensor = th.tensor([
        [1, 5, -1, -1],
        [0, 2, 6, -1],
        [1, 3, 7, -1],
        [2, 4, 8, -1],
        [3, 9, -1, -1],

        [0, 6, 10, -1],
        [1, 5, 7, 11],
        [2, 6, 8, 12],
        [3, 7, 9, 13],
        [4, 8, 14, -1],

        [5, 11, 15, -1],
        [6, 10, 12, 16],
        [7, 11, 13, 17],
        [8, 12, 14, 18],
        [9, 13, 19, -1],

        [10, 16, -1, -1],
        [11, 15, 17, -1],
        [12, 16, 18, -1],
        [13, 17, 19, -1],
        [14, 18, -1, -1],
    ], dtype=INDEX)
    n_neighb_nodes: th.IntTensor = th.tensor([
        2, 3, 3, 3, 2,
        3, 4, 4, 4, 3,
        3, 4, 4, 4, 3,
        2, 3, 3, 3, 2,
    ], dtype=INDEX)

    icrow_neighb_nodes: th.Tensor = th.tensor([
        0, 2, 5, 8, 11,
        13, 16, 20, 24, 28,
        31, 34, 38, 42, 46,
        49, 51, 54, 57, 60,
        62,
    ], dtype=INDEX)
    icol_neighb_nodes: th.Tensor = th.tensor([
        1, 5,
        0, 2, 6,
        1, 3, 7,
        2, 4, 8,
        3, 9,

        0, 6, 10,
        1, 5, 7, 11,
        2, 6, 8, 12,
        3, 7, 9, 13,
        4, 8, 14,

        5, 11, 15,
        6, 10, 12, 16,
        7, 11, 13, 17,
        8, 12, 14, 18,
        9, 13, 19,

        10, 16,
        11, 15, 17,
        12, 16, 18,
        13, 17, 19,
        14, 18,
    ], dtype=INDEX)
    csr_neighb_nodes: th.Tensor = th.sparse_csr_tensor(icrow_neighb_nodes, icol_neighb_nodes,
                                                       th.ones(62, dtype=th.bool), (20, 20))

    i_elems_per_node: th.IntTensor = th.tensor([
        [0, -1, -1, -1], [0, 1, -1, -1], [1, 2, -1, -1], [2, 3, -1, -1], [3, -1, -1, -1],
        [0, 4, -1, -1], [0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7], [3, 7, -1, -1],
        [4, 8, -1, -1], [4, 5, 8, 9], [5, 6, 9, 10], [6, 7, 10, 11], [7, 11, -1, -1],
        [8, -1, -1, -1], [8, 9, -1, -1], [9, 10, -1, -1], [10, 11, -1, -1], [11, -1, -1, -1]
    ], dtype=INDEX)
    n_elems_per_node: th.IntTensor = th.tensor([
        1, 2, 2, 2, 1, 2, 4, 4, 4, 2, 2, 4, 4, 4, 2, 1, 2, 2, 2, 1,
    ], dtype=INDEX)

    i_neighb_elems: th.IntTensor = th.tensor([
        [1, 4, -1, -1], [0, 2, 5, -1], [1, 3, 6, -1], [2, 7, -1, -1],
        [0, 5, 8, -1], [1, 4, 6, 9], [2, 5, 7, 10], [3, 6, 11, -1],
        [4, 9, -1, -1], [5, 8, 10, -1], [6, 9, 11, -1], [7, 10, -1, -1],
    ], dtype=INDEX)
    n_neighb_elems: th.IntTensor = th.tensor([
        2, 3, 3, 2, 3, 4, 4, 3, 2, 3, 3, 2,
    ], dtype=INDEX)

    i_bdry_face: th.IntTensor = th.tensor([
        [0, 1], [1, 2], [2, 3], [3, 4],
        [4, 9], [9, 14], [14, 19],
        [19, 18], [18, 17], [17, 16], [16, 15],
        [15, 10], [10, 5], [5, 0]
    ], dtype=INDEX)

    bdry_id: th.IntTensor = th.tensor(
        [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
        dtype=INDEX)

    i_nodes_per_face: th.IntTensor = th.tensor([
        [0, 1],
        [0, 4],
        [1, 2],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 8],
        [5, 6],
        [5, 9],
        [6, 7],
        [6, 10],
        [7, 11],
        [8, 9],
        [9, 10],
        [10, 11],
    ], dtype=INDEX)


################################################################################
################################################################################
@dataclass(frozen=True)
class MockTriMesh:
    x_node: th.DoubleTensor = th.tensor([
        [0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [0.9, 0.0],
        [0.2, 0.2], [0.5, 0.2], [0.8, 0.2], [1.1, 0.2],
        [0.0, 0.4], [0.3, 0.4], [0.6, 0.4], [0.9, 0.4],
        [0.2, 0.6], [0.5, 0.6], [0.8, 0.6], [1.1, 0.6],
    ], dtype=DATA)

    i_corners: th.IntTensor = th.tensor([
        [0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5], [2, 3, 6], [3, 7, 6],
        [4, 9, 8], [4, 5, 9], [5, 10, 9], [5, 6, 10], [6, 11, 10], [6, 7, 11],
        [8, 9, 12], [9, 13, 12], [9, 10, 13], [10, 14, 13], [10, 11, 14], [11, 15, 14],
    ], dtype=INDEX)
    n_corners: th.IntTensor = th.tensor([
        3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3,
    ], dtype=INDEX)

    elem_area: th.DoubleTensor = th.tensor([
        0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
        0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
        0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
    ], dtype=DATA)

    i_neighb_nodes: th.IntTensor = th.tensor([
        [1, 4, -1, -1, -1, -1], [0, 2, 4, 5, -1, -1],
        [1, 3, 5, 6, -1, -1], [2, 6, 7, -1, -1, -1],
        [0, 1, 5, 8, 9, -1], [1, 2, 4, 6, 9, 10],
        [2, 3, 5, 7, 10, 11], [3, 6, 11, -1, -1, -1],
        [4, 9, 12, -1, -1, -1], [4, 5, 8, 10, 12, 13],
        [5, 6, 9, 11, 13, 14], [6, 7, 10, 14, 15, -1],
        [8, 9, 13, -1, -1, -1], [9, 10, 12, 14, -1, -1],
        [10, 11, 13, 15, -1, -1], [11, 14, -1, -1, -1, -1]
    ], dtype=INDEX)
    n_neighb_nodes: th.IntTensor = th.tensor([
        2, 4, 4, 3, 5, 6, 6, 3, 3, 6, 6, 5, 3, 4, 4, 2
    ], dtype=INDEX)
    icrow_neighb_nodes: th.IntTensor = th.tensor([
        0, 2, 6, 10, 13, 18, 24, 30, 33, 36, 42, 48, 53, 56, 60, 64, 66,
    ], dtype=INDEX)
    icol_neighb_nodes: th.IntTensor = th.tensor([
        1, 4, 0, 2, 4, 5,
        1, 3, 5, 6, 2, 6, 7,
        0, 1, 5, 8, 9, 1, 2, 4, 6, 9, 10,
        2, 3, 5, 7, 10, 11, 3, 6, 11,
        4, 9, 12, 4, 5, 8, 10, 12, 13,
        5, 6, 9, 11, 13, 14, 6, 7, 10, 14, 15,
        8, 9, 13, 9, 10, 12, 14,
        10, 11, 13, 15, 11, 14,
    ], dtype=INDEX)
    csr_neighb_nodes: th.Tensor = th.sparse_csr_tensor(
        icrow_neighb_nodes, icol_neighb_nodes, values=th.ones(66, dtype=th.bool), size=(16, 16)
    )

    i_elems_per_node: th.IntTensor = th.tensor([
        [0, -1, -1, -1, -1, -1], [0, 1, 2, -1, -1, -1],
        [2, 3, 4, -1, -1, -1], [4, 5, -1, -1, -1, -1],
        [0, 1, 6, 7, -1, -1], [1, 2, 3, 7, 8, 9],
        [3, 4, 5, 9, 10, 11], [5, 11, -1, -1, -1, -1],
        [6, 12, -1, -1, -1, -1], [6, 7, 8, 12, 13, 14],
        [8, 9, 10, 14, 15, 16], [10, 11, 16, 17, -1, -1],
        [12, 13, -1, -1, -1, -1], [13, 14, 15, -1, -1, -1],
        [15, 16, 17, -1, -1, -1], [17, -1, -1, -1, -1, -1],
    ], dtype=INDEX)
    n_elems_per_node: th.IntTensor = th.tensor([
        1, 3, 3, 2, 4, 6, 6, 2, 2, 6, 6, 4, 2, 3, 3, 1,
    ], dtype=INDEX)

    i_neighb_elems: th.IntTensor = th.tensor([
        [1, -1, -1], [0, 2, 7], [1, 3, -1], [2, 4, 9], [3, 5, -1], [4, 11, -1],
        [7, 14, -1], [1, 6, 8], [7, 9, 14], [3, 8, 10], [9, 11, 16], [5, 10, -1],
        [6, 13, -1], [12, 14, -1], [8, 13, 15], [14, 16, -1], [10, 15, 17], [16, -1, -1]
    ], dtype=INDEX)
    n_neighb_elems: th.IntTensor = th.tensor([
        1, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 2, 2, 2, 3, 2, 3, 1,
    ], dtype=INDEX)

    i_bdry_face: th.IntTensor = th.tensor([
        [0, 1], [1, 2], [2, 3],
        [3, 7], [7, 11], [11, 15],
        [15, 14], [14, 13], [13, 12],
        [12, 8], [8, 4], [4, 0],
    ], dtype=INDEX)
    bdry_id: th.IntTensor = th.tensor([
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
    ], dtype=INDEX)


################################################################################
################################################################################
@dataclass(frozen=True)
class MockMixedMesh:
    x_node: th.DoubleTensor = th.tensor([
        [3.0, 0.0],
        [2.0, 0.0],
        [1.0, -0.5],
        [0.0, 0.0],
        [1.0, 0.5],

        [3.0, -1.0],
        [2.0, -1.0],
        [1.0, -1.5],
        [-1.0, 0.0],
        [1.0, 1.5],
        [2.0, 1.0],
        [3.0, 1.0],

        [2.5, -2.0],
        [1.0, -2.5],
        [-0.5, -1.5],
        [-1.5, 0.0],
        [-0.5, 1.5],
        [1.0, 2.5],
        [2.5, 2.0],
    ])
    i_corners: th.IntTensor = th.tensor([
        [0, 1, 6, 5],
        [1, 2, 7, 6],
        [2, 3, 8, 7],
        [3, 4, 9, 8],
        [4, 1, 10, 9],
        [1, 0, 11, 10],

        [5, 6, 12, -1],
        [6, 13, 12, -1],
        [6, 7, 13, -1],
        [7, 14, 13, -1],
        [7, 8, 14, -1],
        [8, 15, 14, -1],

        [8, 16, 15, -1],
        [8, 9, 16, -1],
        [9, 17, 16, -1],
        [9, 10, 17, -1],
        [10, 18, 17, -1],
        [10, 11, 18, -1],
    ], dtype=INDEX)
    n_corners: th.IntTensor = th.tensor([
        4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    ], dtype=INDEX)

    i_elems_per_node: th.IntTensor = th.tensor([
        [0, 5, -1, -1, -1, -1],
        [0, 1, 4, 5, -1, -1],
        [1, 2, -1, -1, -1, -1],
        [2, 3, -1, -1, -1, -1],
        [3, 4, -1, -1, -1, -1],

        [0, 6, -1, -1, -1, -1],
        [0, 1, 6, 7, 8, -1],
        [1, 2, 8, 9, 10, -1],
        [2, 3, 10, 11, 12, 13],
        [3, 4, 13, 14, 15, -1],
        [4, 5, 15, 16, 17, -1],
        [5, 17, -1, -1, -1, -1],

        [6, 7, -1, -1, -1, -1],
        [7, 8, 9, -1, -1, -1],
        [9, 10, 11, -1, -1, -1],
        [11, 12, -1, -1, -1, -1],
        [12, 13, 14, -1, -1, -1],
        [14, 15, 16, -1, -1, -1],
        [16, 17, -1, -1, -1, -1],
    ], dtype=INDEX)
    n_elems_per_node: th.IntTensor = th.tensor([
        2, 4, 2, 2, 2,
        2, 5, 5, 6, 5, 5, 2,
        2, 3, 3, 2, 3, 3, 2,
    ], dtype=INDEX)

    i_bdry_face: th.IntTensor = th.tensor([
        [18, 17], [17, 16], [16, 15], [15, 14], [14, 13], [13, 12],
        [12, 5], [5, 0], [0, 11], [11, 18],
        [1, 4], [4, 3], [3, 2], [2, 1],
    ], dtype=INDEX)
    bdry_id: th.IntTensor = th.tensor([
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2,
    ], dtype=INDEX)

    i_neighb_nodes: th.Tensor = th.tensor([
        [1, 5, 11, -1, -1, -1],
        [0, 2, 4, 6, 10, -1],
        [1, 3, 7, -1, -1, -1],
        [2, 4, 8, -1, -1, -1],
        [1, 3, 9, -1, -1, -1],

        [0, 6, 12, -1, -1, -1],
        [1, 5, 7, 12, 13, -1],
        [2, 6, 8, 13, 14, -1],
        [3, 7, 9, 14, 15, 16],
        [4, 8, 10, 16, 17, -1],
        [1, 9, 11, 17, 18, -1],
        [0, 10, 18, -1, -1, -1],

        [5, 6, 13, -1, -1, -1],
        [6, 7, 12, 14, -1, -1],
        [7, 8, 13, 15, -1, -1],
        [8, 14, 16, -1, -1, -1],
        [8, 9, 15, 17, -1, -1],
        [9, 10, 16, 18, -1, -1],
        [10, 11, 17, -1, -1, -1]
    ], dtype=INDEX)
    n_neighb_nodes: th.Tensor = th.tensor([
        3, 5, 3, 3, 3,
        3, 5, 5, 6, 5, 5, 3,
        3, 4, 4, 3, 4, 4, 3,
    ], dtype=INDEX)
    csr_neighb_nodes: th.Tensor = th.sparse_csr_tensor(
        crow_indices=th.tensor([
            0, 3, 8, 11, 14,
            17, 20, 25, 30, 36, 41, 46,
            49, 52, 56, 60, 63, 67, 71,
            74,
        ], dtype=INDEX),
        col_indices=th.tensor([
            1, 5, 11,
            0, 2, 4, 6, 10,
            1, 3, 7,
            2, 4, 8,
            1, 3, 9,

            0, 6, 12,
            1, 5, 7, 12, 13,
            2, 6, 8, 13, 14,
            3, 7, 9, 14, 15, 16,
            4, 8, 10, 16, 17,
            1, 9, 11, 17, 18,
            0, 10, 18,

            5, 6, 13,
            6, 7, 12, 14,
            7, 8, 13, 15,
            8, 14, 16,
            8, 9, 15, 17,
            9, 10, 16, 18,
            10, 11, 17,
        ], dtype=INDEX),
        values=th.ones(74, dtype=th.bool), size=(19, 19)
    )


################################################################################
################################################################################
class TestMeshConstructor(unittest.TestCase):

    ############################################################################
    def test_empty_mesh(self):
        mesh = msh.Mesh()
        self.assertEqual(mesh.nnodes, 0)
        self.assertEqual(mesh.nelems, 0)

    ############################################################################
    def test_mixed_mesh(self):
        mesh = msh.Mesh(x_node=MockMixedMesh.x_node, corners=MockMixedMesh.i_corners)
        self.assertEqual(19, mesh.nnodes)
        self.assertEqual(18, mesh.nelems)
        assert_close(mesh.x_node, MockMixedMesh.x_node)
        assert_close(mesh.corners, MockMixedMesh.i_corners)

    ############################################################################
    def test_quadmesh(self):
        mesh = msh.Mesh(x_node=MockQuadMesh.x_node, corners=MockQuadMesh.i_corners)
        self.assertEqual(mesh.nnodes, 20)
        self.assertEqual(mesh.nelems, 12)
        assert_close(mesh.x_node, MockQuadMesh.x_node)
        assert_close(mesh.corners, MockQuadMesh.i_corners)

    ############################################################################
    def test_trimesh(self):
        mesh = msh.Mesh(x_node=MockTriMesh.x_node, corners=MockTriMesh.i_corners)
        self.assertEqual(16, mesh.nnodes)
        self.assertEqual(18, mesh.nelems)
        assert_close(mesh.x_node, MockTriMesh.x_node)
        assert_close(mesh.corners, MockTriMesh.i_corners)


################################################################################
################################################################################
class TestCalcNCorners(unittest.TestCase):

    ############################################################################
    def test_mixed_mesh(self):
        n_corners = msh.calc_n_corners(MockMixedMesh.i_corners)
        assert_close(n_corners, MockMixedMesh.n_corners)

    ############################################################################
    def test_quadmesh(self):
        n_corners = msh.calc_n_corners(MockQuadMesh.i_corners)
        assert_close(n_corners, MockQuadMesh.n_corners)

    ############################################################################
    def test_trimesh(self):
        n_corners = msh.calc_n_corners(MockTriMesh.i_corners)
        assert_close(n_corners, MockTriMesh.n_corners)


################################################################################
################################################################################
class TestCalcElemsPerNode(unittest.TestCase):

    ############################################################################
    def test_mixed_mesh(self):
        i_elems_per_node, n_elems_per_node = \
            msh.calc_elems_per_node(19, MockMixedMesh.i_corners, MockMixedMesh.n_corners, 6)
        assert_close(i_elems_per_node, MockMixedMesh.i_elems_per_node)
        assert_close(n_elems_per_node, MockMixedMesh.n_elems_per_node)

    ############################################################################
    def test_quadmesh(self):
        i_elems_per_node, n_elems_per_node = \
            msh.calc_elems_per_node(20, MockQuadMesh.i_corners, MockQuadMesh.n_corners, 4)
        assert_close(i_elems_per_node, MockQuadMesh.i_elems_per_node)
        assert_close(n_elems_per_node, MockQuadMesh.n_elems_per_node)

    ############################################################################
    def test_trimesh(self):
        i_elems_per_node, n_elems_per_node = \
            msh.calc_elems_per_node(16, MockTriMesh.i_corners, MockTriMesh.n_corners, 6)
        assert_close(i_elems_per_node, MockTriMesh.i_elems_per_node)
        assert_close(n_elems_per_node, MockTriMesh.n_elems_per_node)


################################################################################
################################################################################
class TestCalcNNeighbNodes(unittest.TestCase):

    ############################################################################
    def test_mixed_mesh(self):
        mock = MockMixedMesh()
        i_neighb_nodes, n_neighb_nodes = msh.calc_neighb_nodes(mock.x_node.size(0), mock.i_corners,
                                                               mock.n_corners, 8)
        assert_close(n_neighb_nodes, mock.n_neighb_nodes)
        assert_close(i_neighb_nodes[:, :6], mock.i_neighb_nodes)

    ############################################################################
    def test_quad_mesh(self):
        mock = MockQuadMesh()
        i_neighb_nodes, n_neighb_nodes = msh.calc_neighb_nodes(mock.x_node.size(0), mock.i_corners,
                                                               mock.n_corners, 8)
        assert_close(n_neighb_nodes, mock.n_neighb_nodes)
        assert_close(i_neighb_nodes[:, :4], mock.i_neighb_nodes)

    ############################################################################
    def test_tri_mesh(self):
        mock = MockTriMesh()
        i_neighb_nodes, n_neighb_nodes = msh.calc_neighb_nodes(mock.x_node.size(0), mock.i_corners,
                                                               mock.n_corners, 8)
        assert_close(n_neighb_nodes, mock.n_neighb_nodes)
        assert_close(i_neighb_nodes[:, :6], mock.i_neighb_nodes)


################################################################################
################################################################################
class TestCreateNeighbNodesCSR(unittest.TestCase):

    ############################################################################
    def test_mixedmesh(self):
        mock = MockMixedMesh()
        neighb_nodes_csr = msh.create_neighb_nodes_csr(mock.n_neighb_nodes, mock.i_neighb_nodes)

        assert_close(neighb_nodes_csr, mock.csr_neighb_nodes)

    ############################################################################
    def test_quadmesh(self):
        mock = MockQuadMesh()
        neighb_nodes_csr = msh.create_neighb_nodes_csr(mock.n_neighb_nodes, mock.i_neighb_nodes)
        # v_act = neighb_nodes_csr.values()

        assert_close(neighb_nodes_csr, mock.csr_neighb_nodes)

    ############################################################################
    def test_trimesh(self):
        mock = MockTriMesh()
        neighb_nodes_csr = msh.create_neighb_nodes_csr(mock.n_neighb_nodes, mock.i_neighb_nodes)
        # v_act = neighb_nodes_csr.values()

        assert_close(neighb_nodes_csr, mock.csr_neighb_nodes)


################################################################################
################################################################################
class TestCalcNNeighbElems(unittest.TestCase):

    ############################################################################
    def test_mixed_mesh(self):
        mock = MockMixedMesh()
        i_neighb_elems, n_neighb_elems = msh.calc_neighb_elems(mock.i_corners, mock.n_corners, mock.i_elems_per_node,
                                                               mock.n_elems_per_node, 4)

        assert_close(n_neighb_elems, mock.n_neighb_elems)
        assert_close(i_neighb_elems, mock.i_neighb_elems)

    ############################################################################
    def test_quad_mesh(self):
        mock = MockQuadMesh()
        i_neighb_elems, n_neighb_elems = msh.calc_neighb_elems(mock.i_corners, mock.n_corners, mock.i_elems_per_node,
                                                               mock.n_elems_per_node, 4)
        assert_close(n_neighb_elems, mock.n_neighb_elems)
        assert_close(i_neighb_elems, mock.i_neighb_elems)

    ############################################################################
    def test_tri_mesh(self):
        mock = MockTriMesh()
        i_neighb_elems, n_neighb_elems = msh.calc_neighb_elems(mock.i_corners, mock.n_corners, mock.i_elems_per_node,
                                                               mock.n_elems_per_node, 4)
        assert_close(n_neighb_elems, mock.n_neighb_elems)
        assert_close(i_neighb_elems, mock.i_neighb_elems)


################################################################################
################################################################################
if __name__ == '__main__':
    unittest.main()
