from bastelfv import util
import unittest
import torch as th
from torch.testing import assert_close


################################################################################
################################################################################
class TestRaggedTensorRowlen(unittest.TestCase):

    ############################################################################
    def setUp(self) -> None:
        self.dense: th.Tensor = th.tensor([
            [1, 2, -1],
            [3, -1, -1],
            [4, 5, 6],
            [-1, -1, -1],
            [7, -1, -1],
        ], dtype=th.int32)
        self.icrow: th.Tensor = th.tensor([0, 2, 3, 6, 6, 7], dtype=th.long)
        self.icol: th.Tensor = th.tensor([0, 1, 0, 0, 1, 2, 0], dtype=th.int32)
        self.values: th.Tensor = th.tensor([1, 2, 3, 4, 5, 6, 7], dtype=th.int32)
        self.ndense: th.Tensor = th.tensor([2, 1, 3, 0, 1], dtype=th.int64)
        self.t = util.RaggedTensor(icrow=self.icrow, data=self.values)

    ############################################################################
    def test_empty(self):
        assert_close(self.t.rowlen(), self.ndense)
        assert_close(self.t.rowlen(None), self.ndense)
        assert_close(self.t.rowlen(...), self.ndense)

    ############################################################################
    def test_single(self):
        self.assertEqual(2, self.t.rowlen(0))
        self.assertEqual(1, self.t.rowlen(1))
        self.assertEqual(3, self.t.rowlen(2))
        self.assertEqual(0, self.t.rowlen(3))
        self.assertEqual(1, self.t.rowlen(4))

    ############################################################################
    def test_single_negative(self):
        self.assertEqual(2, self.t.rowlen(-5))
        self.assertEqual(1, self.t.rowlen(-4))
        self.assertEqual(3, self.t.rowlen(-3))
        self.assertEqual(0, self.t.rowlen(-2))
        self.assertEqual(1, self.t.rowlen(-1))

    ############################################################################
    def test_slice(self):
        self.assertEqual(2, self.t.rowlen(0))
        self.assertEqual(1, self.t.rowlen(1))
        self.assertEqual(3, self.t.rowlen(2))
        self.assertEqual(0, self.t.rowlen(3))
        self.assertEqual(1, self.t.rowlen(4))


################################################################################
################################################################################
class TestRaggedTensorGetitem(unittest.TestCase):

    ############################################################################
    def setUp(self) -> None:
        self.dense: th.Tensor = th.tensor([
            [1, 2, -1],
            [3, -1, -1],
            [4, 5, 6],
            [-1, -1, -1],
            [7, -1, -1],
        ], dtype=th.int32)
        self.icrow: th.Tensor = th.tensor([0, 2, 3, 6, 6, 7], dtype=th.long)
        self.icol: th.Tensor = th.tensor([0, 1, 0, 0, 1, 2, 0], dtype=th.int32)
        self.values: th.Tensor = th.tensor([1, 2, 3, 4, 5, 6, 7], dtype=th.int32)
        self.ndense: th.Tensor = th.tensor([2, 1, 3, 0, 1], dtype=th.int32)
        self.t = util.RaggedTensor(icrow=self.icrow, data=self.values)

    ############################################################################
    def test_ellipsis(self):
        assert_close(self.t[...], self.values)

    ############################################################################
    def test_int(self):
        assert_close(self.t[0], th.tensor([1, 2], dtype=th.int32))
        assert_close(self.t[1], th.tensor([3], dtype=th.int32))
        assert_close(self.t[2], th.tensor([4, 5, 6], dtype=th.int32))
        assert_close(self.t[3], th.tensor([], dtype=th.int32))
        assert_close(self.t[4], th.tensor([7], dtype=th.int32))

    ############################################################################
    def test_int_reverse(self):
        assert_close(self.t[-1], th.tensor([7], dtype=th.int32))
        assert_close(self.t[-2], th.tensor([], dtype=th.int32))
        assert_close(self.t[-3], th.tensor([4, 5, 6], dtype=th.int32))
        assert_close(self.t[-4], th.tensor([3], dtype=th.int32))
        assert_close(self.t[-5], th.tensor([1, 2], dtype=th.int32))

    ############################################################################
    def test_int_bounds_exceeded(self):
        with self.assertRaises(IndexError):
            self.t[5]
        with self.assertRaises(IndexError):
            self.t[6]
        with self.assertRaises(IndexError):
            self.t[-6]

    ############################################################################
    def test_slice(self):
        assert_close(self.t[0:1], th.tensor([1, 2], dtype=th.int32))
        assert_close(self.t[1:2], th.tensor([3], dtype=th.int32))
        assert_close(self.t[2:3], th.tensor([4, 5, 6], dtype=th.int32))
        assert_close(self.t[3:4], th.tensor([], dtype=th.int32))
        assert_close(self.t[4:5], th.tensor([7], dtype=th.int32))

    ############################################################################
    def test_slice_all(self):
        assert_close(self.t[:], self.values)

    ############################################################################
    def test_slice_multiple(self):
        assert_close(self.t[0:2], th.tensor([1, 2, 3], dtype=th.int32))
        assert_close(self.t[1:3], th.tensor([3, 4, 5, 6], dtype=th.int32))
        assert_close(self.t[2:5], th.tensor([4, 5, 6, 7], dtype=th.int32))

    ############################################################################
    def test_slice_negative(self):
        assert_close(self.t[0:-2], th.tensor([1, 2, 3, 4, 5, 6], dtype=th.int32))
        assert_close(self.t[-4:3], th.tensor([3, 4, 5, 6], dtype=th.int32))
        assert_close(self.t[-3:-1], th.tensor([4, 5, 6], dtype=th.int32))

    ############################################################################
    def test_slice_openstart(self):
        assert_close(self.t[:0], th.tensor([], dtype=th.int32))
        assert_close(self.t[:3], th.tensor([1, 2, 3, 4, 5, 6], dtype=th.int32))
        assert_close(self.t[:5], self.values)

    ############################################################################
    def test_slice_openstop(self):
        assert_close(self.t[0:], th.tensor([1, 2, 3, 4, 5, 6, 7], dtype=th.int32))
        assert_close(self.t[2:], th.tensor([4, 5, 6, 7], dtype=th.int32))
        assert_close(self.t[3:], th.tensor([7], dtype=th.int32))
        assert_close(self.t[-3:], th.tensor([4, 5, 6, 7], dtype=th.int32))

    ############################################################################
    def test_slice_single_negative(self):
        assert_close(self.t[-5:-4], th.tensor([1, 2], dtype=th.int32))
        assert_close(self.t[-4:-3], th.tensor([3], dtype=th.int32))
        assert_close(self.t[-3:-2], th.tensor([4, 5, 6], dtype=th.int32))
        assert_close(self.t[-2:-1], th.tensor([], dtype=th.int32))
        assert_close(self.t[-1:], th.tensor([7], dtype=th.int32))

    ############################################################################
    def test_tuple_all_all(self):
        assert_close(self.t[:, :], self.values)

    ############################################################################
    # TODO: What behavior do we want here? Index error when longer than the shortest line?
    # def test_tuple_all_int(self):
    #     with self.assertRaises(IndexError):
    #         self.t[:, 0]

    ############################################################################
    def test_tuple_all_ellipsis(self):
        assert_close(self.t[:, ...], self.values)

    ############################################################################
    def test_tuple_ellipsis_all(self):
        assert_close(self.t[..., :], self.values)

    ############################################################################
    def test_tuple_int_all(self):
        assert_close(self.t[0, :], th.tensor([1, 2], dtype=th.int32))
        assert_close(self.t[1, :], th.tensor([3], dtype=th.int32))
        assert_close(self.t[2, :], th.tensor([4, 5, 6], dtype=th.int32))
        assert_close(self.t[3, :], th.tensor([], dtype=th.int32))
        assert_close(self.t[4, :], th.tensor([7], dtype=th.int32))

    ############################################################################
    def test_tuple_int_all_negative(self):
        assert_close(self.t[-5, :], th.tensor([1, 2], dtype=th.int32))
        assert_close(self.t[-4, :], th.tensor([3], dtype=th.int32))
        assert_close(self.t[-3, :], th.tensor([4, 5, 6], dtype=th.int32))
        assert_close(self.t[-2, :], th.tensor([], dtype=th.int32))
        assert_close(self.t[-1, :], th.tensor([7], dtype=th.int32))

    ############################################################################
    def test_tuple_int_int(self):
        assert_close(self.t[0, 0], th.tensor(1, dtype=th.int32))
        assert_close(self.t[0, 1], th.tensor(2, dtype=th.int32))
        assert_close(self.t[1, 0], th.tensor(3, dtype=th.int32))
        assert_close(self.t[2, 0], th.tensor(4, dtype=th.int32))
        assert_close(self.t[2, 1], th.tensor(5, dtype=th.int32))
        assert_close(self.t[2, 2], th.tensor(6, dtype=th.int32))
        assert_close(self.t[4, 0], th.tensor(7, dtype=th.int32))

    ############################################################################
    def test_tuple_int_int_bothnegative(self):
        assert_close(self.t[-5, -2], th.tensor(1, dtype=th.int32))
        assert_close(self.t[-5, -1], th.tensor(2, dtype=th.int32))
        assert_close(self.t[-4, -1], th.tensor(3, dtype=th.int32))
        assert_close(self.t[-3, -3], th.tensor(4, dtype=th.int32))
        assert_close(self.t[-3, -2], th.tensor(5, dtype=th.int32))
        assert_close(self.t[-3, -1], th.tensor(6, dtype=th.int32))
        assert_close(self.t[-1, -1], th.tensor(7, dtype=th.int32))

    ############################################################################
    def test_tuple_int_int_firstexceeded(self):
        with self.assertRaises(IndexError):
            self.t[5, 0]
        with self.assertRaises(IndexError):
            self.t[-6, 0]

    ############################################################################
    def test_tuple_int_int_firstnegative(self):
        assert_close(self.t[-5, 0], th.tensor(1, dtype=th.int32))
        assert_close(self.t[-5, 1], th.tensor(2, dtype=th.int32))
        assert_close(self.t[-4, 0], th.tensor(3, dtype=th.int32))
        assert_close(self.t[-3, 0], th.tensor(4, dtype=th.int32))
        assert_close(self.t[-3, 1], th.tensor(5, dtype=th.int32))
        assert_close(self.t[-3, 2], th.tensor(6, dtype=th.int32))
        assert_close(self.t[-1, 0], th.tensor(7, dtype=th.int32))

    ############################################################################
    def test_tuple_int_int_secondexceeded(self):
        with self.assertRaises(IndexError):
            self.t[0, 2]
        with self.assertRaises(IndexError):
            self.t[1, 1]
        with self.assertRaises(IndexError):
            self.t[2, 3]
        with self.assertRaises(IndexError):
            self.t[3, 0]
        with self.assertRaises(IndexError):
            self.t[4, 1]

    ############################################################################
    def test_tuple_int_int_secondexceedednegative(self):
        with self.assertRaises(IndexError):
            self.t[0, -3]
        with self.assertRaises(IndexError):
            self.t[1, -2]
        with self.assertRaises(IndexError):
            self.t[2, -4]
        with self.assertRaises(IndexError):
            self.t[3, -1]
        with self.assertRaises(IndexError):
            self.t[4, -2]

    ############################################################################
    def test_tuple_int_int_secondnegative(self):
        assert_close(self.t[0, -2], th.tensor(1, dtype=th.int32))
        assert_close(self.t[0, -1], th.tensor(2, dtype=th.int32))
        assert_close(self.t[1, -1], th.tensor(3, dtype=th.int32))
        assert_close(self.t[2, -3], th.tensor(4, dtype=th.int32))
        assert_close(self.t[2, -2], th.tensor(5, dtype=th.int32))
        assert_close(self.t[2, -1], th.tensor(6, dtype=th.int32))
        assert_close(self.t[4, -1], th.tensor(7, dtype=th.int32))


################################################################################
################################################################################
class TestRaggedTensorSetitem(unittest.TestCase):

    ############################################################################
    def setUp(self) -> None:
        self.dense: th.Tensor = th.tensor([
            [1, 2, -1],
            [3, -1, -1],
            [4, 5, 6],
            [-1, -1, -1],
            [7, -1, -1],
        ], dtype=th.int32)
        self.icrow: th.Tensor = th.tensor([0, 2, 3, 6, 6, 7], dtype=th.long)
        self.icol: th.Tensor = th.tensor([0, 1, 0, 0, 1, 2, 0], dtype=th.int32)
        self.values: th.Tensor = th.tensor([1, 2, 3, 4, 5, 6, 7], dtype=th.int32)
        self.ndense: th.Tensor = th.tensor([2, 1, 3, 0, 1], dtype=th.int32)
        self.t = util.RaggedTensor(icrow=self.icrow, data=self.values)

    ############################################################################
    def test_all(self):
        self.t[:] = 8
        assert_close(self.t.data, th.tensor([8, 8, 8, 8, 8, 8, 8], dtype=th.int32))

    ############################################################################
    def test_ellipsis(self):
        self.t[...] = 8
        assert_close(self.t.data, th.tensor([8, 8, 8, 8, 8, 8, 8], dtype=th.int32))

    ############################################################################
    def test_int0(self):
        self.t[0] = 8
        assert_close(self.t.data, th.tensor([8, 8, 3, 4, 5, 6, 7], dtype=th.int32))

    ############################################################################
    def test_int1(self):
        self.t[1] = 8
        assert_close(self.t.data, th.tensor([1, 2, 8, 4, 5, 6, 7], dtype=th.int32))

    ############################################################################
    def test_int2(self):
        self.t[2] = 8
        assert_close(self.t.data, th.tensor([1, 2, 3, 8, 8, 8, 7], dtype=th.int32))

    ############################################################################
    def test_int3(self):
        self.t[3] = 8
        assert_close(self.t.data, th.tensor([1, 2, 3, 4, 5, 6, 7], dtype=th.int32))

    ############################################################################
    def test_int4(self):
        self.t[4] = 8
        assert_close(self.t.data, th.tensor([1, 2, 3, 4, 5, 6, 8], dtype=th.int32))


################################################################################
################################################################################
if __name__ == '__main__':
    unittest.main()
