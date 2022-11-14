import torch as th


DATA = th.double
INDEX = th.int32


################################################################################
################################################################################
class RaggedTensor:
    ############################################################################
    def __init__(self, n=None, nnz=None, icrow=None, data=None, *args, **kwargs):

        if icrow is not None:
            if n is not None:
                assert icrow.size == n + 1
            assert icrow.dim() == 1, "Error constructing RaggedTensor: icrow must be a 1-D tensor!"
        elif n is not None:
            icrow = th.full((n+1,),-1, dtype=th.long)

        self.icrow = icrow

        if data is not None:
            if nnz is not None:
                assert data.size == nnz
            assert data.dim() == 1, "Error constructing RaggedTensor: data must be a 1-D tensor!"
        elif nnz is not None:
            data = th.empty((nnz,), *args, **kwargs)

        self.data = data

    ############################################################################
    @property
    def n(self):
        return self.icrow.size(0) - 1

    ############################################################################
    @property
    def nnz(self):
        return self.data.size(0)

    ############################################################################
    def rowlen(self, row=None):

        if row is None or row is Ellipsis:
            return self.icrow[1:] - self.icrow[:-1]
        elif isinstance(row, slice):
            return self.rowlen()[row]

        if row < 0:
            row += self.n

        return self.icrow[row+1] - self.icrow[row]

    ############################################################################
    def __getitem__(self, item=Ellipsis):

        if item is Ellipsis:
            return self.data

        elif isinstance(item, int):
            if item < 0:
                item -= 1
            return self.data[self.icrow[item]:self.icrow[item+1]]

        elif isinstance(item, slice):
            assert item.step == 1 or item.step is None,\
                "Error accessing item in RaggedTensor: "\
                "slicing steps != 1 not supported!"
            if item.start is None and item.stop is None:
                return self.data

            if item.start is not None and item.start < 0:
                start = self.icrow[item.start + self.n]
            elif item.start is not None:
                start = self.icrow[item.start]
            else:
                start = 0

            if item.stop is not None and item.stop < 0:
                stop = self.icrow[item.stop + self.n]
            elif item.stop is not None:
                stop = self.icrow[item.stop]
            else:
                stop = self.data.size(0)

            return self.data[start:stop]

        elif isinstance(item, tuple):
            rest = item[1:]
            data = self[item[0]]
            result = data[rest]
            return result

    ############################################################################
    def __setitem__(self, key, value):
        proxy = self[key]
        proxy[...] = value