import unittest
import numpy as np

import pyponca
from pyponca import _pyponca

"""
    Test the API, not that its results are correct !
"""
class TestAPI(unittest.TestCase):
    def setUp(self):
        self.N = 16
        self.pc2d = np.random.uniform(0, 1, (self.N, 2)).astype(np.float64)
        self.pc3d = np.random.uniform(0, 1, (self.N, 3)).astype(np.float64)
        self.pc2f = np.random.uniform(0, 1, (self.N, 2)).astype(np.float32)
        self.pc3f = np.random.uniform(0, 1, (self.N, 3)).astype(np.float32)

        self.pcs = [self.pc2d, self.pc3d, self.pc2f, self.pc3f]

    def test_pointcloud(self):
        """
            Test that we can build a PointCloud and that mangling works properly
        """
        self.assertIsInstance(pyponca.PointCloud(self.pc2d).object, _pyponca.PointCloud2dPN)
        self.assertIsInstance(pyponca.PointCloud(self.pc3d).object, _pyponca.PointCloud3dPN)
        self.assertIsInstance(pyponca.PointCloud(self.pc2f).object, _pyponca.PointCloud2fPN)
        self.assertIsInstance(pyponca.PointCloud(self.pc3f).object, _pyponca.PointCloud3fPN)

        # Test with normal
        self.assertIsInstance(pyponca.PointCloud(self.pc2d, self.pc2d).object, _pyponca.PointCloud2dPN)
        self.assertIsInstance(pyponca.PointCloud(self.pc3d, self.pc3d).object, _pyponca.PointCloud3dPN)
        self.assertIsInstance(pyponca.PointCloud(self.pc2f, self.pc2f).object, _pyponca.PointCloud2fPN)
        self.assertIsInstance(pyponca.PointCloud(self.pc3f, self.pc3f).object, _pyponca.PointCloud3fPN)

        # Test that we can't construct mismatched objects
        # for i in range(len(self.pcs)):
        #     for j in range(len(self.pcs)): 
        #         if i != j:
        #             print(i, j)
        #             with self.assertRaises(RuntimeError):
        #                 pyponca.PointCloud(self.pcs[i], self.pcs[j])

    def test_kdtree(self):
        """
            Test that we can build a KDtree and that mangling works properly
        """
        # Test we can build a kdtree
        self.assertIsInstance(pyponca.KDTree(self.pc2d).object, _pyponca.KdTree2dPN)
        self.assertIsInstance(pyponca.KDTree(self.pc3d).object, _pyponca.KdTree3dPN)
        self.assertIsInstance(pyponca.KDTree(self.pc2f).object, _pyponca.KdTree2fPN)
        self.assertIsInstance(pyponca.KDTree(self.pc3f).object, _pyponca.KdTree3fPN)

    def test_kdtree_query(self):
        """
            Test queries can be performed on kdtrees
        """
        for pc in self.pcs:
            kdtree = pyponca.KDTree(pc)
            kdtree.rangeNeighbors(pc[0], 1)
            kdtree.rangeNeighbors(pc, np.ones((pc.shape[0])))
    
    def test_compute(self):
        """
            Test that we can attach pointclouds and filter to a compute object
        """
        for co in _pyponca.ComputeObjectList:
            for pc in self.pcs:
                for flt in pyponca.Filters:
                    try:
                        ts = np.random.uniform(0, 1, (self.N)).astype(pc.dtype)

                        object = pyponca.__dict__[co]()
                        
                        object.setNeighborFilter(pc, ts, flt)
                        object.attach(pyponca.KDTree(pc))
                        object.attach(pyponca.PointCloud(pc, pc))
                        object.attach(pc)
                    except NotImplementedError:
                        # We let notimplementederror that may come from classes that only support 3d
                        pass
    
    def test_compute_func(self):
        """
            Test for potential memory errors and settings in 
        """
        for co in _pyponca.ComputeObjectList:
            for pc in [self.pc3d, self.pc3f]:
                ts = np.random.uniform(0, 1, (self.N)).astype(pc.dtype)

                cls = pyponca.__dict__[co]   
                def test(data):
                    obj = cls()
                    obj.setNeighborFilter(pc, ts)
                    obj.attach(data)
                    obj.potential()
                
                try:
                    test(pc)
                    test(pyponca.PointCloud(pc))
                    test(pyponca.KDTree(pc))
                except AttributeError:
                    pass


if __name__ == "__main__":
    unittest.main()