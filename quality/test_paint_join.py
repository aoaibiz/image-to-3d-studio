import unittest

import numpy as np

from quality.paint_join import boundary_runs


class PaintJoinTests(unittest.TestCase):
    def setUp(self):
        h, w = 90, 120
        self.fields = np.full((3, h, w, 3), 225, np.uint8)
        self.labels = np.zeros((h, w), np.int8)
        self.labels[:, w // 2 :] = 2
        self.available = np.ones((3, h, w), bool)
        self.normals = np.zeros((h, w, 3))
        self.normals[:, :, 2] = 1
        self.depth = np.zeros((h, w))
        self.face_ids = np.zeros((h, w), np.int32)
        self.faces = np.arange(6).reshape(2, 3)

    def runs(self):
        return boundary_runs(self.fields, self.labels, self.available, self.depth, self.normals,
                             30, self.face_ids, self.faces)

    def test_cut_motif_vs_continuation(self):
        self.fields[0, 35:55] = (24, 48, 96)
        self.assertTrue(self.runs())
        self.fields[2, 35:55] = (24, 48, 96)
        self.assertFalse(self.runs())

    def test_existing_stripe_edge_is_allowed(self):
        self.fields[0, :, 60:] = (24, 48, 96)
        self.fields[2, :, 60:] = (24, 48, 96)
        self.assertFalse(self.runs())

    def test_occluded_part_and_hard_crease(self):
        self.fields[2] = 40
        self.depth[:, 60:] = 1
        self.assertFalse(self.runs())
        self.depth[:] = 0
        self.normals[:, 60:] = (1, 0, 0)
        self.assertFalse(self.runs())

    def test_incomparable_field_or_minor_tone(self):
        self.fields[2] = 208
        self.assertFalse(self.runs())
        self.fields[2] = 50
        self.available[2, :, :60] = False
        self.assertFalse(self.runs())

    def test_close_parallel_surfaces_do_not_form_a_join(self):
        self.fields[2] = 40
        self.face_ids[:, 60:] = 1
        self.assertFalse(self.runs())
        self.faces[1] = [1, 2, 3]
        self.assertTrue(self.runs())
