import tempfile
import unittest
from pathlib import Path

import numpy as np

from dockmap.cli import _kabsch_rigid_transform, _rmsd
from dockmap.io import AtomRecord, Pose, parse_pdb_atoms, write_pdb_atoms, write_pdb_poses


class AlignmentMathTests(unittest.TestCase):
    def test_kabsch_rigid_transform_reduces_rmsd(self):
        rng = np.random.default_rng(123)
        reference = rng.normal(size=(60, 3))

        a = rng.normal(size=(3, 3))
        u, _, vt = np.linalg.svd(a)
        q = u @ vt
        if np.linalg.det(q) < 0:
            u[:, -1] *= -1
            q = u @ vt
        shift = np.array([2.0, -1.3, 0.7])

        mobile = reference @ q + shift
        rmsd_before = _rmsd(mobile, reference)

        r, t = _kabsch_rigid_transform(mobile, reference)
        aligned = mobile @ r + t
        rmsd_after = _rmsd(aligned, reference)

        self.assertLess(rmsd_after, 1e-10)
        self.assertLessEqual(rmsd_after, rmsd_before + 1e-12)


class PdbExportTests(unittest.TestCase):
    def test_write_pdb_atoms_and_poses(self):
        atoms = [
            AtomRecord(chain="A", resname="GLY", resseq=1, icode="", name="CA", element="C", coord=np.array([1.0, 2.0, 3.0])),
            AtomRecord(chain="A", resname="GLY", resseq=1, icode="", name="N", element="N", coord=np.array([1.5, 2.5, 3.5])),
        ]
        poses = [Pose(pose_id="pose0001", peptide_atoms=atoms, vina_score=-8.1)]

        with tempfile.TemporaryDirectory() as td:
            protein_path = Path(td) / "protein.pdb"
            poses_path = Path(td) / "poses.pdb"
            write_pdb_atoms(protein_path, atoms)
            write_pdb_poses(poses_path, poses)

            protein_text = protein_path.read_text(encoding="utf-8")
            poses_text = poses_path.read_text(encoding="utf-8")

            self.assertIn("ATOM", protein_text)
            self.assertIn("MODEL", poses_text)
            self.assertIn("ENDMDL", poses_text)
            self.assertIn("END\n", poses_text)

            parsed = parse_pdb_atoms(protein_text)
            self.assertEqual(len(parsed), len(atoms))


if __name__ == "__main__":
    unittest.main()
