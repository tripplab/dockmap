import tempfile
import unittest
from pathlib import Path

import numpy as np

from dockmap.cli import _angle_between_vectors_deg, _apply_center_pose_offsets, _build_parser, _kabsch_rigid_transform, _md_threshold_colors, _normalize_pose_layers, _pose_heavy_atom_cog, _rmsd
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


class MdPoseLayerTests(unittest.TestCase):
    def test_pose_layer_md_is_valid_and_ordered_above_scatter(self):
        self.assertEqual(_normalize_pose_layers(["scatter,md"]), ["md", "scatter"])

    def test_pose_heavy_atom_cog_ignores_hydrogens(self):
        atoms = [
            AtomRecord(chain="A", resname="LIG", resseq=1, icode="", name="C1", element="C", coord=np.array([0.0, 0.0, 0.0])),
            AtomRecord(chain="A", resname="LIG", resseq=1, icode="", name="O1", element="O", coord=np.array([2.0, 0.0, 0.0])),
            AtomRecord(chain="A", resname="LIG", resseq=1, icode="", name="H1", element="H", coord=np.array([100.0, 0.0, 0.0])),
        ]
        pose = Pose(pose_id="pose0001", peptide_atoms=atoms, vina_score=-1.0)

        np.testing.assert_allclose(_pose_heavy_atom_cog(pose), np.array([1.0, 0.0, 0.0]))

    def test_angle_between_vectors_is_degrees(self):
        angle = _angle_between_vectors_deg(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        self.assertAlmostEqual(angle, 90.0)

    def test_md_threshold_colors_include_threshold_as_green(self):
        colors = _md_threshold_colors(np.array([15.9, 16.0, 16.1]), 16.0)
        np.testing.assert_array_equal(colors, np.array(["green", "green", "red"]))

    def test_pose_layer_help_is_structured(self):
        help_text = _build_parser().format_help()
        self.assertIn("  scatter  - one marker per pose", help_text)
        self.assertIn("rank:size/<cluster_avg_vina>", help_text)
        self.assertIn("--center-pose [N]", help_text)
        self.assertIn("Use --center-pose with no value to center on pose 1", help_text)
        self.assertIn("dockmap ... --center-pose 3 --pose-layer scatter", help_text)
        self.assertNotIn("rank:size\n", help_text)

    def test_center_pose_parser_defaults_to_first_pose_when_value_omitted(self):
        args = _build_parser().parse_args([
            "--set",
            "s1:target.pdb:poses.pdb:scores.txt",
            "--ppi-file",
            "ppi.txt",
            "--center-pose",
        ])
        self.assertEqual(args.center_pose, 1)

    def test_center_pose_offsets_shift_reference_to_zero_zero(self):
        theta = np.array([0.75, -0.25, 1.25], dtype=float)
        phi = np.array([1.10, 1.20, 1.30], dtype=float)
        centered_theta, centered_phi = _apply_center_pose_offsets(theta, phi, -theta[0], -phi[0])
        self.assertAlmostEqual(centered_theta[0], 0.0)
        self.assertAlmostEqual(centered_phi[0], 0.0)
        np.testing.assert_allclose(centered_phi, np.array([0.0, 0.10, 0.20]))


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
