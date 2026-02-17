from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re
import numpy as np

from .util import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class AtomRecord:
    chain: str
    resname: str
    resseq: int
    icode: str
    name: str
    element: str
    coord: np.ndarray  # (3,)


@dataclass(frozen=True)
class Pose:
    pose_id: str
    peptide_atoms: list[AtomRecord]
    vina_score: float


@dataclass(frozen=True)
class ResidueID:
    chain: str
    resseq: int
    icode: str = ""


def _guess_element(atom_name: str) -> str:
    s = "".join([c for c in atom_name.strip() if not c.isdigit()]).strip()
    if not s:
        return "C"
    s = s.upper()
    if len(s) >= 2 and s[:2] in {"CL", "BR"}:
        return s[:2]
    return s[0]


def parse_pdb_atoms(pdb_text: str) -> list[AtomRecord]:
    atoms: list[AtomRecord] = []
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = (line[21].strip() or "?")
        resseq = int(line[22:26])
        icode = line[26].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        element = line[76:78].strip().upper()
        if not element:
            element = _guess_element(name)
        atoms.append(
            AtomRecord(
                chain=chain,
                resname=resname,
                resseq=resseq,
                icode=icode,
                name=name,
                element=element,
                coord=np.array([x, y, z], dtype=float),
            )
        )
    return atoms


def load_protein_atoms(protein_pdb: str | Path) -> list[AtomRecord]:
    txt = Path(protein_pdb).read_text(encoding="utf-8", errors="ignore")
    atoms = parse_pdb_atoms(txt)
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records parsed from protein PDB: {protein_pdb}")
    log.debug("Loaded protein atoms: %d", len(atoms))
    return atoms


_END_RE = re.compile(r"^(END|ENDMDL)\b", re.IGNORECASE)
_MODEL_RE = re.compile(r"^MODEL\b", re.IGNORECASE)


def split_peptide_poses(peptide_pdb: str | Path) -> list[str]:
    """Robust pose splitting: END, ENDMDL, and/or MODEL/ENDMDL."""
    lines = Path(peptide_pdb).read_text(encoding="utf-8", errors="ignore").splitlines()
    has_model = any(_MODEL_RE.match(ln.strip()) for ln in lines)
    blocks: list[list[str]] = []
    cur: list[str] = []

    def flush():
        nonlocal cur
        if any(l.startswith(("ATOM", "HETATM")) for l in cur):
            blocks.append(cur)
        cur = []

    if has_model:
        for ln in lines:
            s = ln.strip()
            if _MODEL_RE.match(s):
                flush()
                continue
            if _END_RE.match(s) and s.upper().startswith("ENDMDL"):
                flush()
                continue
            if s.upper() == "END":
                flush()
                continue
            cur.append(ln)
        flush()
    else:
        for ln in lines:
            s = ln.strip()
            if _END_RE.match(s):
                flush()
                continue
            cur.append(ln)
        flush()

    log.debug("Split peptide poses: %d blocks (has_model=%s)", len(blocks), has_model)
    return ["\n".join(b) + "\n" for b in blocks]


def load_scores(scores_txt: str | Path) -> np.ndarray:
    vals = []
    for line in Path(scores_txt).read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError(f"No scores found in {scores_txt}")
    scores = np.array(vals, dtype=float)
    log.debug("Loaded scores: %d", len(scores))
    return scores


def load_poses(peptide_pdb: str | Path, scores_txt: str | Path) -> list[Pose]:
    blocks = split_peptide_poses(peptide_pdb)
    scores = load_scores(scores_txt)
    if len(blocks) != len(scores):
        raise ValueError(
            f"Mismatch: {len(blocks)} peptide pose blocks in {peptide_pdb} "
            f"but {len(scores)} scores in {scores_txt}"
        )
    poses: list[Pose] = []
    for i, (blk, sc) in enumerate(zip(blocks, scores), start=1):
        atoms = parse_pdb_atoms(blk)
        if not atoms:
            raise ValueError(f"Pose block {i} has no atoms.")
        poses.append(Pose(pose_id=f"pose{i:04d}", peptide_atoms=atoms, vina_score=float(sc)))
    log.debug("Loaded poses: %d", len(poses))
    return poses


def coords_from_atoms(atoms: list[AtomRecord]) -> np.ndarray:
    return np.array([a.coord for a in atoms], dtype=float)

def pose_ca_trace_coords(atoms: list[AtomRecord]) -> np.ndarray:
    """
    Extract an ordered Cα trace from a pose.

    Returns:
        (N, 3) array of Cα coordinates in residue order (approx N-terminus -> C-terminus).
    Notes:
        - Uses AtomRecord.name == 'CA'
        - Deduplicates by (chain, resseq, icode) keeping the first seen CA.
        - Sort key: (chain, resseq, icode)
    """
    ca_by_res: dict[tuple[str, int, str], np.ndarray] = {}

    for a in atoms:
        if a.name != "CA":
            continue
        key = (a.chain, a.resseq, a.icode or "")
        if key not in ca_by_res:
            ca_by_res[key] = a.coord

    if not ca_by_res:
        return np.zeros((0, 3), dtype=float)

    keys = sorted(ca_by_res.keys(), key=lambda k: (k[0], k[1], k[2]))
    coords = np.array([ca_by_res[k] for k in keys], dtype=float)
    return coords

def center_of_geometry(coords: np.ndarray) -> np.ndarray:
    return coords.mean(axis=0)


def center_of_mass(coords: np.ndarray, elements: list[str]) -> np.ndarray:
    weights = {
        "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974, "S": 32.06,
        "CL": 35.45, "BR": 79.904, "F": 18.998, "I": 126.904
    }
    w = np.array([weights.get(e.upper().strip(), 12.0) for e in elements], dtype=float)
    wsum = w.sum()
    if wsum <= 0:
        return coords.mean(axis=0)
    return (coords * w[:, None]).sum(axis=0) / wsum


def parse_ppi_file(ppi_path: str | Path) -> set[ResidueID]:
    out: set[ResidueID] = set()
    for line in Path(ppi_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        try:
            i_chain = toks.index("Chain")
            chain = toks[i_chain + 1]
            resseq = int(toks[-1])
            out.add(ResidueID(chain=chain, resseq=resseq, icode=""))
        except Exception as e:
            raise ValueError(f"Cannot parse PPI line: {line!r}") from e
    log.debug("Parsed PPI residues: %d", len(out))
    return out


def protein_residue_inventory(atoms: list[AtomRecord]) -> tuple[set[ResidueID], dict[str, int]]:
    """
    Returns:
      - set of residues present in protein (chain, resseq, icode)
      - dict of chain -> atom count (for reporting)
    """
    residues: set[ResidueID] = set()
    chain_counts: dict[str, int] = {}
    for a in atoms:
        chain_counts[a.chain] = chain_counts.get(a.chain, 0) + 1
        residues.add(ResidueID(chain=a.chain, resseq=a.resseq, icode=a.icode or ""))
    return residues, chain_counts


def validate_ppi_residues_exist(
    ppi: set[ResidueID],
    protein_residues: set[ResidueID],
    max_examples: int = 20,
) -> tuple[bool, dict]:
    """
    Validate that every residue in PPI set exists in protein residue set.

    Returns (ok, report_dict).
    If ok is False, caller should exit.
    """
    ppi_chains = sorted({r.chain for r in ppi})
    prot_chains = sorted({r.chain for r in protein_residues})

    missing = sorted(ppi - protein_residues, key=lambda r: (r.chain, r.resseq, r.icode))
    present = sorted(ppi & protein_residues, key=lambda r: (r.chain, r.resseq, r.icode))

    report = {
        "ppi_total": len(ppi),
        "protein_res_total": len(protein_residues),
        "ppi_chains": ppi_chains,
        "protein_chains": prot_chains,
        "present_count": len(present),
        "missing_count": len(missing),
        "missing_examples": missing[:max_examples],
    }
    ok = (len(missing) == 0)
    return ok, report


