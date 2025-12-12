# inspect_internal_coords.py
from pathlib import Path

from omegaconf import OmegaConf

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from gflownet.envs.conformers.conformer import Conformer


def build_env():
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    config_dir = repo_root / "config"

    # 1) Experiment config (env overrides)
    exp_cfg_path = config_dir / "experiments" / "ai4mat23" / "mlp_torchani.yaml"
    exp_cfg = OmegaConf.load(exp_cfg_path)

    # 2) Base conformer env config
    base_env_path = config_dir / "env" / "conformers" / "conformer.yaml"
    base_env_cfg = OmegaConf.load(base_env_path)

    # 3) Merge
    merged_env_cfg = OmegaConf.merge(base_env_cfg, exp_cfg.env)

    print("=== FINAL ENV CONFIG (short) ===")
    print("smiles:", merged_env_cfg.smiles)
    print("reward_beta:", merged_env_cfg.reward_beta)
    print()

    env = Conformer(**OmegaConf.to_container(merged_env_cfg, resolve=True))
    return env


def main():
    env = build_env()
    conf = env.conformer          # RDKitConformer wrapper
    mol = conf.rdk_mol            # underlying RDKit Mol
    rd_conf = mol.GetConformer()

    print("=== Molecule info ===")
    print("SMILES:", env.smiles)
    print("n_atoms:", mol.GetNumAtoms())
    print("n_bonds:", mol.GetNumBonds())
    print("n_dim (torsion DOFs):", env.n_dim)
    print()

    # -------------------------
    # Torsions used by env
    # -------------------------
    torsions = getattr(conf, "freely_rotatable_tas", [])
    print(f"Number of torsions in env: {len(torsions)}")
    for ti, ta in enumerate(torsions):
        a, b, c, d = ta
        sa = mol.GetAtomWithIdx(a).GetSymbol()
        sb = mol.GetAtomWithIdx(b).GetSymbol()
        sc = mol.GetAtomWithIdx(c).GetSymbol()
        sd = mol.GetAtomWithIdx(d).GetSymbol()
        print(f"  T{ti:2d}: ({a:2d}-{b:2d}-{c:2d}-{d:2d}) "
              f"{sa}{a}–{sb}{b}–{sc}{c}–{sd}{d}")
    print()

    # Precompute which bonds are part of any torsion (central bond b–c)
    torsion_bonds = set()
    for (a, b, c, d) in torsions:
        torsion_bonds.add(tuple(sorted((b, c))))

    # -------------------------
    # 1) Bonds with labels
    # -------------------------
    bond_info = []
    for bidx, bond in enumerate(mol.GetBonds()):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ai = mol.GetAtomWithIdx(i)
        aj = mol.GetAtomWithIdx(j)

        sym_i = ai.GetSymbol()
        sym_j = aj.GetSymbol()
        in_ring = bond.IsInRing()
        btype = bond.GetBondType()  # SINGLE, DOUBLE, AROMATIC, etc.

        length = rdMolTransforms.GetBondLength(rd_conf, i, j)

        # Heuristic: candidate flexible bond length?
        #  - single bond
        #  - NOT in ring
        #  - both heavy atoms (no H)
        #  - not part of carbonyl (i.e. not DOUBLE / AROMATIC, already filtered)
        is_single = (btype == Chem.BondType.SINGLE)
        heavy = (sym_i != "H" and sym_j != "H")
        torsion_central = tuple(sorted((i, j))) in torsion_bonds

        flex_len = is_single and (not in_ring) and heavy

        bond_info.append(
            (bidx, i, j, sym_i, sym_j, length, in_ring, btype, torsion_central, flex_len)
        )

    print(f"Found {len(bond_info)} bonds.")
    print("First 20 bonds (index, atoms, length, flags):")
    print("idx  (i,sym)-(j,sym)   d(Å)  in_ring  type       on_torsion  flex_len?")
    for (bidx, i, j, si, sj, d, in_ring, btype, on_t, flex) in bond_info[:20]:
        print(
            f"{bidx:3d}  ({i:2d},{si})-({j:2d},{sj})  {d:6.3f}   "
            f"{str(in_ring):7s}  {str(btype):10s}  "
            f"{str(on_t):10s}  {str(flex):8s}"
        )
    print()

    # -------------------------
    # 2) Angles with labels
    # -------------------------
    angle_info = []
    n_atoms = mol.GetNumAtoms()
    ang_idx = 0
    for j in range(n_atoms):
        center = mol.GetAtomWithIdx(j)
        nbrs = [a.GetIdx() for a in center.GetNeighbors()]
        for a_pos in range(len(nbrs)):
            for b_pos in range(a_pos + 1, len(nbrs)):
                i = nbrs[a_pos]
                k = nbrs[b_pos]
                ai = mol.GetAtomWithIdx(i)
                ak = mol.GetAtomWithIdx(k)

                sym_i = ai.GetSymbol()
                sym_j = center.GetSymbol()
                sym_k = ak.GetSymbol()

                ang = rdMolTransforms.GetAngleDeg(rd_conf, i, j, k)

                in_ring = (
                    ai.IsInRing() and center.IsInRing() and ak.IsInRing()
                )

                # Simple heuristic: candidate flexible angle?
                #  - central atom not in ring
                #  - degree >= 3 (branchy, sp3-like)
                degree = len(nbrs)
                flex_ang = (not center.IsInRing()) and (degree >= 3)

                angle_info.append(
                    (ang_idx, i, j, k, sym_i, sym_j, sym_k, ang, in_ring, degree, flex_ang)
                )
                ang_idx += 1

    print(f"Found {len(angle_info)} bond angles.")
    print("First 20 angles (index, atoms, angle, flags):")
    print("idx   (i,sym)-(j,sym)-(k,sym)   angle(deg)  all_in_ring  deg(j)  flex_ang?")
    for (idx, i, j, k, si, sj, sk, ang, in_ring, degj, flex) in angle_info[:20]:
        print(
            f"{idx:3d}   ({i:2d},{si})-({j:2d},{sj})-({k:2d},{sk})   "
            f"{ang:7.2f}      {str(in_ring):10s}  {degj:3d}    {str(flex):8s}"
        )
    print()

    print("=== Summary ===")
    print(f"Torsion DOFs (env.n_dim):      {env.n_dim}")
    print(f"Candidate bond-length DOFs:    {sum(flex for *_, flex in bond_info)} "
          f"(flex_len=True)")
    print(f"Candidate bond-angle DOFs:     {sum(flex for *_, flex in angle_info)} "
          f"(flex_ang=True)")
    print()
    print("Use the printed indices (bonds / angles with flex_* = True) as a")
    print("first pool of DOFs to add on top of torsions.")


if __name__ == "__main__":
    main()
