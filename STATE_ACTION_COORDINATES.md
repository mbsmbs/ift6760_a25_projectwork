# State Structure, Action Structure, and Coordinate Creation Function

## State Structure

### ContinuousCube Environment (`gflownet/envs/cube.py`)

**State Format:**
- Type: `List[float]`
- Length: `n_dim` (dimensionality of the hyper-cube)
- Value Range: Each dimension value is in the closed interval `[0, 1]`
- Source State: `[-1, -1, ..., -1]` (one `-1` per dimension, `n_dim` total)
- Example: For `n_dim=2`, a state might be `[0.3, 0.7]`

**Key Properties:**
- States represent positions in a hyper-cube
- Source state is abstract (all `-1` values)
- Terminating states have all dimensions in `[0, 1]`
- If any dimension value > `1 - min_incr`, only EOS action is valid

**Location in Code:**
- Defined in `CubeBase.__init__()`: `self.source = [-1 for _ in range(self.n_dim)]`
- State stored in `self.state` (inherited from `GFlowNetEnv`)

### Conformer Environment (`gflownet/envs/conformers/conformer.py`)

**State Format:**
- Type: `List[float]`
- Length: `internal_dim` or `internal_dim + 1`
  - `internal_dim = n_bond_lengths + n_bond_angles + n_torsion_angles`
  - Optional time coordinate `t` may be appended
- Structure:
  ```
  [r_0, ..., r_{M-1},               # Bond lengths (Å) - actual values
   θ_0, ..., θ_{K-1},                # Bond angles (radians) - actual values
   φ_0, ..., φ_{L-1},                # Torsion angles (radians) - actual values
   t]                                 # Optional time coordinate
  ```
  Where:
  - M = `n_bond_lengths`
  - K = `n_bond_angles`
  - L = `n_torsion_angles`

**Key Properties:**
- **State order**: `[bond_lengths, bond_angles, torsion_angles]`
- Bond lengths are actual values in Å (not offsets)
- Bond angles are actual values in radians (not offsets)
- Torsion angles are actual values in radians
- State represents internal coordinates in the order: bond lengths, bond angles, torsions

**Location in Code:**
- State structure documented in `sync_conformer_with_state()` method (lines 304-364)
- Inherits from `ContinuousTorus` which inherits from `CubeBase`

---

## Action Structure

### ContinuousCube Environment (`gflownet/envs/cube.py`)

**Action Format:**
- Type: `Tuple[float, float, ..., float]` (tuple of floats)
- Length: `n_dim + 1`
- Structure:
  ```
  (increment_dim_0, increment_dim_1, ..., increment_dim_{n_dim-1}, special_flag)
  ```
  - First `n_dim` values: absolute increments for each dimension
  - Last value (`special_flag`): 
    - `0`: Normal continuous action
    - `1`: Action from/to source state
    - `np.inf`: EOS (End of Sequence) action

**EOS Action:**
- Represented as: `(np.inf, np.inf, ..., np.inf)` (all `n_dim + 1` values are `np.inf`)
- Defined in `get_action_space()`: `self.eos = tuple([np.inf] * actions_dim)`

**Action Semantics:**
- Forward actions: Increment dimensions by the specified amounts
- Backward actions: Decrement dimensions by the specified amounts
- Actions represent **absolute increments**, not relative
- Relative increments (sampled from Beta distributions) are converted to absolute increments using:
  - Forward: `a = m + r * (1 - x - m)` where `m = min_incr`, `r = relative_increment`, `x = current_dim_value`
  - Backward: `a = m + r * (x - m)`

**Location in Code:**
- Action space defined in `ContinuousCube.get_action_space()` (lines 320-337)
- Action sampling in `_sample_actions_batch_forward()` and `_sample_actions_batch_backward()` (lines 700-897)
- Action execution in `step()` and `step_backwards()` (lines 1206-1278)

### Conformer Environment

**Action Structure:**
- Inherits action structure from `ContinuousTorus` (which inherits from `ContinuousCube`)
- Actions modify internal coordinates (torsion angles, bond lengths, bond angles)
- Same format as `ContinuousCube`: `(increment_0, ..., increment_{n_dim-1}, special_flag)`

---

## Function that Creates Coordinates from State

### Conformer Environment: `statebatch2proxy()`

**Function Signature:**
```python
def statebatch2proxy(self, states: List[List]) -> npt.NDArray:
```

**Location:** `gflownet/envs/conformers/conformer.py`, lines 362-380

**Purpose:**
Converts a batch of states (internal coordinates) into 3D atomic coordinates for use by the proxy/oracle.

**Input:**
- `states`: List of state lists, where each state contains:
  - Torsion angles (radians)
  - Optional bond length offsets (dimensionless)
  - Optional bond angle offsets (dimensionless)
  - Optional time coordinate

**Output:**
- `npt.NDArray` of shape `(n_states, n_atoms, 4)`
  - First column: atomic numbers
  - Last 3 columns: 3D atom positions (x, y, z coordinates in Å)

**Process:**
1. For each state in the batch:
   - Calls `sync_conformer_with_state(st)` to update the RDKit conformer with the state's internal coordinates
   - Extracts atomic numbers via `conf.get_atomic_numbers()`
   - Extracts 3D atom positions via `conf.get_atom_positions()`
   - Concatenates them into a `(n_atoms, 4)` array: `[atomic_number, x, y, z]`

**Key Dependencies:**
- `sync_conformer_with_state()`: Maps internal coordinates (state) to RDKit conformer geometry
- `RDKitConformer.get_atomic_numbers()`: Returns atomic numbers
- `RDKitConformer.get_atom_positions()`: Returns 3D coordinates in Å

**Example Usage:**
```python
states = [[0.5, 1.2, ...], [0.3, 0.8, ...]]  # List of internal coordinate states
coordinates = env.statebatch2proxy(states)
# Returns: array of shape (2, n_atoms, 4) with atomic numbers and 3D positions
```

**Related Functions:**
- `statetorch2proxy()`: Converts torch tensors to proxy format (calls `statebatch2proxy`)
- `statebatch2oracle()`: Alias for `statebatch2proxy` in conformer environment
- `sync_conformer_with_state()`: Core function that maps state to conformer geometry (lines 302-359)

---

## Summary

1. **State Structure:**
   - **ContinuousCube**: List of `n_dim` floats in `[0, 1]`, source is all `-1`
   - **Conformer**: List of internal coordinates in order `[bond_lengths, bond_angles, torsion_angles]`, optionally with time coordinate
     - Bond lengths: actual values in Å
     - Bond angles: actual values in radians
     - Torsion angles: actual values in radians

2. **Action Structure:**
   - **ContinuousCube/Conformer**: Tuple of `n_dim + 1` floats
     - First `n_dim`: absolute increments for each dimension
     - Last value: special flag (0=normal, 1=source, inf=EOS)

3. **Coordinate Creation Function:**
   - **Conformer.statebatch2proxy()**: Converts internal coordinate states to 3D atomic coordinates
   - Input: List of states (internal coordinates in order: bond_lengths, bond_angles, torsion_angles)
   - Output: Array of shape `(n_states, n_atoms, 4)` with atomic numbers and 3D positions

