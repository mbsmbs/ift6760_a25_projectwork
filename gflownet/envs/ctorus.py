"""
Classes to represent hyper-torus environments
"""
import itertools
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.distributions import Categorical, MixtureSameFamily, Uniform, VonMises, Normal
from torchtyping import TensorType

from gflownet.envs.htorus import HybridTorus
from gflownet.utils.common import copy, tfloat


class ContinuousTorus(HybridTorus):
    """
    Purely continuous (no discrete actions) hyper-torus environment in which the
    action space consists of the increment Delta theta of the angle at each dimension.
    The trajectory is of fixed length length_traj.

    The states space is the concatenation of the angle (in radians and within [0, 2 *
    pi]) at each dimension and the number of actions.

    Attributes
    ----------
    ndim : int
        Dimensionality of the torus

    length_traj : int
       Fixed length of the trajectory.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Small floor on std for linear (non-torsion) dimensions
        # (in radians / latent units – same scale as existing outputs)
        self.min_sigma = getattr(self, "min_sigma", 1e-3)

    # --------- New -------------
    def _get_torsion_linear_counts(self):
        """
        Helper for mixed torsion / linear policies.

        Returns
        -------
        n_tors : int
            Number of torsion coordinates (Von Mises).
        n_lin : int
            Number of non-torsion coordinates (bond lengths + bond angles),
            modelled with Gaussians.
        """
        # Torsions
        n_tors = getattr(self, "n_torsion", None)
        if n_tors is None:
            n_tors = getattr(self, "n_torsions", None)
        if n_tors is None:
            # Fallback: everything is torsion
            return int(self.n_dim), 0

        # Linear DOFs = bond lengths + bond angles
        n_bond_lengths = getattr(self, "n_bond_lengths", 0)
        n_bond_angles = getattr(self, "n_bond_angles", 0)
        n_lin = int(n_bond_lengths + n_bond_angles)

        # Optional consistency check
        if hasattr(self, "n_dim") and self.n_dim is not None:
            assert n_tors + n_lin == self.n_dim, (
                f"Inconsistent DOF counts: n_dim={self.n_dim}, "
                f"n_tors={n_tors}, n_lin={n_lin}"
            )

        return int(n_tors), n_lin
    # ---------------------------

    def get_action_space(self):
        """
        The action space is continuous, thus not defined as such here.

        The actions are tuples of length n_dim, where the value at position d indicates
        the increment of dimension d.

        EOS is indicated by np.inf for all dimensions.

        This method defines self.eos and the returned action space is simply a
        representative (arbitrary) action with an increment of 0.0 in all dimensions,
        and EOS.
        """
        self.eos = tuple([np.inf] * self.n_dim)
        self.representative_action = tuple([0.0] * self.n_dim)
        return [self.representative_action, self.eos]

    # def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
    #     """
    #     Defines the structure of the output of the policy model, from which an
    #     action is to be determined or sampled, by returning a vector with a fixed
    #     random policy.

    #     For each dimension d of the hyper-torus and component c of the mixture, the
    #     output of the policy should return
    #       1) the weight of the component in the mixture
    #       2) the location of the von Mises distribution to sample the angle increment
    #       3) the log concentration of the von Mises distribution to sample the angle
    #       increment

    #     Therefore, the output of the policy model has dimensionality D x C x 3, where D
    #     is the number of dimensions (self.n_dim) and C is the number of components
    #     (self.n_comp). The first 3 x C entries in the policy output correspond to the
    #     first dimension, and so on.
    #     """
    #     policy_output = np.ones(self.n_dim * self.n_comp * 3)
    #     policy_output[1::3] = params["vonmises_mean"]
    #     policy_output[2::3] = params["vonmises_concentration"]
    #     return policy_output
    def get_policy_output(self, params: dict) -> npt.NDArray:
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy.

        For each dimension d of the hyper-torus and component c of the mixture, the
        output of the policy should return
          1) the weight of the component in the mixture
          2) the location of the von Mises distribution to sample the angle increment
          3) the log concentration of the von Mises distribution to sample the angle
          increment

        Therefore, the output of the policy model has dimensionality D x C x 3, where D
        is the number of dimensions (self.n_dim) and C is the number of components
        (self.n_comp). The first 3 x C entries in the policy output correspond to the
        first dimension, and so on.
        """
        policy_output = np.ones(self.n_dim * self.n_comp * 3, dtype=float)
        policy_output[1::3] = params["vonmises_mean"]
        policy_output[2::3] = params["vonmises_concentration"]
        return policy_output


    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        The action space is continuous, thus the mask is not of invalid actions as
        in discrete environments, but an indicator of "special cases", for example
        states from which only certain actions are possible.

        The "mask" has 2 elements - to match the mask of backward actions - but only
        one is needed for forward actions, thus both elements take the same value,
        according to the following:

        - If done is True, then the mask is True.
        - If the number of actions (state[-1]) is equal to the (fixed) trajectory
          length, then only EOS is valid and the mask is True.
        - Otherwise, any continuous action is valid (except EOS) and the mask is False.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True] * 2
        elif state[-1] >= self.length_traj:
            return [True] * 2
        else:
            return [False] * 2

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        The action is space is continuous, thus the mask is not of invalid actions as
        in discrete environments, but an indicator of "special cases", for example
        states from which only certain actions are possible.

        The "mask" has 2 elements to capture the 2 special in backward actions. The
        possible values of the mask are the following:

        - mask[0]:
            - True, if only the "return-to-source" action is valid.
            - False otherwise.
        - mask[1]:
            - True, if only the EOS action is valid, that is if done is True.
            - False otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [False, True]
        elif state[-1] == 1:
            return [True, False]
        else:
            return [False, False]

    def get_parents(
        self, state: List = None, done: bool = None, action: Tuple[int, float] = None
    ) -> Tuple[List[List], List[Tuple[int, float]]]:
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state, as a list of length n_angles where each element
            is the position at each dimension.

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : int
            Last action performed

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]
        # If source state
        elif state[-1] == 0:
            return [], []
        else:
            for dim, angle in enumerate(action):
                state[int(dim)] = (state[int(dim)] - angle) % (2 * np.pi)
            state[-1] -= 1
            parents = [state]
            return parents, [action]

    def action2representative(self, action: Tuple) -> Tuple:
        """
        Returns the arbirary, representative action in the action space, so that the
        action can be contrasted with the action space and masks.
        """
        return self.representative_action

    # def sample_actions_batch(
    #     self,
    #     policy_outputs: TensorType["n_states", "policy_output_dim"],
    #     mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
    #     states_from: Optional[List] = None,
    #     is_backward: Optional[bool] = False,
    #     sampling_method: Optional[str] = "policy",
    #     temperature_logits: Optional[float] = 1.0,
    #     max_sampling_attempts: Optional[int] = 10,
    # ) -> Tuple[List[Tuple], TensorType["n_states"]]:
    #     """
    #     Samples a batch of actions from a batch of policy outputs. The angle increments
    #     that form the actions are sampled from a mixture of Von Mises distributions.

    #     A distinction between forward and backward actions is made and specified by the
    #     argument is_backward, in order to account for the following special cases:

    #     Forward:

    #     - If the number of steps is equal to the maximum, then the only valid action is
    #       EOS.

    #     Backward:

    #     - If the number of steps is equal to 1, then the only valid action is to return
    #       to the source. The specific action depends on the current state.

    #     Args
    #     ----
    #     policy_outputs : tensor
    #         The output of the GFlowNet policy model.

    #     mask : tensor
    #         The mask containing information about special cases.

    #     states_from : tensor
    #         The states originating the actions, in GFlowNet format.

    #     is_backward : bool
    #         True if the actions are backward, False if the actions are forward
    #         (default).
    #     """
    #     device = policy_outputs.device
    #     do_sample = torch.all(~mask, dim=1)
    #     n_states = policy_outputs.shape[0]
    #     logprobs = torch.zeros(
    #         (n_states, self.n_dim), dtype=self.float, device=self.device
    #     )
    #     # Initialize actions tensor with EOS actions (inf) since these will be the
    #     # actions for several special cases in both forward and backward actions.
    #     actions_tensor = torch.full(
    #         (n_states, self.n_dim), torch.inf, dtype=self.float, device=device
    #     )
    #     # Sample angle increments
    #     if torch.any(do_sample):
    #         if sampling_method == "uniform":
    #             distr_angles = Uniform(
    #                 torch.zeros(len(ns_range_noeos)),
    #                 2 * torch.pi * torch.ones(len(ns_range_noeos)),
    #             )
    #         elif sampling_method == "policy":
    #             mix_logits = policy_outputs[do_sample, 0::3].reshape(
    #                 -1, self.n_dim, self.n_comp
    #             )
    #             mix = Categorical(logits=mix_logits)
    #             locations = policy_outputs[do_sample, 1::3].reshape(
    #                 -1, self.n_dim, self.n_comp
    #             )
    #             concentrations = policy_outputs[do_sample, 2::3].reshape(
    #                 -1, self.n_dim, self.n_comp
    #             )
    #             vonmises = VonMises(
    #                 locations,
    #                 torch.exp(concentrations) + self.vonmises_min_concentration,
    #             )
    #             distr_angles = MixtureSameFamily(mix, vonmises)
    #         angles_sampled = distr_angles.sample()
    #         actions_tensor[do_sample] = angles_sampled
    #         logprobs[do_sample] = distr_angles.log_prob(angles_sampled)
    #     logprobs = torch.sum(logprobs, axis=1)
    #     # Catch special case for backwards backt-to-source (BTS) actions
    #     if is_backward:
    #         do_bts = mask[:, 0]
    #         if torch.any(do_bts):
    #             source_angles = tfloat(
    #                 self.source[: self.n_dim], float_type=self.float, device=self.device
    #             )
    #             states_from_angles = tfloat(
    #                 states_from, float_type=self.float, device=self.device
    #             )[do_bts, : self.n_dim]
    #             actions_bts = states_from_angles - source_angles
    #             actions_tensor[do_bts] = actions_bts
    #     # TODO: is this too inefficient because of the multiple data transfers?
    #     actions = [tuple(a.tolist()) for a in actions_tensor]
    #     return actions, logprobs
    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
        states_from: Optional[List] = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.

        Mixed policy:
          - torsion coordinates      -> Von Mises mixture
          - bond lengths / angles    -> Gaussian mixture

        A distinction between forward and backward actions is made and specified by
        is_backward, in order to account for the following special cases:

        Forward:
        - If the number of steps is equal to the maximum, then the only valid
          action is EOS (self.eos).

        Backward:
        - If the number of steps is equal to 1, then the only valid action is to
          return to the source. The specific action depends on the current state.
        """
        device = policy_outputs.device
        n_states = policy_outputs.shape[0]

        # If no mask given, everything is "normal" (we sample everywhere)
        if mask is None:
            do_sample = torch.ones(n_states, dtype=torch.bool, device=device)
        else:
            # Sample from policy only when there is no special-case mask
            do_sample = torch.all(~mask, dim=1)

        # logprobs per dimension, then summed at the end
        logprobs = torch.zeros(
            (n_states, self.n_dim), dtype=self.float, device=device
        )

        # Initialize actions tensor with EOS (inf) everywhere; special cases
        # that should be EOS just keep this value.
        actions_tensor = torch.full(
            (n_states, self.n_dim), torch.inf, dtype=self.float, device=device
        )

        # How many torsion vs linear DOFs?
        # Accept 2 or 3 outputs (n_tors, n_lin) or (n_tors, n_bonds, n_angles)
        n_tors, *rest = self._get_torsion_linear_counts()
        n_lin = sum(rest)

        # Bounds for "concentration" head before exponentiation
        conc_min, conc_max = -5.0, 5.0

        # --- Main sampling (policy-based) ---
        if torch.any(do_sample):
            if sampling_method == "uniform":
                # Not used in our experiments; kept explicit to avoid subtle bugs
                raise NotImplementedError(
                    "Uniform sampling is not implemented for mixed torsion/linear ctorus."
                )

            elif sampling_method == "policy":
                # Shape: (n_do_sample, n_dim, n_comp)
                mix_logits = policy_outputs[do_sample, 0::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                locations = policy_outputs[do_sample, 1::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                concentrations = policy_outputs[do_sample, 2::3].reshape(
                    -1, self.n_dim, self.n_comp
                )

                # Temperature on mixture logits only
                if temperature_logits is not None and temperature_logits != 1.0:
                    mix_logits = mix_logits / float(temperature_logits)

                # Clamp concentration/log-σ to avoid σ → 0 or κ → ∞
                concentrations = torch.clamp(
                    concentrations, min=conc_min, max=conc_max
                )

                if n_lin == 0:
                    # Old behaviour: everything is torsion
                    mix = Categorical(logits=mix_logits)
                    vonmises = VonMises(
                        locations,
                        torch.exp(concentrations) + self.vonmises_min_concentration,
                    )
                    distr = MixtureSameFamily(mix, vonmises)

                    angles_sampled = distr.sample()            # (n_do_sample, n_dim)
                    actions_tensor[do_sample] = angles_sampled
                    logprobs[do_sample] = distr.log_prob(angles_sampled)

                else:
                    # Split torsions vs linear DOFs
                    # --- torsions ---
                    mix_logits_t = mix_logits[:, :n_tors, :]
                    loc_t = locations[:, :n_tors, :]
                    conc_t = concentrations[:, :n_tors, :]

                    mix_t = Categorical(logits=mix_logits_t)
                    vm = VonMises(
                        loc_t,
                        torch.exp(conc_t) + self.vonmises_min_concentration,
                    )
                    distr_t = MixtureSameFamily(mix_t, vm)

                    # --- linear DOFs (bond lengths + bond angles) ---
                    mix_logits_l = mix_logits[:, n_tors:, :]
                    loc_l = locations[:, n_tors:, :]
                    conc_l = concentrations[:, n_tors:, :]

                    mix_l = Categorical(logits=mix_logits_l)
                    std_l = torch.exp(conc_l) + self.min_sigma
                    normals = Normal(loc_l, std_l)
                    distr_l = MixtureSameFamily(mix_l, normals)

                    # Sample independently from each mixture and then concatenate
                    sample_t = distr_t.sample()    # (n_do_sample, n_tors)
                    sample_l = distr_l.sample()    # (n_do_sample, n_lin)

                    angles_sampled = torch.cat([sample_t, sample_l], dim=1)
                    actions_tensor[do_sample] = angles_sampled

                    # Per-dimension log-probs
                    lp_t = distr_t.log_prob(sample_t)   # (n_do_sample, n_tors)
                    lp_l = distr_l.log_prob(sample_l)   # (n_do_sample, n_lin)
                    logprobs_do = torch.cat([lp_t, lp_l], dim=1)  # (n_do_sample, n_dim)
                    logprobs[do_sample] = logprobs_do

            else:
                raise NotImplementedError(
                    f"sampling_method = {sampling_method} not supported."
                )

        # --- Special case for backwards "back-to-source" (BTS) actions ---
        if is_backward and mask is not None:
            # BTS if mask[:, 0] is True
            do_bts = mask[:, 0]
            if torch.any(do_bts):
                source_angles = tfloat(
                    self.source[: self.n_dim],
                    float_type=self.float,
                    device=self.device,
                )
                states_from_angles = tfloat(
                    states_from,
                    float_type=self.float,
                    device=self.device,
                )[do_bts, : self.n_dim]
                actions_bts = states_from_angles - source_angles
                actions_tensor[do_bts] = actions_bts
                # Log-prob for deterministic BTS is 0 (log 1), i.e. we leave
                # logprobs[do_bts] as zeros.

        # Sum per-dimension log-probs to get scalar log-prob per state
        logprobs = torch.sum(logprobs, dim=1)

        # Convert to Python tuples for the GFlowNet agent
        actions = [tuple(a.tolist()) for a in actions_tensor]
        return actions, logprobs





    # def get_logprobs(
    #     self,
    #     policy_outputs: TensorType["n_states", "policy_output_dim"],
    #     actions: TensorType["n_states", "n_dim"],
    #     mask: TensorType["n_states", "1"],
    #     states_from: Optional[List] = None,
    #     is_backward: bool = False,
    # ) -> TensorType["batch_size"]:
    #     """
    #     Computes log probabilities of actions given policy outputs and actions.

    #     Args
    #     ----
    #     policy_outputs : tensor
    #         The output of the GFlowNet policy model.

    #     mask : tensor
    #         The mask containing information special cases.

    #     actions : tensor
    #         The actions (angle increments) from each state in the batch for which to
    #         compute the log probability.

    #     states_from : tensor
    #         Ignored.

    #     is_backward : bool
    #         Ignored.
    #     """
    #     device = policy_outputs.device
    #     do_sample = torch.all(~mask, dim=1)
    #     n_states = policy_outputs.shape[0]
    #     logprobs = torch.zeros(n_states, self.n_dim).to(device)
    #     if torch.any(do_sample):
    #         mix_logits = policy_outputs[do_sample, 0::3].reshape(
    #             -1, self.n_dim, self.n_comp
    #         )
    #         mix = Categorical(logits=mix_logits)
    #         locations = policy_outputs[do_sample, 1::3].reshape(
    #             -1, self.n_dim, self.n_comp
    #         )
    #         concentrations = policy_outputs[do_sample, 2::3].reshape(
    #             -1, self.n_dim, self.n_comp
    #         )
    #         vonmises = VonMises(
    #             locations,
    #             torch.exp(concentrations) + self.vonmises_min_concentration,
    #         )
    #         distr_angles = MixtureSameFamily(mix, vonmises)
    #         logprobs[do_sample] = distr_angles.log_prob(actions[do_sample])
    #     logprobs = torch.sum(logprobs, axis=1)
    #     return logprobs
    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", "n_dim"],
        mask: TensorType["n_states", "1"],
        states_from: Optional[List] = None,
        is_backward: bool = False,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.

        Mirrors `sample_actions_batch`:
          - torsion dims        -> Von Mises mixture
          - bond length/angles  -> Gaussian mixture
        """
        device = policy_outputs.device
        do_sample = torch.all(~mask, dim=1)
        n_states = policy_outputs.shape[0]

        logprobs = torch.zeros(n_states, self.n_dim, device=device)

        # Accept 2 or 3 outputs (n_tors, n_lin) or (n_tors, n_bonds, n_angles)
        n_tors, *rest = self._get_torsion_linear_counts()
        n_lin = sum(rest)

        # Same clamp as in sample_actions_batch
        conc_min, conc_max = -5.0, 5.0

        if torch.any(do_sample):
            mix_logits = policy_outputs[do_sample, 0::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            locations = policy_outputs[do_sample, 1::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            concentrations = policy_outputs[do_sample, 2::3].reshape(
                -1, self.n_dim, self.n_comp
            )

            # Clamp concentration/log-σ
            concentrations = torch.clamp(
                concentrations, min=conc_min, max=conc_max
            )

            acts = actions[do_sample]

            if n_lin == 0:
                # old behaviour: all dimensions torsion
                mix = Categorical(logits=mix_logits)
                vonmises = VonMises(
                    locations,
                    torch.exp(concentrations) + self.vonmises_min_concentration,
                )
                distr_angles = MixtureSameFamily(mix, vonmises)
                logprobs[do_sample] = distr_angles.log_prob(acts)

            else:
                # --- torsions ---
                mix_logits_t = mix_logits[:, :n_tors, :]
                loc_t = locations[:, :n_tors, :]
                conc_t = concentrations[:, :n_tors, :]
                acts_t = acts[:, :n_tors]

                mix_t = Categorical(logits=mix_logits_t)
                vm = VonMises(
                    loc_t,
                    torch.exp(conc_t) + self.vonmises_min_concentration,
                )
                distr_t = MixtureSameFamily(mix_t, vm)

                # --- linear dims (bond lengths + bond angles) ---
                mix_logits_l = mix_logits[:, n_tors:, :]
                loc_l = locations[:, n_tors:, :]
                conc_l = concentrations[:, n_tors:, :]
                acts_l = acts[:, n_tors:]

                mix_l = Categorical(logits=mix_logits_l)
                std_l = torch.exp(conc_l) + self.min_sigma
                normals = Normal(loc_l, std_l)
                distr_l = MixtureSameFamily(mix_l, normals)

                lp_t = distr_t.log_prob(acts_t)   # (n_do, n_tors)
                lp_l = distr_l.log_prob(acts_l)   # (n_do, n_lin)
                logprobs_do = torch.cat([lp_t, lp_l], dim=1)
                logprobs[do_sample] = logprobs_do

        # Sum over dimensions to get scalar log-prob per state
        logprobs = torch.sum(logprobs, axis=1)
        return logprobs





    def _step(
        self,
        action: Tuple[float],
        backward: bool,
    ) -> Tuple[List[float], Tuple[float], bool]:
        """
        Updates self.state given a non-EOS action. This method is called by both step()
        and step_backwards(), with the corresponding value of argument backward.

        Forward steps:
            - Add action increments to state angles.
            - Increment n_actions value of state.
        Backward steps:
            - Subtract action increments from state angles.
            - Decrement n_actions value of state.

        Args
        ----
        action : tuple
            Action to be executed. An action is a vector where the value at position d
            indicates the increment in the angle at dimension d.

        backward : bool
            If True, perform backward step. Otherwise (default), perform forward step.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        for dim, angle in enumerate(action):
            if backward:
                self.state[int(dim)] -= angle
            else:
                self.state[int(dim)] += angle
            self.state[int(dim)] = self.state[int(dim)] % (2 * np.pi)
        if backward:
            self.state[-1] -= 1
        else:
            self.state[-1] += 1
        assert self.state[-1] >= 0 and self.state[-1] <= self.length_traj
        # If n_steps is equal to 0, set source to avoid escaping comparison to source.
        if self.state[-1] == 0:
            self.state = copy(self.source)

    def step(
        self, action: Tuple[float], skip_mask_check: bool = False
    ) -> Tuple[List[float], Tuple[float], bool]:
        """
        Executes forward step given an action.

        See: _step().

        Args
        ----
        action : tuple
            Action to be executed. An action is a vector where the value at position d
            indicates the increment in the angle at dimension d.

        skip_mask_check : bool
            Ignored because the action space space is fully continuous, therefore there
            is nothing to check.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        # If done is True, return invalid
        if self.done:
            return self.state, action, False
        # If action is EOS, check that the number of steps is equal to the trajectory
        # length, set done to True, increment n_actions and return same state
        elif action == self.eos:
            assert self.state[-1] == self.length_traj
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # Otherwise perform action
        else:
            self.n_actions += 1
            self._step(action, backward=False)
            return self.state, action, True

    def step_backwards(
        self, action: Tuple[float], skip_mask_check: bool = False
    ) -> Tuple[List[float], Tuple[float], bool]:
        """
        Executes backward step given an action.

        See: _step().

        Args
        ----
        action : tuple
            Action to be executed. An action is a vector where the value at position d
            indicates the increment in the angle at dimension d.

        skip_mask_check : bool
            Ignored because the action space space is fully continuous, therefore there
            is nothing to check.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        # If done is True, set done to False, increment n_actions and return same state
        if self.done:
            assert action == self.eos
            self.done = False
            self.n_actions += 1
            return self.state, action, True
        # Otherwise perform action
        else:
            assert action != self.eos
            self.n_actions += 1
            self._step(action, backward=True)
            return self.state, action, True

    def get_max_traj_length(self):
        return int(self.length_traj) + 1
