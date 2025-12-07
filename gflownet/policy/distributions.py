import torch
from torch import nn
from torch.distributions import Distribution, Normal, VonMises

class MoleculeHeterogeneousDistribution(Distribution):
    """
    A unified distribution wrapper that handles:
    1. Torsions: VonMises (Periodic, [-pi, pi])
    2. Bond Lengths: Normal (Euclidean, Delta from ref)
    3. Bond Angles: Normal (Euclidean, Delta from ref)
    """
    def __init__(self, torsion_params, length_params, angle_params):
        """
        :param torsion_params: Tuple(loc, concentration)
        :param length_params: Tuple(loc, scale)
        :param angle_params: Tuple(loc, scale)
        """
        self.torsion_dist = VonMises(*torsion_params)
        self.length_dist = Normal(*length_params)
        self.angle_dist = Normal(*angle_params)

    def sample(self, sample_shape=torch.Size()):
        # Sample independently
        t_samp = self.torsion_dist.sample(sample_shape)
        l_samp = self.length_dist.sample(sample_shape)
        a_samp = self.angle_dist.sample(sample_shape)
        
        # Concatenate along the feature dimension (dim=-1)
        # Result shape: [batch, n_torsions + n_lengths + n_angles]
        return torch.cat([t_samp, l_samp, a_samp], dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        # Reparameterized sampling (allows gradients to flow)
        t_samp = self.torsion_dist.rsample(sample_shape)
        l_samp = self.length_dist.rsample(sample_shape)
        a_samp = self.angle_dist.rsample(sample_shape)
        return torch.cat([t_samp, l_samp, a_samp], dim=-1)

    def log_prob(self, value):
        """
        value: [batch, total_dims]
        We must split the input vector back into chunks to evaluate log_prob correctly.
        """
        # We need the shapes from the distributions to know where to split
        # VonMises is [batch, n_torsions]
        n_tors = self.torsion_dist.loc.shape[-1]
        n_lens = self.length_dist.loc.shape[-1]
        n_angs = self.angle_dist.loc.shape[-1]
        
        # Split the input value
        t_val = value[..., :n_tors]
        l_val = value[..., n_tors : n_tors + n_lens]
        a_val = value[..., n_tors + n_lens :]

        # Compute log probs
        t_lp = self.torsion_dist.log_prob(t_val)
        l_lp = self.length_dist.log_prob(l_val)
        a_lp = self.angle_dist.log_prob(a_val)
        
        # Sum along the feature dimension to get total log_prob for the state
        # Shape: [batch]
        total_lp = t_lp.sum(-1) + l_lp.sum(-1) + a_lp.sum(-1)
        return total_lp