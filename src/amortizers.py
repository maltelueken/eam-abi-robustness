
import math

import bayesflow as bf
import tensorflow as tf


class CustomAmortizedPosterior(bf.amortizers.AmortizedPosterior):
    def compute_loss(self, input_dict, **kwargs):
        # Get amortizer outputs
        net_out, sum_out = self(input_dict, return_summary=True, **kwargs)
        z, log_det_J = net_out

        # Case summary loss should be computed
        if self.summary_loss is not None:
            sum_loss = self.summary_loss(sum_out)
        # Case no summary loss, simply add 0 for convenience
        else:
            sum_loss = 0.0

        # Case dynamic latent space - function of summary conditions
        if self.latent_is_dynamic:
            logpdf = self.latent_dist(sum_out).log_prob(z)
        # Case _static latent space
        else:
            logpdf = self.latent_dist.log_prob(z)

        logpdf = tf.clip_by_value(logpdf, clip_value_min=math.log(1-12), clip_value_max=math.inf)
        # sum_loss = tf.clip_by_value(sum_loss, clip_value_min=0, clip_value_max=)

        # Compute and return total loss
        total_loss = tf.reduce_mean(-logpdf - log_det_J) + sum_loss
        return total_loss
