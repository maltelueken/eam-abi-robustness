

import bayesflow as bf
import keras

from bayesflow.utils import filter_kwargs


class CustomContinuousApproximator(bf.approximators.ContinuousApproximator):
    def _sample(self, num_samples, inference_conditions = None, summary_variables = None, **kwargs):
        if self.summary_network is None:
            if summary_variables is not None:
                raise ValueError("Cannot use summary variables without a summary network.")
        else:
            if summary_variables is None:
                raise ValueError("Summary variables are required when a summary network is present.")

            summary_outputs = self.summary_network(
                summary_variables
            )

            if inference_conditions is None:
                inference_conditions = summary_outputs
            else:
                inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=1)

        if inference_conditions is not None:
            # conditions must always have shape (batch_size, dims)
            batch_size = keras.ops.shape(inference_conditions)[0]
            inference_conditions = keras.ops.expand_dims(inference_conditions, axis=1)
            inference_conditions = keras.ops.broadcast_to(
                inference_conditions, (batch_size, num_samples, *keras.ops.shape(inference_conditions)[2:])
            )
            batch_shape = (batch_size, num_samples)
        else:
            batch_shape = (num_samples,)

        return self.inference_network.sample(
            batch_shape,
            conditions=inference_conditions,
            **filter_kwargs(kwargs, self.inference_network.sample),
        )
    