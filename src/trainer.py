
import os
from functools import partial

import bayesflow as bf
import tensorflow as tf

from tqdm import tqdm


def logging_backprop_step(input_dict, amortizer, optimizer, writer, **kwargs):
    """Computes the loss of the provided amortizer given an input dictionary and applies gradients.

    Parameters
    ----------
    input_dict  : dict
        The configured output of the generative model
    amortizer   : tf.keras.Model
        The custom amortizer. Needs to implement a compute_loss method.
    optimizer   : tf.keras.optimizers.Optimizer
        The optimizer used to update the amortizer's parameters.
    **kwargs    : dict
        Optional keyword arguments passed to the network's compute_loss method

    Returns
    -------
    loss : dict
        The outputs of the compute_loss() method of the amortizer comprising all
        loss components, such as divergences or regularization.
    """

    # Forward pass and loss computation
    with tf.GradientTape() as tape:
        # Compute custom loss
        loss = amortizer.compute_loss(input_dict, training=True, **kwargs)
        # If dict, add components
        if type(loss) is dict:
            _loss = tf.add_n(list(loss.values()))
        else:
            _loss = loss
        # Collect regularization loss, if any
        if amortizer.losses != []:
            reg = tf.add_n(amortizer.losses)
            _loss += reg
            if type(loss) is dict:
                loss["W.Decay"] = reg
            else:
                loss = {"Loss": loss, "W.Decay": reg}
    # One step backprop and return loss
    gradients = tape.gradient(_loss, amortizer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, amortizer.trainable_variables))
    
    with writer.as_default():
        tf.summary.histogram("Input params", input_dict["parameters"])
        tf.summary.histogram("Gradients/coupling_layer_0", gradients[23])
        tf.summary.histogram("Gradients/coupling_layer_1", gradients[49])
        tf.summary.histogram("Gradients/coupling_layer_2", gradients[75])
        tf.summary.histogram("Gradients/coupling_layer_3", gradients[101])
        tf.summary.histogram("Gradients/coupling_layer_4", gradients[127])
        tf.summary.histogram("Gradients/coupling_layer_5", gradients[153])
        tf.summary.histogram("Gradients/attention_output_layer_0_0", gradients[162])
        tf.summary.histogram("Gradients/attention_output_layer_0_1", gradients[176])
        tf.summary.histogram("Gradients/attention_output_layer_1_0", gradients[191])
        tf.summary.histogram("Gradients/attention_output_layer_1_1", gradients[205])
        tf.summary.histogram("Gradients/attention_output_layer_pooling", gradients[220])
        tf.summary.scalar("Loss", loss["Loss"])
        tf.summary.scalar("W.Decay", loss["W.Decay"])

    return loss


class CustomTrainer(bf.trainers.Trainer):
    def _train_step(self, batch_size, update_step, input_dict=None, writer=None, step=None, **kwargs):
        if input_dict is None:
            input_dict = self._forward_inference(
                batch_size, **kwargs.pop("conf_args", {}), **kwargs.pop("model_args", {})
            )
        if self.simulation_memory is not None:
            self.simulation_memory.store(input_dict)
        loss = update_step(input_dict, self.amortizer, self.optimizer, **kwargs.pop("net_args", {}))
        return loss

    def train_online(self, epochs, iterations_per_epoch, batch_size, save_checkpoint=True, optimizer=None, reuse_optimizer=False, early_stopping=False, use_autograph=True, validation_sims=None, **kwargs):
        assert self.generative_model is not None, "No generative model found. Only offline training is possible!"

        writer = tf.summary.create_file_writer(os.path.join("checkpoints", "logs", "train"))

        backprop_step = partial(logging_backprop_step, writer=writer)

        # Compile update function, if specified
        if use_autograph:
            _backprop_step = tf.function(backprop_step, reduce_retracing=True)
        else:
            _backprop_step = backprop_step

        # Create new optimizer and initialize loss history
        self._setup_optimizer(optimizer, epochs, iterations_per_epoch)
        self.loss_history.start_new_run()
        validation_sims = self._config_validation(validation_sims, **kwargs.pop("val_model_args", {}))

        # Create early stopper, if conditions met, otherwise None returned
        early_stopper = self._config_early_stopping(early_stopping, validation_sims, **kwargs)

        step = tf.Variable(0, dtype=tf.int64)

        # Loop through training epochs
        for ep in range(1, epochs + 1):
            with tqdm(total=iterations_per_epoch, desc=f"Training epoch {ep}", mininterval=bf.trainers.TQDM_MININTERVAL) as p_bar:
                for it in range(1, iterations_per_epoch + 1):
                    tf.summary.experimental.set_step(step)
                    # Perform one training step and obtain current loss value
                    loss = self._train_step(batch_size, update_step=_backprop_step, **kwargs)

                    # Store returned loss
                    self.loss_history.add_entry(ep, loss)

                    # Compute running loss
                    avg_dict = self.loss_history.get_running_losses(ep)

                    # Extract current learning rate
                    lr = bf.trainers.extract_current_lr(self.optimizer)

                    # Format for display on progress bar
                    disp_str = bf.trainers.format_loss_string(ep, it, loss, avg_dict, lr=lr)

                    # Update progress bar
                    p_bar.set_postfix_str(disp_str, refresh=False)
                    p_bar.update(1)
                    step.assign_add(1)

            # Store and compute validation loss, if specified
            self._validation(ep, validation_sims, **kwargs)
            self._save_trainer(save_checkpoint)

            # Check early stopping, if specified
            if self._check_early_stopping(early_stopper):
                break

        # Remove optimizer reference, if not set as persistent
        if not reuse_optimizer:
            self.optimizer = None
        return self.loss_history.get_plottable()
    