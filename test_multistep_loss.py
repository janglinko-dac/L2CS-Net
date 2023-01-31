import numpy as np
import torch

def get_per_step_loss_importance_vector(current_epoch):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        number_of_training_steps_per_iter = 5
        multi_step_loss_num_epochs = 15


        loss_weights = np.ones(shape=(number_of_training_steps_per_iter)) * (
                1.0 / number_of_training_steps_per_iter)
        decay_rate = 1.0 / number_of_training_steps_per_iter / multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (current_epoch * (number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        # loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

if __name__ == '__main__':
    pass