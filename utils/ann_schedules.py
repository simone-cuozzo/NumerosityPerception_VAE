# -*- coding: utf-8 -*-

### beta parameter linear annealing schedules ### 
# start_beta = starting value for beta par
# beta_final = final beta value 
# num_epochs = number of epochs of the training. It is then divided by 2 so that for the first half of the training the beta value is linearly increased and then kept constant for the second half
# epoch = current epoch of training
def linear(epoch, num_epochs, start_beta, beta_final, annealing = False):
  if annealing == True:
    beta = start_beta + ((beta_final-start_beta)/(int(num_epochs/2))) * (epoch)
    if beta >= beta_final:
      beta = beta_final
  elif annealing == False:
    beta = beta_final
  return beta

### beta parameter cyclical annealing schedules ###
# start_beta = starting value for beta par
# beta_final = final beta value 
# num_epochs = number of epochs of the training. It is then divided by 2 so that for the first half of the training the beta value is linearly increased and then kept constant for the second half
# epoch = current epoch of training
# cycles = number of cycles in which the beta parameter is iteratively increased and then kept constant
# ratio = ratio between increasing period amd constant period inside a cycle
def cyclical_linear(epoch, num_epochs, start_beta, beta_final, cycles, ratio = 0.5):
    beta = start_beta + ((beta_final-start_beta)/(int((num_epochs/cycles)*ratio))) * ((epoch) % (num_epochs/cycles))
    if beta > beta_final:
        beta = beta_final
    return beta


