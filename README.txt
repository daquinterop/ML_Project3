## predict(model, x, **kwargs)
    The NN model is a dict object with elements for the weights and bias of the model. The calculation of the nodes (a, h, z, y_hat) is done using the "matmul" funciton from the numpy library. "y_hat" variable is used to to storage the result of the Softmax function. The prediction is in the variable "y", which is the indexes for the maximum values of "y_hat". When the "return_a_h" argument is pased to the function it returns the variables "y_hat", "a" and "h". This feature is very useful, since those variables are used for the "calculate_loss" and "build_model" functions. The function is designed to have more than a sample as input and return the prediction for every sample.

## build_model(X, y, nn_hdim, num_passes=20000, print_loss=False)
    This function returns the model with the resulting parameters after the training. To train the NN it uses the backward propagation algorithm. The model setting all the weights and bias as a random value from a standarized normal distribution. The 1D labels are converted to a 2D array. 

    Every epoch the update of the parameters is done by batches. The dataset is randomly splited in 10 equal parts. Those 10 equal parts are the 10 batches used in every epoch. Using the provided equations, the derivatives are calculated and then used to update the parameters. the "matmul" function is used as it was used in the predict model function. The bias term is the average of the resulting bias for all the samples of that batch.

    When the number of passes (epoch) is reached the function return the model.

## calculate_loss(model, X, Y)
    This function simply calculates the loss ussing the categorical cross-entropy loss function. To do that, the predict function is called to return the "y_hat" variable (2D array). The "Y" variable (labels) are converted to a 2D array. Finally, the function returns the loss.