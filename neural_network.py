import numpy as np

def calculate_loss(model, X, Y):
    N = X.shape[0]
    y_hat, _, _ = predict(model, X, return_a_h=True)
    Y2D = np.zeros_like(y_hat)
    for n, i in enumerate(Y): Y2D[n][i] = 1
    return -np.sum(Y2D * np.log(y_hat))/N

def predict(model, x, **kwargs):
    a = np.matmul(x, model['W1']) + model['b1']
    h = np.tanh(a)
    z = np.matmul(h, model['W2']) + model['b2']
    y_hat = np.exp(z) / np.exp(z).sum(axis=1).reshape(z.shape[0], 1)
    y = y_hat.argmax(axis=1)
    if kwargs.get('return_a_h', False):
        return y_hat, a, h, 
    return y

def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    learning_rate = 0.05
    N = X.shape[0]
    model = {
        'W1': np.random.normal(size=(2, nn_hdim)),
        'b1': np.random.normal(size=(1, nn_hdim)),
        'W2': np.random.normal(size=(nn_hdim, 2)),
        'b2': np.random.normal(size=(1, 2))
    }
    N = X.shape[0]
    Y2D = np.zeros((N, 2))
    for n, i in enumerate(y): Y2D[n][i] = 1
    for n_pass in range(num_passes):
        resample_indexes = np.arange(N)
        np.random.shuffle(resample_indexes)
        resample_indexes = np.array_split(resample_indexes, 10)
        for sample_indexes in resample_indexes:
            sample = X[sample_indexes]
            target = Y2D[sample_indexes]
            y_hat, a, h = predict(model, sample, return_a_h=True)            
            dLdy = y_hat - target
            dLda = (1 - np.tanh(a)**2) * np.matmul(dLdy, model['W2'].T)
            dLdW2 = np.matmul(h.T, dLdy)
            dLdb2 = dLdy[:]
            dLdW1 = np.matmul(sample.T, dLda)
            dLdb1 = dLda[:]
            model['W1'] = model['W1'] - learning_rate*dLdW1
            model['b1'] = model['b1'] - learning_rate*dLdb1.mean(axis=0)
            model['W2'] = model['W2'] - learning_rate*dLdW2
            model['b2'] = model['b2'] - learning_rate*dLdb2.mean(axis=0)
        if print_loss and n_pass % 1000 == 0:
            print(f'Loss: {calculate_loss(model, X, y):.5f}')
    return model
