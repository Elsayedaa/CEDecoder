import numpy as np
from multiprocessing import Manager, Pool
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense 
print(tf.config.list_physical_devices('GPU'))

def model(trainX, trainY, testX, testY, epochs, batch_size):
    model = Sequential()
    model.add(
        LSTM(
            100, # first layer, 100 hidden neurons - reduce the feature space
            input_shape = (trainX.shape[1], trainX.shape[2]), # the shape of the input is timelags x features
            return_sequences = True
        )
    )

    model.add(
        LSTM(
            10 # second layer, 10 hidden neurons - map features to the 10 SFs
        )
    )
    model.add(Dense(1)) # output layer - predict 1 out of the 10 SFs
    model.compile(optimizer = 'adam', loss = 'mae')

    ## fit the model
    history = model.fit(
        trainX, trainY,
        epochs = epochs,
        batch_size = batch_size,
        validation_data = (testX, testY),
        verbose = 2,
        shuffle = False
    )

    ## make a prediction
    yhat = model.predict(testX)

    ## return the mean absolute error over the 250 ms response
    mae = np.abs(testY.reshape(-1,1)-yhat).flatten().reshape(10,250).mean(0)
    print('Done')
    return yhat, mae

def lstm_decoder(
        trainX, trainY,
        testX, testY,
        epochs = 50, batch_size = 250
):
    mgr = Manager()
    pool = Pool(32)
    job = pool.apply_async(
        model,
        (
            trainX, trainY,
            testX, testY,
            epochs, batch_size
        )
    )

    pool.close()
    pool.join()  
    result = job.get()
    return result

def multi_run(
        trainX, trainY,
        testX, testY,
        epochs = 50, batch_size = 250,
        runs = 10
):
    mgr = Manager()
    pool = Pool(32)
    jobs = []
    for run in range(runs):
        job = pool.apply_async(
            model,
            (
                trainX, trainY,
                testX, testY,
                epochs, batch_size
            )
        )
        jobs.append(job)
    pool.close()
    pool.join() 
    results = np.array([job.get() for job in jobs])
    return results

          