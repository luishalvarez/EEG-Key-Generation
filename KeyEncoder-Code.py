"""

@author: Luis Hernández-Álvarez, Elena Barbierato, Stefano Caputo, 
José María de Fuentes, Lorena González-Manzano, Luis Hernández Encinas
and Lorenzo Mucchi.

Code of the article titled: KeyEncoder: Towards a secure and usable
EEG-based cryptographic key generation mechanism.

Data acquired in a measurement campaign.

"""

#%% Import 

import numpy as np
from tensorflow import keras
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import joblib
import pickle

from scipy.stats import entropy

import pandas as pd
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import vstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.utils import shuffle
from sympy import fwht
from sklearn.metrics import hamming_loss
from scipy.spatial import distance
import hashlib
from hexhamming import hamming_distance_string
import coden

import xlsxwriter

############################ LOAD DATA ########################################

folder_path = 'EEG'
file_list = glob.glob(folder_path + "/*.csv")
n_users = 39

data = np.zeros([n_users, 650, 673])
n_samples = np.zeros([n_users])
for user in np.arange(n_users):
    main_dataframe = pd.DataFrame(pd.read_csv(file_list[user]))
    df = main_dataframe.to_numpy()
    n_samples[user] = df.shape[0]
    data[user, :df.shape[0], :] = df
    
########################## USEFUL FUNCTIONS ###################################
    
def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1, lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y


def puntomedio(mini, maxi):
    result = (mini+maxi)/2
    return result
    
# %%

###################### DEFINITION OF VARIABLES ################################

timesteps = 1
n_features = 48
divisor_sigma = 1

n_train = 0.8
n_test = 1-n_train
feat_per_ch = 48
n_channels = 14

keys_train1 = np.zeros([n_users, 200, 500])
numkeys_train1 = np.zeros([n_users])
lenkeys_train1 = np.zeros([n_users])
keys_test1 = np.zeros([n_users, 200, 500])
numkeys_test1 = np.zeros([n_users])
lenkeys_test1 = np.zeros([n_users])
keys_impostor1 = np.zeros([n_users, 200, 250])

keys_list = []
hash_list = []

rep_keys1 = []
final_keys1 = {}
rep_keys_test1 = []
rep_keys_impostor1 = []

samples_train = np.zeros([n_users])
samples_test = np.zeros([n_users])

for user in np.arange(n_users):
    samples_train[user] = int(np.round(n_train*n_samples[user]))
    samples_test[user] = int(n_samples[user]-samples_train[user])

FAR = []
FRR = []
HTER = []


## Uncomment to generate an .xlsx document with information about the experiment
'''
libro = xlsxwriter.Workbook('DatosKeyEncoder.xlsx')
hoja = libro.add_worksheet()

hoja.write(0, 0, 'User (WH-All concatenated)')
hoja.write(0, 1, 'Key lenght')
hoja.write(0, 2, '% val')
hoja.write(0, 3, '% test')
hoja.write(0, 4, 'Most common keys coincide')
hoja.write(0, 5, 'Impostor generates key')
hoja.write(0, 6, 'Key is most common in impostor')
hoja.write(0, 7, 'FAR')
hoja.write(0, 8, 'FRR')
hoja.write(0, 9, 'HTER')
hoja.write(0, 10, '% of 0')
hoja.write(0, 11, '% of 1')
hoja.write(0, 12, 'key')
'''


## Repeat the same process for every user
for myuser in np.arange(n_users):

    row = myuser + 1
    allow = 1
    coincide1 = 0
    coincide_impostor1 = 0
    coincide_group1 = -1
    coincide_impostor_group1 = -1
    impostor_generate = 0

    print('\n User number: ', myuser)
    errors_train = np.zeros([n_users, n_channels])
    errors_test = np.zeros([n_users, n_channels])
    errors_impostor = np.zeros([n_users, n_channels])
    
    traindata = np.zeros([n_channels, 650, feat_per_ch])
    testdata = np.zeros([n_channels, 650, feat_per_ch])
    impostordata = np.zeros([n_channels, 650, feat_per_ch])
    my_samples_train = int(samples_train[myuser])
    my_samples_test = int(samples_test[myuser])
    
    columns1 = []
    
    ## Set different seeds to change the train/test data 
    #np.random.seed(myuser)
    #np.random.shuffle(data[myuser][:int(n_samples[0])])
    
    ## Repeat the same process for every channel
    for channel_used in np.arange(n_channels):
        
###### PREPARE THE TRAIN, TEST AND IMPOSTOR DATA ##############################
        
        traindata[channel_used, :int(my_samples_train), :] = data[myuser, :int(
            my_samples_train), int(channel_used*feat_per_ch):int((channel_used+1)*feat_per_ch)]
        testdata[channel_used, :int(my_samples_test), :] = (data[myuser, int(my_samples_train):int(
            my_samples_train+my_samples_test), int(channel_used*feat_per_ch):int((channel_used+1)*feat_per_ch)])
        counter = 0
        while counter<my_samples_test:
            random_user = random.randint(0,n_users-1)
            if(random_user != myuser):
                random_sample = random.randint(0,n_samples[myuser]-1)
                impostordata[channel_used, counter, :] = (data[random_user, random_sample, int(channel_used*feat_per_ch):int((channel_used+1)*feat_per_ch)])
                counter = counter + 1
            
        scaler = StandardScaler()
        traindata[channel_used][:my_samples_train] = scaler.fit_transform(traindata[channel_used][:my_samples_train])
        testdata[channel_used][:my_samples_test] = scaler.transform(testdata[channel_used][:my_samples_test])
        impostordata[channel_used][:my_samples_test] = scaler.transform(impostordata[channel_used][:my_samples_test])
        
        X, y = temporalize(X=traindata[channel_used][:int(my_samples_train)], y=np.zeros(
            len(traindata[channel_used][:int(my_samples_train)])), lookback=timesteps)
        X = np.array(X)
        X_train = X.reshape(X.shape[0], timesteps, n_features)
        
        X, y = temporalize(X=testdata[channel_used][:int(my_samples_test)], y=np.zeros(
            len(testdata[channel_used][:int(my_samples_test)])), lookback=timesteps)
        X = np.array(X)
        X_test = X.reshape(X.shape[0], timesteps, n_features)
        
        X, y = temporalize(X=impostordata[channel_used][:int(my_samples_test)], y=np.zeros(
            len(impostordata[channel_used][:int(my_samples_test)])), lookback=timesteps)
        X = np.array(X)
        X_impostor = X.reshape(X.shape[0], timesteps, n_features)
        
        
######################### AUTOENCODER DEFINITION ##############################
        
        ## Uncomment to train an autoencoder
        '''
        model = Sequential()
        model.add(LSTM(32, activation='selu', input_shape=(
            timesteps, n_features), return_sequences=True))
        model.add(Dropout(rate=0.1))
        model.add(LSTM(16, activation='selu', return_sequences=False))
        model.add(RepeatVector(timesteps))
        model.add(LSTM(16, activation='selu', return_sequences=True))
        model.add(Dropout(rate=0.1))
        model.add(LSTM(32, activation='selu', return_sequences=True))
        model.add(TimeDistributed(Dense(n_features)))
        ADAM = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.2, beta_2=0.999,
                                      epsilon=None, decay=0.0, amsgrad=False, clipnorm=1.)
        
        
        model.summary()
        
        cp = ModelCheckpoint('model/', save_best_only=True)
        model.compile(optimizer=ADAM, loss='mse')50\%
        
        history = model.fit(X_train, X_train, epochs=750,
                             batch_size=32, validation_split=0.2, verbose=0)
        
        model.save('./Key Generation/Modelos-Dropout0/User ' + str(myuser) + '- Channel ' + str(channel_used) + '.h5')
    
        plt.figure(2)
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title('User ' +  str(myuser) + ' - Channel ' + str(channel_used))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')        
        plt.legend()
        plt.savefig('User ' + str(myuser) + ' - Channel ' + str(channel_used))
        plt.show() 
    
        '''
        
        ## Load the already trained model        
        model = keras.models.load_model('./Key Generation/Modelos-Dropout0/User ' + str(myuser) + '- Channel ' + str(channel_used) + '.h5')
        
        ## Results for training
        yhat = model.predict(X_train, verbose=0)
        mse_train = []
        for i in np.arange(X_train.shape[0]):
            mse_train.append(mean_squared_error(X_train[i], yhat[i]))
            
        ## Results for testing
        yhat_test = model.predict(X_test, verbose=0)
        mse_test = []
        for i in np.arange(X_test.shape[0]):
            mse_test.append(mean_squared_error(X_test[i], yhat_test[i]))
        
        ## Results for impostor
        yhat_impostor = model.predict(X_impostor, verbose=0)
        mse_impostor = []
        for i in np.arange(X_impostor.shape[0]):
            mse_impostor.append(mean_squared_error(X_impostor[i], yhat_impostor[i]))
            
        errors_train[myuser,channel_used] = np.mean(mse_train)
        errors_test[myuser,channel_used] = np.mean(mse_test)
        errors_impostor[myuser,channel_used] = np.mean(mse_impostor)
        
###################### LATENT VECTOR DISCRETIZATION ###########################

        ## Obtain the latent vector
        mymodel = Model(inputs=model.inputs, outputs=model.layers[1].output)
        latent = mymodel.predict(X_test, verbose=0)
        latent_impostor = mymodel.predict(X_impostor, verbose=0)
        ## Perform the Walsh-Hadamard transform
        for index, sample in enumerate(latent):
            latent[index] = np.asarray(fwht(sample))
        for index, sample in enumerate(latent_impostor):
            latent_impostor[index] = np.asarray(fwht(sample))
        
        ## Prepare variables for latent vector discretization
        mean = np.mean(latent, axis=1)
        std = np.std(latent, axis=1)
        med = np.median(latent, axis=1)
        pmizq = (med+np.min(latent, axis=1))/2
        pmder = (med+np.max(latent, axis=1))/2
        mymin = np.min(latent, axis=1)
        mymax = np.max(latent, axis=1)
        pm = puntomedio(mymin,mymax)
        
        mean_impostor = np.mean(latent_impostor, axis=1)
        std_impostor = np.std(latent_impostor, axis=1)
        med_impostor = np.median(latent_impostor, axis=1)
        pmizq_impostor = (med_impostor+np.min(latent_impostor, axis=1))/2
        pmder_impostor = (med_impostor+np.max(latent_impostor, axis=1))/2
        mymin_impostor = np.min(latent_impostor, axis=1)
        mymax_impostor = np.max(latent_impostor, axis=1)
        pm_impostor = puntomedio(mymin_impostor,mymax_impostor)

        col_perc = 1
        
        ## Change the seed to shuffle the latent vectors
        #np.random.seed(0)
        #np.random.shuffle(latent)
        #np.random.shuffle(latent_impostor)
        
        ## Latent vector discretization
        latent1 = np.zeros([latent.shape[0],latent.shape[1]])
        for it in np.arange(4):
            if(it == 0):
                number = mean
                number_impostor = mean_impostor                
            elif(it == 1):
                number = med
                number_impostor = med_impostor
            elif(it == 2):
                number = np.zeros(latent.shape[0])
                number_impostor = np.zeros(latent.shape[0])                                
            elif(it == 3):
                number = pm
                number_impostor = pm_impostor
            
        
            for j in np.arange(latent.shape[0]):
                latent1[j] = np.digitize(latent[j],[number[j]])
            latent_train1 = latent1[:int(latent1.shape[0]/2),:]
            latent_val1 = latent1[int(latent1.shape[0]/2):int(3*latent1.shape[0]/4),:]
            latent_test1 = latent1[int(3*latent1.shape[0]/4):,:]
            
            latent_impostor1 = np.zeros([latent_impostor.shape[0], latent_impostor.shape[1]])
            for j in np.arange(latent_impostor.shape[0]):
                latent_impostor1[j] = np.digitize(latent_impostor[j],[number_impostor[j]])
            latent_impostor_test1 = latent_impostor1[int(3*latent_impostor1.shape[0]/4):,:]
            
            for c in np.arange(latent_train1.shape[1]):
                unique1, counts1 = np.unique(latent_train1[:,c], return_counts=True)
                if np.where(counts1 >= int(col_perc*latent_train1.shape[0]))[0].size != 0:
                    columns1.append(c+it*16+channel_used*4*16)
            if allow == 1:
                key_generator_train1 = np.zeros([latent_val1.shape[0], latent_val1.shape[1]])
                key_generator_test1 = np.zeros([latent_test1.shape[0], latent_test1.shape[1]])
                key_generator_impostor1 = np.zeros([latent_impostor_test1.shape[0], latent_impostor_test1.shape[1]])
                allow = 0
                
            key_generator_train1 = np.hstack((key_generator_train1, latent_val1))
            key_generator_test1 = np.hstack((key_generator_test1, latent_test1))
            key_generator_impostor1 = np.hstack((key_generator_impostor1, latent_impostor_test1))
        
    key_generator_train1 = np.delete(key_generator_train1,np.arange(latent_val1.shape[1]),1)
    key_generator_test1 = np.delete(key_generator_test1,np.arange(latent_test1.shape[1]),1)
    key_generator_impostor1 = np.delete(key_generator_impostor1,np.arange(latent_impostor_test1.shape[1]),1)
    
############################ SEED GENERATION ##################################

    numkeys_train1[myuser] = latent_val1.shape[0]
    lenkeys_train1[myuser] = len(columns1)
    
    numkeys_test1[myuser] = latent_test1.shape[0]
    lenkeys_test1[myuser] = len(columns1)
    
    if(len(columns1) != 0):
        
        keys_train1[myuser][:int(latent_val1.shape[0]),:int(len(columns1))] = key_generator_train1[:,columns1]
        
        df = pd.DataFrame(keys_train1[myuser][:int(numkeys_train1[myuser]),:int(lenkeys_train1[myuser])])
        df1 = df[df.duplicated(keep=False)]
        df1 = df1.groupby(df1.columns.tolist()).apply(lambda x: x.index.tolist()).values.tolist()
        max_pos = 0
        max_l = 0
        for group in np.arange(len(df1)):
            if(len(df1[group]) > max_l):
                max_l = len(df1[group])
                max_pos = group
        if(len(df1)==0):
            rep_keys1.append(0)
            actual_key1 = 0
        else:
            rep_keys1.append(len(df1[max_pos])/numkeys_train1[myuser])
            actual_key1 = keys_train1[myuser][df1[max_pos][0],:int(lenkeys_train1[myuser])]
        
        final_keys1[str(myuser)] = actual_key1       
        
        
        
        keys_test1[myuser][:int(latent_test1.shape[0]),:int(len(columns1))] = key_generator_test1[:,columns1]
        
        df = pd.DataFrame(keys_test1[myuser][:int(numkeys_test1[myuser]),:int(lenkeys_test1[myuser])])
        df1 = df[df.duplicated(keep=False)]
        df1 = df1.groupby(df1.columns.tolist()).apply(lambda x: x.index.tolist()).values.tolist()
        max_pos = 0
        max_l = 0
        
        for group in np.arange(len(df1)):
            if(len(df1[group]) > max_l):
                max_l = len(df1[group])
                max_pos = group
            potential_key = keys_test1[myuser][df1[group][0],:int(lenkeys_test1[myuser])]
            if (potential_key == actual_key1).all() == True:
                coincide_group1 = group
        
        
        if(max_pos == coincide_group1):
            coincide1 = 2
        if(coincide_group1 != -1):
            if(max_pos != coincide_group1):
                coincide1 = 1
                
        if(len(df1)==0):
            rep_keys_test1.append(0)
        else:
            rep_keys_test1.append(len(df1[max_pos])/numkeys_test1[myuser])
            
            
            
        keys_impostor1[myuser][:int(latent_impostor_test1.shape[0]),:int(len(columns1))] = key_generator_impostor1[:,columns1]
        
        df_impostor = pd.DataFrame(keys_impostor1[myuser][:int(numkeys_test1[myuser]),:int(lenkeys_test1[myuser])])
        df1_impostor = df_impostor[df_impostor.duplicated(keep=False)]
        df1_impostor = df1_impostor.groupby(df1_impostor.columns.tolist()).apply(lambda x: x.index.tolist()).values.tolist()
        max_pos_impostor = 0
        max_l_impostor = 0
        
        for group in np.arange(len(df1_impostor)):
            if(len(df1_impostor[group]) > max_l_impostor):
                max_l_impostor = len(df1_impostor[group])
                max_pos_impostor = group
            potential_key = keys_impostor1[myuser][df1_impostor[group][0],:int(lenkeys_test1[myuser])]
            if (potential_key == actual_key1).all() == True:
                coincide_impostor_group1 = group
            
        if(max_pos_impostor == coincide_impostor_group1):
            coincide_impostor1 = 2
        if(coincide_impostor_group1 != -1):
            if(max_pos != coincide_impostor_group1):
                coincide_impostor1 = 1
            
        
        for key_imp in keys_impostor1[myuser][:int(numkeys_test1[myuser]),:int(lenkeys_test1[myuser])]:
            if(actual_key1 == key_imp).all() == True:
                impostor_generate = impostor_generate + 1
        far = impostor_generate/(numkeys_test1[myuser])
                
        if(len(df1_impostor)==0):
            rep_keys_impostor1.append(0)
        else:
            rep_keys_impostor1.append(len(df1_impostor[max_pos_impostor])/numkeys_test1[myuser])
        
    
        
        print('Key length: ' + str(len(columns1)))
        print('Repeated keys in val: ' + str(rep_keys1[myuser]))
        print('Repeated keys in test: ' + str(rep_keys_test1[myuser]))
        print('Repeated keys in impostor: ' + str(rep_keys_impostor1[myuser]))
        print('The key is: ')
        print(actual_key1)
        if(coincide1 == 2):
            print('The most common keys coincide')
        elif(coincide1 == 1):
            print('The val key is not the most common in test')
        else:
            print('The val key is not repeated in test')
        if(impostor_generate != 0):
            print('The impostor generates the key: ' + str(far))
            if(coincide_impostor1 == 2):
                print('The key is the most common in impostor')
            elif(coincide_impostor1 == 1):
                print('The key is not the most common in impostor')
            else:
                print('The key is not repeated in impostor')
        else:
            print('The impostor does not generate the key')
    else:
        rep_keys1.append(0)
        rep_keys_test1.append(0)
        rep_keys_impostor1.append(0)
        print('The length of columns is 0')
        
    FAR.append(far)
    FRR.append(1-rep_keys_test1[myuser])
    HTER.append(puntomedio(far,1-rep_keys_test1[myuser]))
    
    ## Count number of 0s and 1s
    num_0 = 0
    num_1 = 0
    for num in actual_key1:
        if(num==0):
            num_0 = num_0 + 1
        elif(num==1):
            num_1 = num_1 + 1
    num_0 = num_0/len(actual_key1)
    num_1 = num_1/len(actual_key1)
    
    ## Uncomment to write in the .xlsx document    
    '''
    hoja.write(row, 0, myuser)
    hoja.write(row, 1, len(columns1))
    hoja.write(row, 2, rep_keys1[myuser])
    hoja.write(row, 3, rep_keys_test1[myuser])
    if(coincide1 == 2):
        hoja.write(row, 4, 'Yes')
    elif(coincide1 == 1):
        hoja.write(row, 4, 'No, but the key is repeated in test' + str(len(df1[max_pos])/numkeys_test1[myuser]))
    else:
        hoja.write(row, 4, 'No')
    if(impostor_generate != 0):
        hoja.write(row, 5, 'Yes')
        if(coincide_impostor1 == 2):
            hoja.write(row, 6, 'The key is the most common')
        elif(coincide_impostor1 == 1):
            hoja.write(row, 6, 'The key is not the most common')
        else:
            hoja.write(row, 6, 'The key is not repeated')
    else:
        hoja.write(row, 5, 'No')
        hoja.write(row, 6, '-')
    hoja.write(row, 7, far)
    hoja.write(row, 8, 1-rep_keys_test1[myuser])
    hoja.write(row, 9, puntomedio(far,1-rep_keys_test1[myuser]))
    hoja.write(row, 10, num_0)
    hoja.write(row, 11, num_1)
    hoja.write(row, 12, str(actual_key1))
    '''
    
    ## Key generation
    keys_list.append(actual_key1)
    hash1 = hashlib.sha3_256(str(actual_key1).encode())
    h = coden.hex_to_bin(hash1.hexdigest())
    k = [int(x) for x in str(h)]
    k = [0]*(256-len(k)) + k
    
    hash_list.append(k)
    
print('\nThe mean FAR is: ' + str(np.mean(FAR)))
print('The mean FRR is: ' + str(np.mean(FRR)))
print('The mean HTER is: ' + str(np.mean(HTER)))
    

'''
libro.close()
'''

## Entropy calculation
entropy_list = []
for i in np.arange(n_users):
    a = keys_list[i]
    n1 = np.count_nonzero(a == 1)
    n0 = len(a) - n1
    entropy_list.append(entropy([n1,n0], base=2))
    
print(np.mean(entropy_list))
print(np.std(entropy_list))


## Confussion Matrix
conf_matrix = np.zeros([n_users,n_users])
ham_matrix = np.zeros([n_users,n_users])
for index, key in enumerate(keys_list):
    for index2, key2 in enumerate(keys_list):
        if(len(key) < len(key2)):
            mylength = len(key)
        else:
            mylength = len(key2)
        ham_matrix[index, index2] = distance.hamming(key[:mylength],key2[:mylength])
        if(key[:mylength] == key2[:mylength]).all() == True:
            conf_matrix[index,index2] = 1
            
plt.imshow(conf_matrix, cmap='hot', interpolation='nearest')
plt.title('Seed Coincidence Heat Map')
plt.xlabel('User seed')
plt.ylabel('User seed')
plt.show()

## Hamming Distance Matrix
plt.imshow(ham_matrix, cmap='hot', vmin=0, vmax=1)
plt.title('Seed Hamming Distance Heat Map')
plt.xlabel('User seed')
plt.ylabel('User seed')
plt.colorbar()
plt.show()

ham_matrix_h = np.zeros([n_users, n_users])
for index, key in enumerate(hash_list):
    for index2, key2 in enumerate(hash_list):
        ham_matrix_h[index, index2] = distance.hamming(key,key2)

plt.imshow(ham_matrix_h, cmap='hot', vmin=0, vmax=1)
plt.title('Key Hamming Distance Heat Map')
plt.xlabel('User keys')
plt.ylabel('User keys')
plt.colorbar()
plt.show()
