import numpy as np

def load_dataset(data_dict):
    time = []
    data = []
    I_p = []
    for _, contents in data_dict.items():
        time.append(contents['time'])
        data.append(contents['data'])
        I_p.append(contents['I_p'])
    
    data = np.array(data)
    x_data = data[:-1,:]
    y_data = data[1:,:]
    dt_data = (time[1] - time[0]) * np.ones((data.shape[0] -1 , 1))
    # I_p = np.reshape(np.array(I_p)[:-1], (-1,1))
    u1_data = np.concatenate((dt_data, np.reshape(np.array(I_p)[:-1], (-1,1)), np.reshape(np.array(I_p)[1:], (-1,1))), axis = 1)
    u2_data = np.concatenate((dt_data, np.reshape(np.array(I_p)[1:], (-1,1)), np.reshape(np.concatenate((np.array(I_p)[2:], np.array([I_p[0]]))), (-1,1))), axis = 1)
    return x_data, y_data, u1_data, u2_data
