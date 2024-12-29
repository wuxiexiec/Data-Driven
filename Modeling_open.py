import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, concatenate
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA

def read_file(file_name, sheet_name):
    mer_information = pd.read_excel(file_name, sheet_name=sheet_name)
    mer_information = np.array(mer_information.values[1:, :])

    num_mer = int(len(mer_information[0])/6)
    num_max_depth = len(mer_information)

    mer_coo = np.empty(shape=(num_max_depth*num_mer, 3))
    mer_data = np.empty(shape=(num_max_depth*num_mer, 2))
    for i in range(0, num_mer):
        mer_coo[i*num_max_depth:i*num_max_depth+num_max_depth, :] = mer_information[0:num_max_depth, i*6:i*6+3]
        mer_data[i*num_max_depth:i*num_max_depth+num_max_depth, :] = mer_information[0:num_max_depth, i*6+3:i*6+5]
    # Delete nan
    mer_coo = mer_coo[~np.isnan(mer_coo).any(axis=1), :]
    mer_data = mer_data[~np.isnan(mer_data).any(axis=1), :]
    return mer_coo, mer_data

def build_all_coo(x_set, y_set, z_set):
    all_x = np.arange(x_set[0], x_set[1], x_set[2])
    all_y = np.arange(y_set[0], y_set[1], y_set[2])
    all_z = np.arange(z_set[0], z_set[1], z_set[2])
    all_coo = np.empty(shape=(len(all_x)*len(all_y)*len(all_z), 3))
    num = 0
    for i in range(0, len(all_x)):
        for j in range(0, len(all_y)):
            for k in range(0, len(all_z)):
                all_coo[num, 0] = all_x[i]
                all_coo[num, 1] = all_y[j]
                all_coo[num, 2] = all_z[k]
                num += 1
    return all_x, all_y, all_z, all_coo

# Build plane coordinates
def build_plane_coo(set_1, set_2):
    all_1 = np.arange(set_1[0], set_1[1], set_1[2])
    all_2 = np.arange(set_2[0], set_2[1], set_2[2])
    plane_coo = np.empty(shape=(len(all_1)*len(all_2), 2))
    num = 0
    for i in range(0, len(all_1)):
        for j in range(0, len(all_2)):
            plane_coo[num, 0] = all_1[i]
            plane_coo[num, 1] = all_2[j]
            num += 1
    return plane_coo

def build_Mat(GCF_coo, n_components, plane_all_coo, all_x, all_y, all_z, sof_1, sof_2, mer_coo, mark_num):
    nn_Mat = np.empty(shape=(len(GCF_coo), len(GCF_coo)))
    for i in range(0, len(GCF_coo)):
        d_1 = np.abs(GCF_coo[:, 0] - GCF_coo[i, 0])
        d_2 = np.abs(GCF_coo[:, 1] - GCF_coo[i, 1])

        xishu = (1 + 4 * (d_1 / sof_1)) * (1 + 4 * (d_2 / sof_2)) \
                * np.exp(-4 * ((d_1 / sof_1) + (d_2 / sof_2)))
        nn_Mat[i, :] = xishu

    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit_transform(nn_Mat)

    Nn_Mat = np.empty(shape=(len(plane_all_coo), len(GCF_coo)))
    for i in range(0, len(plane_all_coo)):
        d_1 = np.abs(GCF_coo[:, 0] - plane_all_coo[i, 0])
        d_2 = np.abs(GCF_coo[:, 1] - plane_all_coo[i, 1])

        xishu = (1 + 4 * (d_1 / sof_1)) * (1 + 4 * (d_2 / sof_2)) \
                * np.exp(-4 * ((d_1 / sof_1) + (d_2 / sof_2)))
        Nn_Mat[i, :] = xishu
    Nn_Mat = pca.transform(Nn_Mat)

    all_Mat = np.empty(shape=(len(all_x), len(all_y), len(all_z), n_components))

    if mark_num == 0:
        tiao_Nn_Mat = Nn_Mat.reshape(len(all_x), len(all_y), n_components)
        for i in range(0, len(all_z)):
            all_Mat[:, :, i, :] = tiao_Nn_Mat[:, :, :]
        all_Mat = all_Mat.reshape(-1, n_components)

    elif mark_num == 1:
        tiao_Nn_Mat = Nn_Mat.reshape(len(all_x), len(all_z), n_components)
        for i in range(0, len(all_y)):
            all_Mat[:, i, :, :] = tiao_Nn_Mat[:, :, :]
        all_Mat = all_Mat.reshape(-1, n_components)

    elif mark_num == 2:
        tiao_Nn_Mat = Nn_Mat.reshape(len(all_y), len(all_z), n_components)
        for i in range(0, len(all_x)):
            all_Mat[i, :, :, :] = tiao_Nn_Mat[:, :, :]
        all_Mat = all_Mat.reshape(-1, n_components)

    mer_Mat = np.empty(shape=(len(mer_coo), len(GCF_coo)))
    for i in range(0, len(mer_coo)):
        d_1 = np.abs(GCF_coo[:, 0] - mer_coo[i, 0])
        d_2 = np.abs(GCF_coo[:, 1] - mer_coo[i, 1])

        xishu = (1 + 4 * (d_1 / sof_1)) * (1 + 4 * (d_2 / sof_2)) \
                * np.exp(-4 * ((d_1 / sof_1) + (d_2 / sof_2)))
        mer_Mat[i, :] = xishu
    mer_Mat = pca.transform(mer_Mat)

    return all_Mat, mer_Mat

def build_and_train_model(train_x, train_y):
    # Define the model
    kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=666)
    input1 = Input(shape=(train_x.shape[-1],))
    layer1 = Dense(64, activation="relu", kernel_initializer=kernel_initializer)(input1)
    layer2 = Dense(64, activation="relu", kernel_initializer=kernel_initializer)(layer1)
    layer3 = Dense(64, activation="relu", kernel_initializer=kernel_initializer)(layer2)
    combined = concatenate([layer3, input1])
    layer4 = Dense(128, activation="relu", kernel_initializer=kernel_initializer)(combined)
    layer5 = Dense(1, activation="relu")(layer4)
    model = Model(inputs=input1, outputs=layer5)

    model.compile(optimizer=tf.keras.optimizers.Nadam(10 ** (-3)), loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    # Train the model
    model.fit(train_x, train_y, batch_size=25, epochs=500, verbose=1)
    return model

def main():
    sof_x, sof_y, sof_z = 30, 30, 0.04
    mer_coo, mer_data = read_file('./汇总.xlsx', 2)

    x_min, x_max, x_step, x_GCF_step = 0.125, 13, 0.25, 0.5
    y_min, y_max, y_step, y_GCF_step = 0.125, 26, 0.25, 0.5
    z_min, z_max, z_step, z_GCF_step = 0.68, 5.32, 0.04, 0.04

    x_set = [x_min, x_max, x_step]
    y_set = [y_min, y_max, y_step]
    z_set = [z_min, z_max, z_step]

    all_x, all_y, all_z, all_coo = build_all_coo(x_set, y_set, z_set)

    xy_all_coo = build_plane_coo(x_set, y_set)
    xz_all_coo = build_plane_coo(x_set, z_set)
    yz_all_coo = build_plane_coo(y_set, z_set)

    xy_GCF_coo = build_plane_coo([x_min, x_max, x_GCF_step], [y_min, y_max, y_GCF_step])
    xz_GCF_coo = build_plane_coo([x_min, x_max, x_GCF_step], [z_min, z_max, z_GCF_step])
    yz_GCF_coo = build_plane_coo([y_min, y_max, y_GCF_step], [z_min, z_max, z_GCF_step])

    print("Building GCFs xy")
    xy_all_Mat, xy_mer_Mat = build_Mat(xy_GCF_coo, 300, xy_all_coo, all_x, all_y, all_z, sof_x, sof_y, mer_coo, 0)
    print("Building GCFs xz")
    xz_all_Mat, xz_mer_Mat = build_Mat(xz_GCF_coo, 100, xz_all_coo, all_x, all_y, all_z, sof_x, sof_z, mer_coo, 1)
    print("Building GCFs yz")
    yz_all_Mat, yz_mer_Mat = build_Mat(yz_GCF_coo, 100, yz_all_coo, all_x, all_y, all_z, sof_y, sof_z, mer_coo, 2)

    # Training and Test set
    train_x = np.hstack((xy_mer_Mat, xz_mer_Mat, yz_mer_Mat))
    train_y = mer_data[:, 0]
    test_x = np.hstack((xy_all_Mat, xz_all_Mat, yz_all_Mat))

    min = np.min(train_y)
    max = np.max(train_y)
    train_y = (train_y - min) / (max - min)

    model = build_and_train_model(train_x, train_y)

    # predict
    pre_test = np.array(model.predict(test_x)).reshape(-1, 1)
    pre_test = (pre_test * (max - min)) + min

    # save
    np.save(file="predict.npy", arr=pre_test)

if __name__ == '__main__':
    main()


