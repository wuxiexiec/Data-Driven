import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, concatenate
from tensorflow.keras.models import Model

# Read file
mer_information = pd.read_excel('./Input_file.xlsx', sheet_name=0)
mer_information = np.array(mer_information.values[1:, :])

num_mer = int(len(mer_information[0])/6)
num_max_depth = len(mer_information)

mer_coo = np.empty(shape=(num_max_depth*num_mer, 3))
mer_data = np.empty(shape=(num_max_depth*num_mer, 2))
for i in range(0, num_mer):
    mer_coo[i*num_max_depth:i*num_max_depth+num_max_depth, :] = mer_information[0:num_max_depth, i*6:i*6+3]
    mer_data[i*num_max_depth:i*num_max_depth+num_max_depth, :] = mer_information[0:num_max_depth, i*6+3:i*6+5]

# Build all coordinates
print("Build all coordinates")
all_x = np.arange(0.125, 13, 0.25)
all_y = np.arange(0.125, 26, 0.25)
all_z = np.arange(0.68, 5.32, 0.04)
all_coo = np.empty(shape=(len(all_x)*len(all_y)*len(all_z), 3))
hangshu = 0
for i in range(0, len(all_x)):
    for j in range(0, len(all_y)):
        for k in range(0, len(all_z)):
            all_coo[hangshu, 0] = all_x[i]
            all_coo[hangshu, 1] = all_y[j]
            all_coo[hangshu, 2] = all_z[k]
            hangshu = hangshu + 1

# Build xy plane coordinates
all_x = np.arange(0.125, 13, 0.25)
all_y = np.arange(0.125, 26, 0.25)
xy_all_coo = np.empty(shape=(len(all_x)*len(all_y), 2))
hangshu = 0
for i in range(0, len(all_x)):
    for j in range(0, len(all_y)):
        xy_all_coo[hangshu, 0] = all_x[i]
        xy_all_coo[hangshu, 1] = all_y[j]
        hangshu = hangshu + 1

# Build xz plane coordinates
all_x = np.arange(0.125, 13, 0.25)
all_z = np.arange(0.68, 5.32, 0.04)
xz_all_coo = np.empty(shape=(len(all_x)*len(all_z), 2))
hangshu = 0
for i in range(0, len(all_x)):
    for j in range(0, len(all_z)):
        xz_all_coo[hangshu, 0] = all_x[i]
        xz_all_coo[hangshu, 1] = all_z[j]
        hangshu = hangshu + 1

# Build yz plane coordinates
all_y = np.arange(0.125, 26, 0.25)
all_z = np.arange(0.68, 5.32, 0.04)
yz_all_coo = np.empty(shape=(len(all_y)*len(all_z), 2))
hangshu = 0
for i in range(0, len(all_y)):
    for j in range(0, len(all_z)):
        yz_all_coo[hangshu, 0] = all_y[i]
        yz_all_coo[hangshu, 1] = all_z[j]
        hangshu = hangshu + 1

print("build GCF coordinates")

GCF_x = np.arange(0.125, 13, 0.5)
GCF_y = np.arange(0.125, 26, 0.5)
xy_GCF_coo = np.empty(shape=(len(GCF_x)*len(GCF_y), 2))
hangshu = 0
for i in range(0, len(GCF_x)):
    for j in range(0, len(GCF_y)):
        xy_GCF_coo[hangshu, 0] = GCF_x[i]
        xy_GCF_coo[hangshu, 1] = GCF_y[j]
        hangshu = hangshu + 1


GCF_x = np.arange(0.125, 26, 0.5)
GCF_z = np.arange(0.68, 5.32, 0.04)
xz_GCF_coo = np.empty(shape=(len(GCF_x)*len(GCF_z), 2))
hangshu = 0
for i in range(0, len(GCF_x)):
    for j in range(0, len(GCF_z)):
        xz_GCF_coo[hangshu, 0] = GCF_x[i]
        xz_GCF_coo[hangshu, 1] = GCF_z[j]
        hangshu = hangshu + 1


GCF_y = np.arange(0.125, 13, 0.5)
GCF_z = np.arange(0.68, 5.32, 0.04)
yz_GCF_coo = np.empty(shape=(len(GCF_y)*len(GCF_z), 2))
hangshu = 0
for i in range(0, len(GCF_y)):
    for j in range(0, len(GCF_z)):
        yz_GCF_coo[hangshu, 0] = GCF_y[i]
        yz_GCF_coo[hangshu, 1] = GCF_z[j]
        hangshu = hangshu + 1

# ------------------------------------------------------------
sof_x = 30
sof_y = 30
sof_z = 0.04


print("Genrate xy plane GCFs field")
xy_nn_Mat = np.empty(shape=(len(xy_GCF_coo), len(xy_GCF_coo)))
for i in range(0, len(xy_GCF_coo)):
    d_x = np.abs(xy_GCF_coo[:, 0] - xy_GCF_coo[i, 0])
    d_y = np.abs(xy_GCF_coo[:, 1] - xy_GCF_coo[i, 1])

    xishu = (1 + 4 * (d_x / sof_x)) * (1 + 4 * (d_y / sof_y)) \
            * np.exp(-4 * ((d_x / sof_x) + (d_y / sof_y) ))
    xy_nn_Mat[i, :] = xishu

from sklearn.decomposition import PCA
n_components = 100
pca = PCA(n_components=n_components, svd_solver='full')
xy_nn_Mat = pca.fit_transform(xy_nn_Mat)

xy_Nn_Mat = np.empty(shape=(len(xy_all_coo), len(xy_GCF_coo)))
for i in range(0, len(xy_all_coo)):
    d_x = np.abs(xy_GCF_coo[:, 0] - xy_all_coo[i, 0])
    d_y = np.abs(xy_GCF_coo[:, 1] - xy_all_coo[i, 1])

    xishu = (1 + 4 * (d_x / sof_x)) * (1 + 4 * (d_y / sof_y)) \
            * np.exp(-4 * ((d_x / sof_x) + (d_y / sof_y)))
    xy_Nn_Mat[i, :] = xishu
xy_Nn_Mat = pca.transform(xy_Nn_Mat)
tiao_xy_Nn_Mat = xy_Nn_Mat.reshape(len(all_x), len(all_y), n_components)
xy_Mat = np.empty(shape=(len(all_x), len(all_y), len(all_z), n_components))
for i in range(0, len(all_z)):
    xy_Mat[:, :, i, :] = tiao_xy_Nn_Mat[:, :, :]
xy_Mat = xy_Mat.reshape(-1, n_components)

xy_mer_Mat = np.empty(shape=(len(mer_coo), len(xy_GCF_coo)))
for i in range(0, len(mer_coo)):
    d_x = np.abs(xy_GCF_coo[:, 0] - mer_coo[i, 0])
    d_y = np.abs(xy_GCF_coo[:, 1] - mer_coo[i, 1])

    xishu = (1 + 4 * (d_x / sof_x)) * (1 + 4 * (d_y / sof_y)) \
            * np.exp(-4 * ((d_x / sof_x) + (d_y / sof_y)))
    xy_mer_Mat[i, :] = xishu
xy_mer_Mat = pca.transform(xy_mer_Mat)

print("Genrate yz plane GCFs field")
xz_nn_Mat = np.empty(shape=(len(xz_GCF_coo), len(xz_GCF_coo)))
for i in range(0, len(xz_GCF_coo)):
    d_x = np.abs(xz_GCF_coo[:, 0] - xz_GCF_coo[i, 0])
    d_z = np.abs(xz_GCF_coo[:, 1] - xz_GCF_coo[i, 1])

    xishu = (1 + 4 * (d_x / sof_x)) * (1 + 4 * (d_z / sof_z)) \
            * np.exp(-4 * ((d_x / sof_x) + (d_z / sof_z) ))
    xz_nn_Mat[i, :] = xishu

from sklearn.decomposition import PCA
n_components = 200
pca = PCA(n_components=n_components, svd_solver='full')
xz_nn_Mat = pca.fit_transform(xz_nn_Mat)

xz_Nn_Mat = np.empty(shape=(len(xz_all_coo), len(xz_GCF_coo)))
for i in range(0, len(xz_all_coo)):
    d_x = np.abs(xz_GCF_coo[:, 0] - xz_all_coo[i, 0])
    d_z = np.abs(xz_GCF_coo[:, 1] - xz_all_coo[i, 1])

    xishu = (1 + 4 * (d_x / sof_x)) * (1 + 4 * (d_z / sof_z)) \
            * np.exp(-4 * ((d_x / sof_x) + (d_z / sof_z)))
    xz_Nn_Mat[i, :] = xishu
xz_Nn_Mat = pca.transform(xz_Nn_Mat)
tiao_xz_Nn_Mat = xz_Nn_Mat.reshape(len(all_x), len(all_z), n_components)
xz_Mat = np.empty(shape=(len(all_x), len(all_y), len(all_z), n_components))
for i in range(0, len(all_y)):
    xz_Mat[:, i, :, :] = tiao_xz_Nn_Mat[:, :, :]
xz_Mat = xz_Mat.reshape(-1, n_components)

xz_mer_Mat = np.empty(shape=(len(mer_coo), len(xz_GCF_coo)))
for i in range(0, len(mer_coo)):
    d_x = np.abs(xz_GCF_coo[:, 0] - mer_coo[i, 0])
    d_z = np.abs(xz_GCF_coo[:, 1] - mer_coo[i, 2])

    xishu = (1 + 4 * (d_x / sof_x)) * (1 + 4 * (d_z / sof_z)) \
            * np.exp(-4 * ((d_x / sof_x) + (d_z / sof_z)))
    xz_mer_Mat[i, :] = xishu
xz_mer_Mat = pca.transform(xz_mer_Mat)


# print("Genrate yz plane GCFs field")
yz_nn_Mat = np.empty(shape=(len(yz_GCF_coo), len(yz_GCF_coo)))
for i in range(0, len(yz_GCF_coo)):
    d_y = np.abs(yz_GCF_coo[:, 0] - yz_GCF_coo[i, 0])
    d_z = np.abs(yz_GCF_coo[:, 1] - yz_GCF_coo[i, 1])

    xishu = (1 + 4 * (d_y / sof_y)) * (1 + 4 * (d_z / sof_z)) \
            * np.exp(-4 * ((d_y / sof_y) + (d_z / sof_z) ))
    yz_nn_Mat[i, :] = xishu

from sklearn.decomposition import PCA
n_components = 200
pca = PCA(n_components=n_components, svd_solver='full')
yz_nn_Mat = pca.fit_transform(yz_nn_Mat)

yz_Nn_Mat = np.empty(shape=(len(yz_all_coo), len(yz_GCF_coo)))
for i in range(0, len(yz_all_coo)):
    d_y = np.abs(yz_GCF_coo[:, 0] - yz_all_coo[i, 0])
    d_z = np.abs(yz_GCF_coo[:, 1] - yz_all_coo[i, 1])

    xishu = (1 + 4 * (d_y / sof_y)) * (1 + 4 * (d_z / sof_z)) \
            * np.exp(-4 * ((d_y / sof_y) + (d_z / sof_z)))
    yz_Nn_Mat[i, :] = xishu
yz_Nn_Mat = pca.transform(yz_Nn_Mat)
tiao_yz_Nn_Mat = yz_Nn_Mat.reshape(len(all_y), len(all_z), n_components)
yz_Mat = np.empty(shape=(len(all_x), len(all_y), len(all_z), n_components))
for i in range(0, len(all_x)):
    yz_Mat[i, :, :, :] = tiao_yz_Nn_Mat[:, :, :]
yz_Mat = yz_Mat.reshape(-1, n_components)

yz_mer_Mat = np.empty(shape=(len(mer_coo), len(yz_GCF_coo)))
for i in range(0, len(mer_coo)):
    d_y = np.abs(yz_GCF_coo[:, 0] - mer_coo[i, 0])
    d_z = np.abs(yz_GCF_coo[:, 1] - mer_coo[i, 2])

    xishu = (1 + 4 * (d_y / sof_y)) * (1 + 4 * (d_z / sof_z)) \
            * np.exp(-4 * ((d_y / sof_y) + (d_z / sof_z)))
    yz_mer_Mat[i, :] = xishu
yz_mer_Mat = pca.transform(yz_mer_Mat)

# ------------------------------叠加xy和xz和yz平面特征训练qc模型------------------------------
#
xunlian_x = np.hstack((xy_mer_Mat, xz_mer_Mat, yz_mer_Mat))
xunlian_y = mer_data[:, 0]

test_x = np.hstack((xy_Mat, xz_Mat, yz_Mat))


# 使用神经网络建模
min = np.min(xunlian_y)
max = np.max(xunlian_y)
xunlian_y = (xunlian_y - min) / (max - min)

kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=666)
input1 = Input(shape=(xunlian_x.shape[-1],))
layer1 = Dense(64, activation="relu",
               kernel_initializer=kernel_initializer)(input1)
# dropout1 = Dropout(0.2)(layer1)
layer2 = Dense(64, activation="relu",
               kernel_initializer=kernel_initializer)(layer1)
# dropout2 = Dropout(0.2)(layer2)
layer3 = Dense(64, activation="relu",
               kernel_initializer=kernel_initializer)(layer2)
combined = concatenate([layer3, input1])
# dropout3 = Dropout(0.2)(combined)
layer4 = Dense(128, activation="relu",
               kernel_initializer=kernel_initializer)(combined)
layer5 = Dense(1, activation="linear")(layer4)
model = Model(inputs=input1, outputs=layer5)

model.compile(optimizer=tf.keras.optimizers.Nadam(10 ** (-3)),
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

history = model.fit(xunlian_x, xunlian_y, batch_size=25, epochs=500, verbose=1)

# 预测
pre_test = np.array(model.predict(test_x)).reshape(-1, 1)
pre_test = (pre_test * (max - min)) + min
tf.keras.backend.clear_session()

# 保存结果
cun = np.empty(shape=(len(test_x), 4))
cun[:, 0:3] = all_coo
cun[:, [3]] = pre_test
np.save(file="newANN新案例预测结果.npy", arr=cun)


