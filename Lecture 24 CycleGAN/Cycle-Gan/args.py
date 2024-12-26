# 训练多少轮和batch_size
batch_size = 1
epochs = 5

# 这里写你的x,y两种风格的图片
# 训练数据的路径，
# 数据集路径如何写？
# 把所有图片装在一个文件夹，然后把地址写为它的上级文件夹
# 比如把所有图片放在D:/a/b/下，那么x_path写成D:/a。
# 数据集在这里下载，用迅雷复制链接下载比较快
# 用迅雷复制链接下载更快
# http://efrosgans.eecs.berkeley.edu/cyclegta/
# https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
# 下载解压完成后按上述写路径就可以
x_path = 'C:/Work/gandataset2/appl'
y_path = 'C:/Work/gandataset2/orang'

# 这里写测试图片的路径，每轮训练完会利用测试图片绘制图像
# 选一张 x domain和 y domain 的图像，比如我训练的是苹果转橘子，这里就用一张苹果一张橘子
domain_x_test_img_path = 'apple.png'
domain_y_test_img_path = 'orange.png'

# 图片大小
image_shape = (28,28,3)

# 下面是训练超参数
# code_dim是U_net的将图像编码为多长的code
code_dim = 256

# 学习率，分别代表Gx2y,Gy2x,Discriminator的学习率，自己可以多调试
gx2y_learning_rate = 0.001
gy2x_learning_rate = 0.001
d_learning_rate = 0.001


# 控制损失函数项，自己多调试
cycle_consistency_loss = 10
identity_loss_weight = 1