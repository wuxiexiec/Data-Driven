import glob
from tensorflow.keras.preprocessing.image import load_img,save_img,img_to_array,array_to_img


# 这里写你的数据集的地址，写到train/文件夹下
dataset_path = 'C:/Work/facades/facades/train/'

# 这里写把剪开的两张图片分别保存在什么位置
save_data_x_directory = 'C:/Work/facades/facades2/train/facades/'
save_data_y_directory = 'C:/Work/facades/facades2/train/images/'



if __name__ == '__main__':
    i = 0
    for name in glob.glob(dataset_path + '*'):
        i = i + 1
        image = load_img(name)
        image = img_to_array(image)
        # 256:512 是 左边的图片，0:256 是右边的图片，根据你的数据集来调整
        data_x = image[:,256:512,:]
        data_y = image[:,0:256,:]
        data_x = array_to_img(data_x)
        data_y = array_to_img(data_y)
        save_img(save_data_x_directory+str(i)+'.jpg',data_x)
        save_img(save_data_y_directory+str(i)+'.jpg',data_y)
