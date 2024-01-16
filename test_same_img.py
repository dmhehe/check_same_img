import json
import cv2
from PIL import Image
import numpy as np
import simui
from simui import sim_class, sim_meth, sim_func, init_sim_global


def lerp(start, end, min_score, max_score, my_score):
    
    alpha = 0
    if my_score > max_score:
        alpha = 1
    elif my_score < min_score:
        alpha = 0
    else:
        alpha = (my_score-min_score)/(max_score-min_score)
        
    return start + alpha * (end - start)




def check_img_same(image1, image2, threshold1=1000, threshold2=0.998, threshold3=1000):
    w1, h1 = get_image_size(image1)
    w2, h2 = get_image_size(image2)
    
    w_r = w1/w2
    h_r = h1/h2
    
    wh_r1 = min(w1,h1)/max(w1,h1)
    wh_r2 = min(w2,h2)/max(w2,h2)
    
    #大小相差过大，认为不同
    if w_r > 3 or w_r < 1.0/3 or h_r > 3 or h_r < 1.0/3: 
        return False
    
    #大小比例相差过大，认为不同
    if abs(wh_r1-wh_r2)>0.1:
        return False
    
    
    img1 = Image.open(image1)
    img_array1 = np.array(img1)
    
    img2 = Image.open(image2)
    img_array2 = np.array(img2)

    
    #一个透明，一个不透明，认为不同
    if img_array1.shape[-1] != img_array2.shape[-1]:
        return False
        
    
    size = (w1 + w2 + h1 + h2)/4
    
    curr_threshold = lerp(threshold1, threshold1*0.5, 50, 512, size)
    
    sim1 = jfc_cal_image_sim(image1, image2)
    sim2 = yszft_cal_image_sim(image1, image2)
    if sim1 > 10000:
        return False
    
    if sim2 < 0.1:
        return False
    
    
    if sim1 < curr_threshold:
        return True
    
    
    if sim2 > threshold2:
        return True
    
    sim3 = dct_ycrcb_cal_image_sim(image1, image2)
    if sim3 < threshold3:
        return True
    
    return False





def get_y_cr_cb(img1):
    bgr_img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
    ycrcb_image1 = cv2.cvtColor(bgr_img1, cv2.COLOR_BGR2YCrCb)
    Y1, Cr1, Cb1 = cv2.split(ycrcb_image1)
    return Y1, Cr1, Cb1

def ycrcb_cal_image_sim(image1_path, image2_path):
    img1 = cv_read_img(image1_path)
    img2 = cv_read_img(image2_path)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 分别提取Y, Cr, Cb通道
    Y1, Cr1, Cb1 = get_y_cr_cb(img1)
    Y2, Cr2, Cb2 = get_y_cr_cb(img2)

    # 分别计算Y, Cr, Cb通道的均方差
    mse_Y = mse(Y1, Y2)
    mse_Cr = mse(Cr1, Cr2)
    mse_Cb = mse(Cb1, Cb2)
    
    return mse_Y*2 + mse_Cr + mse_Cb



def dct_transform(image):
    # 将图像转换为浮点数，以进行DCT
    image_float = np.float32(image)
    # 执行DCT
    dct = cv2.dct(image_float)
    return dct

def compare_images_dct(image1, image2, top_left_corner=(50, 50)):
    # 应用DCT变换
    float_channel1 = np.float32(image1)
    float_channel2 = np.float32(image2)
    dct1 = dct_transform(float_channel1)
    dct2 = dct_transform(float_channel2)

    # 提取左上角的DCT系数
    dct1_top_left = dct1[:top_left_corner[0], :top_left_corner[1]]
    dct2_top_left = dct2[:top_left_corner[0], :top_left_corner[1]]

    # 计算均方误差
    mse = np.mean((dct1_top_left - dct2_top_left) ** 2)
    return mse


def dct_ycrcb_cal_image_sim(image1_path, image2_path):
    img1 = cv_read_img(image1_path)
    img2 = cv_read_img(image2_path)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 分别提取Y, Cr, Cb通道
    Y1, Cr1, Cb1 = get_y_cr_cb(img1)
    Y2, Cr2, Cb2 = get_y_cr_cb(img2)

    top_left_corner = (int(img1.shape[1]*0.25+0.5), int(img1.shape[0]*0.25+0.5))
    similarity1 = compare_images_dct(Y1, Y2, top_left_corner)
    similarity2 = compare_images_dct(Cr1, Cr2, top_left_corner)
    similarity3 = compare_images_dct(Cb1, Cb2, top_left_corner)
    return (similarity1*2 + similarity2 + similarity3)/4.0



def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size[0], img.size[1]  # 返回的是一个元组 (宽度, 高度)
    
    

def yszft_cal_image_sim(image1_path, image2_path):
    img1 = cv_read_img(image1_path)
    img2 = cv_read_img(image2_path)
    hist1 = calculate_histogram_with_alpha(img1)
    hist2 = calculate_histogram_with_alpha(img2)
    similarity = compare_histograms(hist1, hist2)
    return similarity


def calculate_histogram_with_alpha(image, bins=256):
    
    # 检查图像是否包含alpha通道
    if image.shape[2] == 4:
        # 分别计算RGBA通道的直方图
        histogram = [cv2.calcHist([image], [i], None, [bins], [0, 256]) for i in range(4)]
    else:
        # 如果没有alpha通道，退回到标准的RGB直方图
        histogram = [cv2.calcHist([image], [i], None, [bins], [0, 256]) for i in range(3)]

    # 归一化直方图
    norm_histogram = [cv2.normalize(h, None) for h in histogram]
    return np.concatenate(norm_histogram)

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    similarity = cv2.compareHist(hist1, hist2, method)
    return similarity







def mse(imageA, imageB):
    # 计算两幅图像的均方差
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def jfc_cal_image_sim(image1, image2):
    # 读取图像
    img1 = cv_read_img(image1)
    img2 = cv_read_img(image2)

    # 如果两幅图像尺寸不同，则调整第二幅图像的尺寸以匹配第一幅图像
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 确保通道数一致
    if img1.shape[2] != img2.shape[2]:
        raise ValueError("Images must have the same number of channels to compare.")

    # 计算均方差
    return mse(img1, img2)








def cv_read_img(path):
    # 使用Pillow读取图像
    img = Image.open(path).convert("RGBA")

    # 将Pillow图像转换为numpy数组
    img_array = np.array(img)

    # 如果图像是RGBA格式，转换为BGRA格式以供OpenCV使用
    if img_array.shape[-1] == 4:
        # OpenCV使用BGRA顺序
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
    else:
        # 对于RGB图像，转换为BGR格式
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    return img_array



def cv_write_img(cv_img, path):
    # OpenCV图片是BGR/ BGRA格式，首先转换为RGB/ RGBA
    if cv_img.shape[-1] == 4:
        # 对于带有Alpha通道的图像
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
    else:
        # 对于标准BGR图像
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # 将OpenCV图像转换为Pillow图像
    img = Image.fromarray(cv_img)

    # 使用Pillow保存图像
    img.save(path)



attr_list2 = [
    ["aaa", "全局属性aaa", "str", {"iPrior": 4}],
    ["bbb", "全局属性bbb",  "str", {"iPrior": 4}],
]

gg = init_sim_global("gg_DDD", attr_list2)


@sim_func("test", {"iPrior": 0})
def main():
    image1_path = gg.aaa
    image2_path = gg.bbb
    print(check_img_same(image1_path, image2_path, threshold1=1000, threshold2=0.998))
    print("1111111111111111",  ycrcb_cal_image_sim(image1_path, image2_path))
    print("2222222222222222222",  dct_ycrcb_cal_image_sim(image1_path, image2_path))
    
    
simui.show_ui("test ui")