import numpy as np
import cv2
import scipy.ndimage as ndimage
from scipy.signal import butter, filtfilt

def detect_artifacts(image):
    """
    检测图像中的伪影。

    参数:
    image (numpy.ndarray): 输入的图像，形状为 (nx, ny)。

    返回:
    bool: 如果检测到伪影，返回 True，否则返回 False。
    """
    # 获取图像的尺寸
    nx, ny = image.shape[0], image.shape[1]

    if np.count_nonzero(image)/(nx*ny)<0.98:
        return None
    
    # 定义一个4x4的均值滤波器
    mean_filter = np.ones((4, 4)) / 16

    # 对图像的第一象限部分进行傅里叶变换
    f_upper_left = np.fft.fft2(image[0:nx//2, 0:ny//2])
    fshift_upper_left = np.fft.fftshift(f_upper_left)
    magnitude_spectrum_upper_left = ndimage.convolve(np.log(np.abs(fshift_upper_left)), mean_filter)

    # 对图像的第四象限部分进行傅里叶变换
    f_lower_right = np.fft.fft2(image[0:nx//2, ny//2:ny])
    fshift_lower_right = np.fft.fftshift(f_lower_right)
    magnitude_spectrum_lower_right = ndimage.convolve(np.log(np.abs(fshift_lower_right)), mean_filter)

    # 计算两个频谱的差异矩阵
    matrix = (magnitude_spectrum_upper_left / np.mean(magnitude_spectrum_upper_left) -
              magnitude_spectrum_lower_right / np.mean(magnitude_spectrum_lower_right))

    # 获取图像的中心点
    center = (nx//4, ny//4)
    
    # 定义角度步长为2度
    angles = np.arange(0, 180, 2)
    
    # 存储每个角度的均值
    mean_values = []

    for angle in angles:
        # 将角度转换为弧度
        theta = np.radians(angle)
        
        # 计算直线上的点
        points = []
        for r in range(-nx//4, nx//4):
            x = int(center[0] + r * np.cos(theta))
            y = int(center[1] + r * np.sin(theta))
            
            # 确保点在矩阵范围内
            if 0 <= x < 512 and 0 <= y < 512:
                points.append(matrix[x, y])
        
        # 计算均值
        mean_value = np.mean(points)
        mean_values.append(mean_value)

    # 设置高通滤波器的截止频率
    cutoff_frequency = 0.1  # 例如，截止频率为0.1

    # 设计巴特沃斯高通滤波器
    order = 2  # 滤波器阶数
    nyquist_freq = 0.5  # Nyquist频率
    normal_cutoff = cutoff_frequency / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    # 应用高通滤波器
    filtered_signal = filtfilt(b, a, np.array(mean_values))
    
    # 计算滤波后信号的标准差
    std_dev = np.std(filtered_signal)

    # 计算矩阵中每个点与前一个点的差值的绝对值
    differences = np.abs(np.diff(filtered_signal))

    # remove 90°
    differences[44] = 0
    differences[45] = 0
    
    # 检查差值是否大于标准差的3倍
    result = differences > 5 * std_dev

    # 如果有任何差值大于标准差的3倍，则返回 True
    if np.any(result):
        return True
    else:
        return False

# 示例用法
# image = cv2.imread('example.png', cv2.IMREAD_GRAYSCALE)
# artifact_detected = detect_artifacts(image)
# print(artifact_detected)