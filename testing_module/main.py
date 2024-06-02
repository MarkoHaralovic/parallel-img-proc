from Tester import FunctionTester
from functions import *
from skimage import data, img_as_float
import cupy as cp
import numpy as np
import csv

image1 = img_as_float(data.immunohistochemistry())
image2 = img_as_float(data.astronaut())

image_shape = image1.shape

images = [image1, image2]
kernel = np.ones((5, 5)) / 25

def dummy_classifier(window):
    return np.random.choice([False, True])

denoising_tester = FunctionTester()
denoising_tester.init(images, time_synchronize, perform_nl_means_denoising)

images = [image1, image2] * 100 # 200 images

fft_tester = FunctionTester()
fft_tester.init(images, time_synchronize, perform_fft)

ifft_tester = FunctionTester()
ifft_tester.init(images, time_synchronize, perform_ifft)

wavelet_tester = FunctionTester()
wavelet_tester.init(images, time_synchronize, perform_wavelet_transform)

convolution_tester = FunctionTester()
convolution_tester.init(images, time_synchronize, lambda img: perform_convolution(img, kernel))


def image_registration_wrapper(image):
    return perform_image_registration(image1, image)

registration_tester = FunctionTester()
registration_tester.init(images, time_synchronize, image_registration_wrapper)

inpainting_tester = FunctionTester()
mask = np.zeros(image1.shape, dtype=bool)
mask[100:150, 100:150] = True
inpainting_tester.init(images[0:5], time_synchronize, lambda img: perform_inpainting(img, mask))

mean_std_tester = FunctionTester()
mean_std_tester.init(images, time_synchronize, mean_and_std)

def calculate_and_store_times(tester, device_func, operation_name, device_name,image_shape):
    print(f"running {operation_name} on {device_name}...\n")
    execution_time = tester.calculate_time(device_func, operation_name)
    num_images = len(tester.images) 
    image_shape = image_shape
    return [operation_name, execution_time, device_name, num_images, image_shape]

data = []

data.append(calculate_and_store_times(fft_tester, np.array, "FFT", "CPU"))
data.append(calculate_and_store_times(ifft_tester, np.array, "IFFT", "CPU"))
data.append(calculate_and_store_times(wavelet_tester, np.array, "Wavelet Transform", "CPU"))
data.append(calculate_and_store_times(convolution_tester, np.array, "Convolution", "CPU"))
data.append(calculate_and_store_times(denoising_tester, np.array, "NL-means Denoising", "CPU"))
data.append(calculate_and_store_times(registration_tester, np.array, "Image Registration", "CPU"))
data.append(calculate_and_store_times(inpainting_tester, np.array, "Inpainting", "CPU"))
data.append(calculate_and_store_times(mean_std_tester, np.array, "Mean and Std", "CPU"))

data.append(calculate_and_store_times(fft_tester, cp.array, "FFT", "GPU"))
data.append(calculate_and_store_times(ifft_tester, cp.array, "IFFT", "GPU"))
data.append(calculate_and_store_times(wavelet_tester, cp.array, "Wavelet Transform", "GPU"))
data.append(calculate_and_store_times(convolution_tester, cp.array, "Convolution", "GPU"))
data.append(calculate_and_store_times(denoising_tester, cp.array, "NL-means Denoising", "GPU"))
data.append(calculate_and_store_times(registration_tester, cp.array, "Image Registration", "GPU"))
data.append(calculate_and_store_times(inpainting_tester, cp.array, "Inpainting", "GPU"))
data.append(calculate_and_store_times(mean_std_tester, cp.array, "Mean and Std", "GPU"))

with open('times_cpu_gpu.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Operation', 'Time', 'Device', 'Number of Images', 'Image Shape'])
    writer.writerows(data)

print("Data has been saved to times.csv")