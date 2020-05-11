import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


print("opencv版本：", cv.__version__)
print("numpy版本：", np.__version__)


# 读取图片、显示图片、保存图片
def read_image():
    # 读取图片
    src = cv.imread("E:/Desktop/opencv_learning/data/lena.jpg")
    # 显示图片
    cv.imshow("input image", src)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # 保存图片
    cv.imwrite("E:/Desktop/opencv_learning/data/lena_copy.jpg", src)


# waitKey函数的使用
def waitkey():
    src = cv.imread("E:/Desktop/opencv_learning/data/lena.jpg", 0)
    cv.imshow("input image", src)
    k = cv.waitKey(0)
    if k == 27:  # 按下ESC释放所有图像窗口
        cv.destroyAllWindows()
    if k == ord("q"):  # 按下q键保存图片
        cv.imwrite("E:/Desktop/opencv_learning/data/lena_copy.jpg", src)
        cv.destroyAllWindows()


# 使用matplotlib.pyplot库显示图片
def mat_image():
    src = cv.imread("E:/Desktop/opencv_learning/data/lena.jpg", -1)
    plt.imshow(src)
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    # read_image()
    # waitkey()
    mat_image()
