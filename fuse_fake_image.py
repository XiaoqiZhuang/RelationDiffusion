from PIL import Image, ImageOps
import numpy as np
import os
import random


def load_images_with_prefix(directory, prefix="cat"):
    match_images = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            img = Image.open(os.path.join(directory, filename))
            match_images.append(img)
    return match_images


def fuse_fake_image(image1, image2, background, output_path, output_size=(512, 512)):
    # 按比例缩放两张图片，高度不超过下半部分的高度
    target_height = output_size[1] // 2  # 下半部分高度
    image1 = ImageOps.contain(image1, (output_size[0] // 2, target_height))  # 左图最大宽度为背景宽度的一半
    image2 = ImageOps.contain(image2, (output_size[0] // 2, target_height))  # 右图最大宽度为背景宽度的一半

    # 计算粘贴位置
    left_x = (output_size[0] // 2 - image1.size[0]) // 2  # 左图居中对齐左侧区域
    left_y = output_size[1] // 2  # 下半部分的起始高度

    right_x = output_size[0] // 2 + (output_size[0] // 2 - image2.size[0]) // 2  # 右图居中对齐右侧区域
    right_y = output_size[1] // 2  # 下半部分的起始高度

    # 将两张图粘贴到背景上
    image = background.copy()
    image.paste(image2, (left_x, left_y), image2)  # 粘贴左图
    image.paste(image1, (right_x, right_y), image1)  # 粘贴右图

    image.save(output_path)


if __name__ == '__main__':
    BG = 'data/bg'
    MATERIAL = 'data/material'
    S1 = 'cat'
    S2 = 'dog'

    bgs = []
    bg_paths = os.listdir(BG)
    for bg_path in bg_paths:
        bg = Image.open(f'{BG}/{bg_path}')
        bgs.append(bg)
    
    s1s = load_images_with_prefix(MATERIAL, S1)
    s2s = load_images_with_prefix(MATERIAL, S2)

    i = 0
    for bg in bgs:
        for img1 in s1s:
            for img2 in s2s:
                fuse_fake_image(img1, img2, bg, f'data/fake/{i}.png')
                i+=1


