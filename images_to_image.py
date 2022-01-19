import cv2
import numpy
import numpy as np
import os

COLORS = []
COLORS_DICT = {}


def init(img_dir, img_width, img_height):
    files = os.listdir(img_dir)
    for file in files:
        filepath = img_dir + '/' + file
        avg = get_avg_color(filepath)
        COLORS.append(avg)
        COLORS_DICT[tuple(avg)] = resize_img(cv2.imread(filepath, cv2.COLOR_BGR2RGB), img_width, img_height)
        # COLORS_DICT[tuple(avg)] = file


def closest_img(color):
    r, g, b = color
    color = b, g, r
    colors = np.array(COLORS)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors - color) ** 2, axis=1))
    index_of_smallest = np.where(distances == np.amin(distances))
    smallest_distance = colors[index_of_smallest]
    return COLORS_DICT[list(map(tuple, smallest_distance))[0]]


def get_avg_color(file):
    img = cv2.imread(file)
    avg_color_per_row = numpy.average(img, axis=0)
    avg_color = numpy.average(avg_color_per_row, axis=0)
    return list(map(lambda c: int(c), avg_color))


def resize_img(img, new_width, new_height):
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def create(file, all_img_width, all_img_height, new_img_width_count, new_img_height_count):
    input_img = cv2.imread(file, cv2.COLOR_BGR2RGB)
    input_img = resize_img(input_img, 200, 200)

    # in_height = input_img.shape[0]
    # in_width = input_img.shape[1]

    input_img = resize_img(input_img, new_img_width_count, new_img_height_count)
    new_img = np.zeros((all_img_height * new_img_height_count, all_img_width * new_img_width_count, 3), np.uint8)

    # loop through ori img
    for y in range(0, new_img_height_count):
        for x in range(0, new_img_width_count):
            img = closest_img(input_img[y, x])
            new_img[y * all_img_height: y * all_img_height + all_img_height,
            x * all_img_width:x * all_img_width + all_img_width] = img

    return new_img


def main():
    input_file = 'butterfly_gae.jpg'
    all_img_width = 200
    all_img_height = 200
    new_img_width_count = 50
    new_img_height_count = 50

    print('Loading Images ...')
    init("res", all_img_width, all_img_height)
    print('Creating big Image ...')
    new_img = create(input_file, all_img_width, all_img_height, new_img_width_count, new_img_height_count)
    cv2.imwrite("new.png", new_img)
    print('Finished')


if __name__ == '__main__':
    main()
