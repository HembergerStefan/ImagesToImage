import numpy as np
import cv2
import os

# key: avg_color as tuple
# value: loaded image (np.array)
COLORS_DICT = {}


def load_all_images_from_dir(directory, img_width, img_height):
    files = os.listdir(directory)
    # loading all images
    for file in files:
        filepath = directory + '/' + file
        if os.path.isfile(filepath):
            try:
                # load image and store it into COLORS_DICT
                img = cv2.imread(filepath, cv2.COLOR_BGR2RGB)
                avg_color = get_avg_color(img)
                COLORS_DICT[avg_color] = resize_img(img, img_width, img_height)
            except np.AxisError:
                print(f'Could not load: {file}')
        else:
            print(f'{filepath} is not a file!')


def closest_img(color):
    available_colors = list(COLORS_DICT.keys())  # all keys of the dict in a list
    colors = np.array(available_colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors - color) ** 2, axis=1))
    index_of_smallest = np.where(distances == np.amin(distances))
    best_fitting_color = tuple(colors[index_of_smallest][0])
    return COLORS_DICT[best_fitting_color]


def get_avg_color(img):
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return tuple(map(lambda c: int(c), avg_color))


def resize_img(img, new_width, new_height):
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def create_big_image(input_img, img_pixels_width, img_pixels_height, new_img_width, new_img_height):
    # scale input image to output size
    input_img = resize_img(input_img, new_img_width, new_img_height)

    # create blank image
    new_img = np.zeros((img_pixels_height * new_img_height, img_pixels_width * new_img_width, 3), np.uint8)

    # loop through original img
    for y in range(0, new_img_height):
        for x in range(0, new_img_width):
            img = closest_img(input_img[y, x])  # get the best image for that pixel
            # insert the image at the right spot
            new_img[y * img_pixels_height: y * img_pixels_height + img_pixels_height,
            x * img_pixels_width: x * img_pixels_width + img_pixels_width] = img

    return new_img


def main():
    folder_with_images = 'resource_images'
    input_file_path = 'rainbow_butterfly.jpg'
    output_file = 'new.png'
    img_pixels_width = 50  # width of the small images
    img_pixels_height = 50  # height of the small images
    new_img_width = 200  # width of the output file in small images
    new_img_height = 200  # height of the output file in small images

    print('Loading Images ...')
    load_all_images_from_dir(folder_with_images, img_pixels_width, img_pixels_height)

    print('Creating big Image ...')
    input_img = cv2.imread(input_file_path, cv2.COLOR_BGR2RGB)
    new_img = create_big_image(input_img, img_pixels_width, img_pixels_height, new_img_width, new_img_height)

    print('Saving Image ...')
    cv2.imwrite(output_file, new_img)

    print('Finished')


if __name__ == '__main__':
    main()
