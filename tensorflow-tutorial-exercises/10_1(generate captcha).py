from captcha.image import ImageCaptcha
import numpy as np
import random
import sys
from PIL import Image

# captcha numbers
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# captcha alphabets
alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
             'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# captcha ALPHABETS
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def random_content_text(char_set=numbers, char_size=4):
    content_text = []
    for i in range(char_size):
        r = random.choice(char_set)
        content_text.append(r)
    return content_text


def gen_captcha_text_and_image():
    content_text = random_content_text()
    content_text = ''.join(content_text)
    image = ImageCaptcha()
    # captcha = image.generate(content_text)
    image.write(content_text, 'captcha/images/' + content_text + '.jpg')


# will less than 10000
num = 10000
if __name__ == "__main__":
    for i in range(num):
        gen_captcha_text_and_image()
        sys.stdout.write("\r>> Creating captcha %d/%d" % (i + 1, num))
        sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    print("finished!!")
