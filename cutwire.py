from PIL import Image

image_path = "./aaaa.png"
image = Image.open(image_path)
bounding_box = [0.411263, 0.465223, 0.192926, 0.216653]
image_width, image_height = image.size
x_center = bounding_box[0] * image_width
y_center = bounding_box[1] * image_height
width = bounding_box[2] * image_width
height = bounding_box[3] * image_height
left = int(x_center - width / 2)
right = int(x_center + width / 2)
top = int(y_center - height / 2)
bottom = int(y_center + height / 2)
cropped_image = image.crop((left, top, right, bottom))

cropped_image_path = "./cropped_image.png"
cropped_image.save(cropped_image_path)

cropped_image_path
