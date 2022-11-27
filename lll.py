from PIL import Image
from torchvision import transforms

img = Image.open("/home/xyc/datasets/vessel_query_gallery/query/6295_0002_1051.jpg")
w, h = img.size
img.save('/home/xyc/datasets/vessel_query_gallery/0.jpg')
resize = transforms.Resize([384,128])
img = resize(img)
img.save('/home/xyc/datasets/vessel_query_gallery/1.jpg')
resize2 = transforms.Resize([h, w])
img = resize2(img)
img.save('/home/xyc/datasets/vessel_query_gallery/2.jpg')
