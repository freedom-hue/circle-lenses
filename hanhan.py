from PIL import Image

# 打开憨憨的图片并转成 ico
img = Image.open("hanhan.jpg")
img.save("hanhan.ico", format="ICO", sizes=[(256, 256)])
print("ico 图标生成成功！")