from PIL import Image  
  
def concat_images(img1_path, img2_path, img3_path, output_path, height=640):
    # 打开图片  
    img1 = Image.open(img1_path)
    original_width1, original_height1 = img1.size
    print_width1 = int((original_width1 * height) / original_height1)
    img1 = img1.resize((print_width1, height))

    img2 = Image.open(img2_path)
    original_width2, original_height2 = img2.size
    print_width2 = int((original_width2 * height) / original_height2)
    img2 = img2.resize((print_width2, height))

    img3 = Image.open(img3_path)
    original_width3, original_height3 = img3.size
    print_width3 = int((original_width3 * height) / original_height3)
    img3 = img3.resize((print_width3, height))

    img3 = img3.transpose(Image.FLIP_LEFT_RIGHT)

    # 确保两张图片的高度相同，否则拼接会出问题  
    if img1.height != img2.height or img2.height != img3.height or img1.height !=img3.height:
        print("两张图片的高度必须相同才能进行水平拼接")  
        return  
  
    # 计算拼接后的图片宽度  
    width = img1.width + img2.width + img3.width
  
    # 创建一个新的空白图片，大小是两张图片的宽度之和，高度与两张图片相同  
    new_img = Image.new('RGB', (width, img1.height))  
  
    # 将两张图片粘贴到新图片上  
    new_img.paste(img1, (0, 0))  # 第一个图片放在左边  
    new_img.paste(img2, (img1.width, 0))  # 第二个图片放在右边  
    new_img.paste(img3, (img1.width+img2.width, 0))

    # 保存新图片  
    new_img.save(output_path)  
  
# 使用示例  
concat_images( 'Rousseau.jpg', 'kant.jpg', 'marx.jpg', 'Kalki.jpg')