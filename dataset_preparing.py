import os
import csv
from PIL import Image

def tiff_to_png(path, out):
    """Конвертирует .tiff файлы в .png и сохраняет их в директории out"""    
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff":
                if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + '.png'):
                    print(f"A png file already exists for {name}") 
                # If a png is *NOT* present, create one from the tiff.
                else:
                    # outfile = os.path.splitext(os.path.join(root, name))[0] + '.png'
                    outfile = out + os.path.splitext(name)[0] + '.png'
                    print(outfile)
                    try:
                        im = Image.open(os.path.join(root, name))
                        print(f"Generating png for {name}")
                        im.thumbnail(im.size)
                        im.save(outfile, "PNG", quality=100)
                    except Exception as e:
                        print(e)

def create_labels(path: str, csv_path, img_format='.png'):
    """Находит в path файлы с расширением img_format и записывают их в csv
    labels - метки изображений (каждый файл должен начинаться с 3-х букв метки класса)"""
    with open(csv_path, 'w', newline='') as csvfile:
        for root, _, files in os.walk(path, topdown=False):
            for name in files:
                if os.path.splitext(os.path.join(root, name))[1].lower() == img_format:
                    label = name[:3] # class name (example LYT or MON)
                    label_writer = csv.writer(csvfile, delimiter=' ', quotechar=' ')
                    label_writer.writerow([name, label])
                else:
                    continue
    csvfile.close()