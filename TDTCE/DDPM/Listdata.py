import os

datasets_path   = "datasets/"

if __name__ == "__main__":
    photos_names    = os.listdir(datasets_path)
    photos_names    = sorted(photos_names)

    list_file       = open('train_traffic.txt', 'w')
    for photo_name in photos_names:
        if(photo_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
            list_file.write(os.path.join(os.path.abspath(datasets_path), photo_name))
            list_file.write('\n')
    list_file.close()
