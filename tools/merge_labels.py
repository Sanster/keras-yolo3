import argparse
import os

if __name__ == "__main__":
    """
    将使用 labelImg 生成的 txt 文件转换成 train.txt
    """
    label_dir = '/home/cwq/data/yolo/cube2'
    label_names = list(filter(lambda x: x.endswith('.txt'), os.listdir(label_dir)))
    label_paths = list(map(lambda x: os.path.join(label_dir, x), label_names))
    img_paths = list(map(lambda x: os.path.join(label_dir, x.split('.')[0] + '.png'), label_names))

    print("Labels num: {}".format(len(label_paths)))
    """
    原始格式： class_id, xcen, ycen, w, h
    Row format: `image_file_path box1 box2 ... boxN`;
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space). 绝对坐标
    """
    img_w = 752
    img_h = 416

    train_file = open('./train.txt', mode='w')
    for i, p in enumerate(label_paths):
        with open(p) as f:
            line = f.readline()
        line = line.split(' ')
        line = list(map(lambda x: float(x), line))

        w = line[3]
        h = line[4]

        class_id = line[0]
        x_min = int((line[1] - w / 2.) * img_w)
        y_min = int((line[2] - h / 2.) * img_h)
        x_max = int((line[1] + w / 2.) * img_w)
        y_max = int((line[2] + h / 2.) * img_h)
        train_file.write("%s %d,%d,%d,%d,%d\n" % (img_paths[i], x_min, y_min, x_max, y_max, class_id))

    train_file.close()
