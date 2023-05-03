#-------------------------------------------------------------------------------------------------------#
#   kmeans?然???据集中的框?行聚?，但是很多?据集由于框的大小相近，聚?出?的9?框相差不大，
#   ??的框反而不利于模型的??。因?不同的特征?适合不同大小的先?框，shape越小的特征?适合越大的先?框
#   原始网?的先?框已?按大中小比例分配好了，不?行聚?也?有非常好的效果。
#-------------------------------------------------------------------------------------------------------#
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    #-------------------------------------------------------------#
    #   input_shape ?入的shape大小，一定要是32的倍?
    #-------------------------------------------------------------#
    input_shape = [640, 640]
    #-------------------------------------------------------------#
    #   anchors_num 先?框的?量
    #-------------------------------------------------------------#
    anchors_num = 9
    #-------------------------------------------------------------#
    #   train_annotation_path ???片路?和??
    #-------------------------------------------------------------#
    train_annotation_path = '2007_train.txt'

    np.random.seed(0)

    def cas_ratio(box,cluster):
        ratios_of_box_cluster = box / cluster
        ratios_of_cluster_box = cluster / box
        ratios = np.concatenate([ratios_of_box_cluster, ratios_of_cluster_box], axis = -1)
        return np.max(ratios, -1)

    def avg_ratio(box,cluster):
        return np.mean([np.min(cas_ratio(box[i],cluster)) for i in range(box.shape[0])])

    def kmeans(box,k):
        #-------------------------------------------------------------#
        #   取出一共有多少框
        #-------------------------------------------------------------#
        row = box.shape[0]
        
        #-------------------------------------------------------------#
        #   每?框各??的位置
        #-------------------------------------------------------------#
        distance = np.empty((row,k))
        
        #-------------------------------------------------------------#
        #   最后的聚?位置
        #-------------------------------------------------------------#
        last_clu = np.zeros((row,))

        np.random.seed()

        #-------------------------------------------------------------#
        #   ?机?5??聚?中心
        #-------------------------------------------------------------#
        cluster = box[np.random.choice(row,k,replace = False)]

        iter = 0
        while True:
            #-------------------------------------------------------------#
            #   ?算?前框和先?框的?高比例
            #-------------------------------------------------------------#
            for i in range(row):
                distance[i] = cas_ratio(box[i],cluster)
            
            #-------------------------------------------------------------#
            #   取出最小?
            #-------------------------------------------------------------#
            near = np.argmin(distance,axis=1)

            if (last_clu == near).all():
                break
            
            #-------------------------------------------------------------#
            #   求每一??的中位?
            #-------------------------------------------------------------#
            for j in range(k):
                cluster[j] = np.median(
                    box[near == j],axis=0)

            last_clu = near
            if iter % 5 == 0:
                print('iter: {:d}. avg_ratio:{:.2f}'.format(iter, avg_ratio(box,cluster)))
            iter += 1

        return cluster, near

    def load_data(train_annotation_path):
        #---------------------------#
        #   ?取?据集??的txt
        #---------------------------#
        with open(train_annotation_path, encoding='utf-8') as f:
            train_lines = f.readlines()

        data = []
        #-------------------------------------------------------------#
        #   ?于每一?xml都?找box
        #-------------------------------------------------------------#
        for line in tqdm(train_lines):
            line    = line.split()
            #------------------------------#
            #   ?取?像并??成RGB?像
            #------------------------------#
            image   = Image.open(line[0])
            #------------------------------#
            #   ?得?像的高?与目?高?
            #------------------------------#
            iw, ih  = image.size
            #------------------------------#
            #   ?得??框
            #------------------------------#
            boxes   = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
            for box in boxes:
                xmin = int(float(box[0])) / iw
                ymin = int(float(box[1])) / ih
                xmax = int(float(box[2])) / iw
                ymax = int(float(box[3])) / ih

                xmin = np.float64(xmin)
                ymin = np.float64(ymin)
                xmax = np.float64(xmax)
                ymax = np.float64(ymax)
                # 得到?高
                data.append([xmax - xmin, ymax - ymin])

        return np.array(data)
        
    #-------------------------------------------------------------#
    #   ?入所有的xml
    #   存?格式??化?比例后的width,height
    #-------------------------------------------------------------#
    print('Load xmls.')
    data = load_data(train_annotation_path)
    print('Load xmls done.')
    
    #-------------------------------------------------------------#
    #   使用k聚?算法
    #-------------------------------------------------------------#
    print('K-means boxes.')
    cluster, near   = kmeans(data, anchors_num)
    print('K-means boxes done.')
    data            = data * np.array([input_shape[1], input_shape[0]])
    cluster         = cluster * np.array([input_shape[1], input_shape[0]])

    #-------------------------------------------------------------#
    #   ??
    #-------------------------------------------------------------#
    for j in range(anchors_num):
        plt.scatter(data[near == j][:,0], data[near == j][:,1])
        plt.scatter(cluster[j][0], cluster[j][1], marker='x', c='black')
    plt.savefig("kmeans_for_anchors.jpg")
    plt.show()
    print('Save kmeans_for_anchors.jpg in root dir.')

    cluster = cluster[np.argsort(cluster[:, 0] * cluster[:, 1])]
    print('avg_ratio:{:.2f}'.format(avg_ratio(data, cluster)))
    print(cluster)

    f = open("yolo_anchors.txt", 'w')
    row = np.shape(cluster)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (cluster[i][0], cluster[i][1])
        else:
            x_y = ", %d,%d" % (cluster[i][0], cluster[i][1])
        f.write(x_y)
    f.close()