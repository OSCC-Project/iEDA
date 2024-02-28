import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def draw(input_file, output_dir):
    in_file = open(input_file, 'r')
    line = in_file.readline().split(',')
    outline_lx = float(line[0])
    outline_ly = float(line[1])
    outline_width = float(line[2])
    outline_height = float(line[3])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # add bouding_box
    rect = patches.Rectangle((outline_lx, outline_ly), outline_width, outline_height, linewidth=0.3, edgecolor='k', facecolor='none')
    ax.add_patch(rect)

    lines = in_file.readlines()
    for line in lines:
        line = line.strip().split(',')
        type = line[5]
        if type == 'macro':
            color = 'g'
            facecolor = color
            # facecolor = 'none'
        elif type == 'io':
            color =  'r'
            facecolor = color
        else:
            color = 'c'
            facecolor = 'none'
        
        lx = float(line[0])
        ly = float(line[1])
        w = float(line[2])
        h = float(line[3]) 
        if (type == 'io'):
            lx -= 1e4
            ly -= 1e4
            w = 2e4
            h = 2e4
        rect = patches.Rectangle((lx, ly), w, h, linewidth=0.3, edgecolor=color, facecolor=facecolor)
        ax.add_patch(rect)




    out_name = output_dir + "/" + os.path.basename(input_file.replace('.txt', '.png'))
    plt.xlim([outline_lx-100000, outline_width * 1.5])
    plt.ylim([outline_ly-100000, outline_height * 1.5])
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print('{} saved...'.format(out_name))
    plt.close()


if __name__ == "__main__":
    input_dir = "/home/liuyuezuo/iEDA-master/build/output/"
    output_dir = input_dir
    # for file_name in os.listdir(input_dir):
    #     if not file_name.endswith('.txt'):
    #         continue
    #     file_name = input_dir + file_name
    #     draw(file_name, output_dir)
    draw(input_dir + "placement_level1_200_0.txt", output_dir)
    draw(input_dir + "placement_level1_200_0_aligned.txt", output_dir)