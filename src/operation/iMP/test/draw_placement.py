import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def draw(input_file, output_dir):
    in_file = open(input_file, 'r')
    line = in_file.readline().split(',')
    outline_width = float(line[0])
    outline_height = float(line[1])

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    lines = in_file.readlines()
    color = ('g')
    for line in lines:
        line = line.split(',')
        rect = patches.Rectangle((float(line[0]), float(line[1])), float(line[2]), float(line[3]), linewidth=0.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    # add bouding_box
    rect = patches.Rectangle((0, 0), outline_width, outline_height, linewidth=0.5, edgecolor='r', facecolor='none')
    ax.add_patch(rect)


    out_name = output_dir + "/" + (input_file.split('/')[-1].split('.')[0] + ".png")
    plt.xlim([0, outline_width * 1.5])
    plt.ylim([0, outline_height * 1.5])
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
    draw(input_dir + "placement.txt", output_dir)