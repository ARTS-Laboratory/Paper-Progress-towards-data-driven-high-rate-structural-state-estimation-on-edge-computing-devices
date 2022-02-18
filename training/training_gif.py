import imageio
import os

frame = 0
with imageio.get_writer('training.gif', mode='I', duration=0.5) as writer:
    filename = "./plots/plt" + str(frame) + ".png"
    while(os.path.exists(filename)):
        image = imageio.imread(filename)
        writer.append_data(image)
        filename = "./plots/plt" + str(frame) + ".png"
        frame += 1
    writer.close()