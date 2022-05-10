#!/usr/bin/env python

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os

#code adapted from https://blog.paperspace.com/train-yolov5-custom-data/

def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    
    try: 
        transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
        transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
        transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
        transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
        transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
        transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    except:
        transformed_annotations[[1,3]] = annotations[[1,3]] * w
        transformed_annotations[[2,4]] = annotations[[2,4]] * h 
    
        transformed_annotations[1] = transformed_annotations[1] - (transformed_annotations[3] / 2)
        transformed_annotations[2] = transformed_annotations[2] - (transformed_annotations[4] / 2)
        transformed_annotations[3] = transformed_annotations[1] + transformed_annotations[3]
        transformed_annotations[4] = transformed_annotations[2] + transformed_annotations[4]  
        
        print(transformed_annotations)
        
        
    for ann in transformed_annotations:
        try:
            obj_cls, x0, y0, x1, y1 = ann
            plotted_image.rectangle(((x0,y0), (x1,y1)), width = 10, outline="#0000ff")
        
        except: 
            obj_cls= transformed_annotations[0]
            x0=transformed_annotations[1]
            y0=transformed_annotations[2]
            x1=transformed_annotations[3]
            y1=transformed_annotations[4]
            plotted_image.rectangle(((x0,y0), (x1,y1)), width = 10, outline="#0000ff")
        
    
    plt.imshow(np.array(image))
    plt.show()

#get an annotation file
annotation_file = './houses.txt'


#Get the corresponding image file
image_file = annotation_file.replace("txt", "png")
assert os.path.exists(image_file)

#Load the image
image = Image.open(image_file)

#Plot the Bounding Box
plot_bounding_box(image, np.loadtxt(annotation_file))
