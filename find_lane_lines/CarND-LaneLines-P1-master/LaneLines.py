# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:26:20 2019

@author: wxq
"""

import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
import moviepy.editor

def show_images(images,cmap=None):
    cols = 1
    rows = len(images)
    plt.figure(figsize=(7,21))
    #plt.figure(figsize=(7,7))
    for i,image in enumerate(images):
        plt.subplot(rows,cols,i+1)
        cmap = 'gray' if len(image.shape)==2 else cmap
        plt.imshow(image,cmap=cmap)
    plt.tight_layout(pad=0,h_pad=0,w_pad=0)
    fig=plt.gcf()
    plt.show()
    return fig

test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]
#test_images = [plt.imread(path) for path in glob.glob('check_images/*.jpg')]
show_images(test_images)

def convert_gray_scale(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

gray_images = list(map(convert_gray_scale,test_images))
show_images(gray_images)

def apply_smoothing(image):
    kernel_size=15
    return cv2.GaussianBlur(image,(kernel_size,kernel_size),0)

blurred_images = list(map(apply_smoothing,gray_images))
show_images(blurred_images)

def detect_edges(image):
    low_threshold = 50
    high_threshold = 150
    return cv2.Canny(image,low_threshold,high_threshold)

edge_images = list(map(detect_edges,blurred_images))
show_images(edge_images)

def filter_region(image,vertices):
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask,vertices,255)
    else:
        cv2.fillPoly(mask,vertices,(255,)*mask.shape[2])
    return cv2.bitwise_and(image,mask)

def select_region(image):
    rows, cols = image.shape[:2]
    bottom_left = [cols*0.1, rows*0.95]
    top_left = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right = [cols*0.6, rows*0.6]
    vertices = np.array([[bottom_left,top_left,top_right,bottom_right]],dtype=np.int32)
    return filter_region(image,vertices)

roi_images = list(map(select_region,edge_images))
show_images(roi_images)

def hough_lines(image):
    return cv2.HoughLinesP(image,rho=1,theta=np.pi/180,threshold=20,minLineLength=20,maxLineGap=300)

list_of_lines = list(map(hough_lines,roi_images))

def draw_lines(image,lines,color=[255,0,0],thickness=2,make_copy=True):
    if make_copy:
        image = np.copy(image)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image,(x1,y1),(x2,y2),color,thickness)
    return image

line_images = []
for image, lines in zip(test_images,list_of_lines):
    line_images.append(draw_lines(image,lines))

show_images(line_images)

def average_slope_intercept(lines):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue
            slope = (y2-y1)/(x2-x1)
            intercept = y1-slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope<0:
                left_lines.append((slope,intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope,intercept))
                right_weights.append((length))

    left_lane = np.dot(left_weights,left_lines)/np.sum(left_weights) if len(left_weights)>0 else None
    right_lane = np.dot(right_weights,right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    return left_lane, right_lane

def make_line_points(y1,y2,line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1,y1),(x2,y2))

def lane_lines(image,lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1*0.6
    left_line = make_line_points(y1,y2,left_lane)
    right_line = make_line_points(y1,y2,right_lane)
    return left_line, right_line

def draw_lane_lines(image,lines,color=[255,0,0],thickness=20):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image,*line,color,thickness)
    return cv2.addWeighted(image,1.0,line_image,0.95,0.0)

lane_images = []
for image, lines in zip(test_images,list_of_lines):
    lane_images.append(draw_lane_lines(image,lane_lines(image,lines)))
fig=show_images(lane_images)
fig.savefig('lane_images.png')

def process_image(image):
    gray = convert_gray_scale(image)
    smooth_gray = apply_smoothing(gray)
    edges = detect_edges(smooth_gray)
    regions = select_region(edges)
    lines = hough_lines(regions)
    left_line, right_line = lane_lines(image,lines)
    return draw_lane_lines(image,(left_line,right_line))

def process_video(video_input,video_output):
    clip = moviepy.editor.VideoFileClip(os.path.join('test_videos',video_input))
    processed = clip.fl_image(process_image)
    processed.write_videofile(os.path.join('test_videos_output',video_output),audio=False)
    processed.close()

process_video('solidWhiteRight.mp4','white.mp4')
process_video('solidYellowLeft.mp4','yellow.mp4')
process_video('challenge.mp4','challenge.mp4')