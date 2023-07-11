from tqdm import tqdm
import argparse
import easyocr
import cv2
import os
import numpy as np
import fitz
import math
import pandas as pd
import re

# from PyPDF2 import PdfReader, PdfWriter
# from pdf2image import convert_from_path

parser = argparse.ArgumentParser(
    description="Process PDF files of NJPD Crash Reports to return wanted values."
)
parser.add_argument("pdf_file", metavar="file", help="path to file")

args = parser.parse_args()
file = args.pdf_file
doc = fitz.open(file)
output_name = os.path.splitext(os.path.basename(file))[0].replace("/", "_").replace("\\", "_")

reader = easyocr.Reader(["en"], gpu=True)

def rotate():
        
    image = cv2.imread('out.png')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize=3)
    
    lines_list =[]
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=1000, # Min number of votes for valid line
                minLineLength=1500, # Min allowed length of line
                maxLineGap=500 # Max allowed gap between line for joining them
                )
    Ang = []
    for points in lines:
        x1,y1,x2,y2=points[0]
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
        lines_list.append([(x1,y1),(x2,y2)])
        if x1!=x2:
            angle = math.atan((y2-y1)/(x2-x1))
            Ang.append(angle)
    if len(Ang) != 0:
        angle = Ang[0]
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle/2, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite('out.png',rotated)

def image_save(num):
    page = doc.load_page(num)
    dpi=300
    pixmap = page.get_pixmap(dpi=300)
    pixmap.set_dpi(dpi, dpi)
    pixmap.save("out.png")

def image_read():
    image = cv2.imread("out.png")
    image = cv2.resize(image, None, fx = 4, fy = 4)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(closing,kernel,iterations = 1)

    return dilation

def check(temp):
    id_end = temp[0].find("-")
    if id_end > 0:
        temp[0] = temp[0][: id_end]
    id_end = temp[0].find("{")
    if id_end > 0:
        temp[0] = temp[0][: id_end]
    temp[0]=re.sub(' +', ' ', temp[0])
    temp[0] = str(temp[0]).replace('FOURLE', 'DOUBLE').replace('FOUBLE', 'DOUBLE').replace('DQUBLE', 'DOUBLE').replace('DUBLE', 'DOUBLE').replace('MURLE', 'DOUBLE').replace('MUBLE', 'DOUBLE').replace('DURLE', 'DOUBLE').replace('DAHN', 'DAMN').replace('KORN', 'BORN')

    temp[1] = str(temp[1]).replace(' ', '').replace('{', '1').replace('|', '1').replace('VAG', 'YA6').replace('YAG', 'YA6').replace('VA6', 'YA6').replace('OT', '01').replace('OI', '01').replace('342', '3A2')

    if len(temp) > 5:
        for i in range(len(temp) - 2):
            if temp[i+2].find('.') == -1:
                temp[i+2] = temp[i+2]+'.'+temp[i+3]
                for j in range(len(temp) - 3 - i):
                    temp[i+3+j] = temp[i + 2 +j]
        temp.pop()
    for i in range(len(temp) - 2):
        temp[i+2] = temp[i+2].replace(' ', '')
        if temp[i+2].find('.') == -1:
            temp[i+2]=temp[i+2][:-2]+'.00'
        if str(temp[i+2][0]) == '.' or str(temp[i+2]) == '0':
            temp[i+2] = '1.0'
        temp[i+2] = str(temp[i+2]).replace('o', '0').replace('U', '0').replace('O', '0').replace('Q', '0').replace('S','5').replace('C','0').replace('J','0').replace('E','8').replace('G','0')
    return temp



def data():
    text_box = reader.readtext(
            image_read(),
            decoder="greedy", 
            # detail=0, 
            # paragraph=True, 
            allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.-{ ",
            blocklist = ', ',
            width_ths = 2,
            text_threshold= 0.6
            )
    width_max = 0
    for each_box in text_box:
        # print(each_box)
        bbox, text, _ = each_box
        if width_max < bbox[1][0]:
            width_max = bbox[1][0]
    result = []
    for each_box in text_box:
        bbox, text, _ = each_box
        temp = []
        if bbox[1][0] > width_max - 200 and bbox[1][0] < width_max + 200:
            for eeach_box in text_box:
                bbbox, ttext, _ = eeach_box
                if bbbox[3][1] < bbox[3][1]-100  and bbbox[3][1] > bbox[3][1] - 300:
                    temp.append(ttext)
                if bbbox[3][1] > bbox[3][1] - 100 and bbbox[3][1] < bbox[3][1] + 100:
                    temp.append(ttext)
        if len(temp) >= 4:
            check(temp)
            if temp[1].find('.')>0:
                continue
            if len(temp[1]) < 5 or len(temp[1]) > 7:
                continue
            result.append(temp)
            print("\n", temp)
    return result

def _main():
    result=[]
    for n in tqdm(range(len(doc))):
        image_save(n)
        rotate()
        result=result+data()

    my_df = pd.DataFrame(result)

    my_df.to_csv(f'{output_name}.csv', index=False, header=False)

if __name__ == "__main__":
    _main()
