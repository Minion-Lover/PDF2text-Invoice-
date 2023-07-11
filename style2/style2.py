from tqdm import tqdm
import argparse
import easyocr
import cv2
import os
import numpy as np
import fitz
import math
import pandas as pd

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

def image_autosave():
    image = cv2.imread("out.png")
    height, width, _ = image.shape
    cutting_h = int(width/4*10)
    delta = int(width / 20)
    num = 0
    while num <= int(height / cutting_h):
        iimage = image
        temp_image = iimage[max(num * cutting_h-delta,0) : min((num+1)*cutting_h, height), 0 : width]
        num += 1
        cv2.imwrite(f"cutting_out{num}.png", temp_image)
    return num

def image_read(num):
    image = cv2.imread(f"cutting_out{num}.png")
    image = cv2.resize(image, None, fx = 4, fy = 4)
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(closing,kernel,iterations = 1)

    return dilation

def check(temp):
    temp[0]=str(temp[0])
    tmpfirst = temp[0][:2]
    tmplast = temp[0][2:]
    tmp = tmpfirst.replace('0', 'O').replace('7', 'T').replace('1', 'T')
    temp[0]=tmp+tmplast
    if len(temp[0]) > 6:
        tmp = temp[0][6:]
        temp[0]= temp[0][:5]
        temp.insert(1, tmp)
    if len(temp) >= 5:
        tmp = ''.join(temp[4:])
        temp[4]=tmp
        ttmp = temp[4][:-1]
        ttmp = ttmp.replace('O', '0').replace('T', '1')
        temp[4] = ttmp + 'T'
    tmp = []
    for i in range(min(5, len(temp))):
        tmp.append(temp[i])
    return tmp



def data(num):
    text_box = reader.readtext(
            image_read(num),
            decoder="greedy", 
            # detail=0, 
            # paragraph=True, 
            allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.-[] ",
            blocklist = ', ',
            width_ths = 25,
            # text_threshold= 0.9
            )
    result = []
    for each_box in text_box:
        bbox, text, _ = each_box
        temp = text.split()
        if "[1]" in temp or "[1" in temp or "1]" in temp:
            result.append(check(temp))
            print(check(temp))
    return result

def _main():
    result=[]
    for n in tqdm(range(len(doc))):
        image_save(n)
        rotate()
        for i in range(image_autosave()):
            result=result+data(i+1)

    my_df = pd.DataFrame(result)

    my_df.to_csv(f'{output_name}.csv', index=False, header=False)

if __name__ == "__main__":
    _main()
