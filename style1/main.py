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
                threshold=400, # Min number of votes for valid line
                minLineLength=1500, # Min allowed length of line
                maxLineGap=800 # Max allowed gap between line for joining them
                )

    Ang = []
    for points in lines:
        x1,y1,x2,y2=points[0]
        # cv2.line(image,(x1,y1),(x2,y2),(255,255,0),2)
        lines_list.append([(x1,y1),(x2,y2)])
        if x1!=x2:
            angle = math.atan((y2-y1)/(x2-x1))
            # print(angle*180/math.pi)
            Ang.append(angle)
    if len(Ang) > 3:
        sum1 = 0 
        num1 = 0
        sum2 = 0
        num2 = 0
        for i in Ang:
            if i > 1.3:
                num1 += 1
                sum1 += i
            if i < -1.3:
                num2 += 1
                sum2 += i
        angle = math.pi/2-(sum1-sum2)/(num1+num2)
        if num1-num2 < 0: angle = angle * -1
        # print(angle*180/math.pi)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle*180/math.pi/2, 1.0)
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
    for i in range(len(temp)):
        if i == 0:
            temp[0] = str(temp[0]).replace(' ', '').replace('{', '1').replace('|', '1').replace('VAG', 'YA6').replace('YAG', 'YA6').replace('VA6', 'YA6').replace('OT', '01').replace('OI', '01').replace('342', '3A2').replace('615', 'G15').replace('EAB', 'EA6').replace("1S0I","1301")
        else:
            temp[i] = str(temp[i]).replace('o', '0').replace('U', '0').replace('O', '0').replace('Q', '0').replace('S','5').replace('C','0').replace('J','0').replace('E','8').replace('G','0')
    # print(temp)
    return temp
def checkid(temp):
    tmp = ''
    num = 1
    for i in temp:
        if num == 1:
            tmp = str(i)
            num = 0
        else:
            tmp = tmp + ' ' + str(i)
    id_end = tmp.find("-")
    if id_end > 0:
        tmp = tmp[: id_end]
    id_end = tmp.find("{")
    if id_end > 0:
        tmp = tmp[: id_end]
    tmp = tmp.replace('FOURLE', 'DOUBLE').replace('FOUBLE', 'DOUBLE').replace('DQUBLE', 'DOUBLE').replace('DUBLE', 'DOUBLE').replace('MURLE', 'DOUBLE').replace('MUBLE', 'DOUBLE').replace('DURLE', 'DOUBLE').replace('DAHN', 'DAMN').replace('KORN', 'BORN').replace('HEAR', 'WEAR').replace('TKIS', 'THIS').replace('SKI ', 'SKIN ').replace('CRLSHED ', 'CRUSHED ').replace('OW ', 'DW ')
    return tmp



def data(num):
    text_box = reader.readtext(
            image_read(num),
            decoder="greedy", 
            # detail=0, 
            # paragraph=True, 
            allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.-[] ",
            blocklist = ', ',
            width_ths = 30,
            # text_threshold= 0.9
            )
    result = []
    tempid = []
    for each_box in text_box:
        bbox, text, _ = each_box
        temp = text.split()
        # print(temp)
        tmp = []
        if len(temp) < 3:
            continue
        if  len(temp[0])==6 and temp[1].find('.')>0 and len(temp[1])==3:
            tmp.append(checkid(tempid))
            for x in check(temp):
                tmp.append(x)
            if len(tmp) != 0: result.append(tmp)
        tempid = temp
    print(result)
    return result

def _main():
    result=[]
    for n in tqdm(range(len(doc))):
        image_save(n)
        rotate()
        for i in range(image_autosave()):
            result = result + data(i+1)
    my_df = pd.DataFrame(result)
    my_df.to_csv(f'{output_name}.csv', index=False, header=False)

if __name__ == "__main__":
    _main()
