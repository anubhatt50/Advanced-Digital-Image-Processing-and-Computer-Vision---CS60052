import cv2
import numpy as np
import copy
import argparse

def findLine(u,v):
    L = np.cross(u,v)
    return L

def intersectionPt(L1,L2):
    intersect = np.cross(L1,L2)
    return intersect

def vanishLine(u,v):
    return np.cross(u,v)

def centerLine(u,size):
    origin = np.array([int(size[0]/2),int(size[1]/2)])
    c = -u[0]*origin[0] - u[1]*origin[1]
    return np.array([u[0],u[1],c])

def lineForm(vanLine,vertPt,size):
    centLine = centerLine(vanLine,size)
    vertLine = np.array([0,1,-vertPt[1]])
    vanPt = np.cross(vertLine,vanLine)
    randPt1 = np.array([np.random.randint(low=1,high=size[0]),np.random.randint(low=1,high=size[1])])
    randPt2 = np.array([np.random.randint(low=1,high=size[0]),np.random.randint(low=1,high=size[1])])
    randPt3 = np.array([np.random.randint(low=1,high=size[0]),np.random.randint(low=1,high=size[1])])
    line1 = np.cross(randPt1,vanPt)
    line2 = np.cross(randPt2,vanPt)
    line3 = np.cross(randPt3,vanPt)
    return line1,line2,line3

def findHomography(line1,line2,line3,vanLine):
    a = np.random.randint()
    b = np.random.randint()
    trnLine1 = np.array([a,b,np.random.randint()])

posList = []
def onMouse(event, x, y, flags, param):
   global posList
   if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))

def draw_line(img_orig):
    img=copy.deepcopy(img_orig)
    global posList
    posList = []
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', onMouse)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    poslist = np.array(posList)
    poslist = np.concatenate((poslist,np.ones((2,1))),axis=1)
    line1 = findLine(poslist[0],poslist[1])
    theta = np.arctan(-line1[0]/line1[1])
    endpt_x = int(poslist[0][0] - 1000*np.cos(theta))
    endpt_y = int(poslist[0][1] - 1000*np.sin(theta))
    startpt_x = int(poslist[0][0] + 1000*np.cos(theta))
    startpt_y = int(poslist[0][1] + 1000*np.sin(theta))
    cv2.line(img, (startpt_x,startpt_y), (endpt_x,endpt_y), (0, 255, 0), thickness=2, lineType=8)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    print("Homogeneous representation of line in P2: ", line1/line1[2])

def find_vanishLine(img):
    img2=copy.deepcopy(img)
    img1=copy.deepcopy(img)
    van_pts=[]
    global posList
    for n in range(2):
        posList = []
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', onMouse)
        cv2.imshow('image',img1)
        cv2.waitKey(0)
        posList = np.array(posList)
        posList = np.concatenate((posList,np.ones((4,1))),axis=1)
        line1 = findLine(posList[0],posList[1])
        theta = np.arctan(-line1[0]/line1[1])
        endpt_x = int(posList[0][0] - 1000*np.cos(theta))
        endpt_y = int(posList[0][1] - 1000*np.sin(theta))
        startpt_x = int(posList[0][0] + 1000*np.cos(theta))
        startpt_y = int(posList[0][1] + 1000*np.sin(theta))
        cv2.line(img1, (startpt_x,startpt_y), (endpt_x,endpt_y), (0, 255, 0), thickness=2, lineType=8)
        line2 = findLine(posList[2],posList[3])
        theta = np.arctan(-line2[0]/line2[1])
        endpt_x = int(posList[2][0] - 1000*np.cos(theta))
        endpt_y = int(posList[2][1] - 1000*np.sin(theta))
        startpt_x = int(posList[2][0] + 1000*np.cos(theta))
        startpt_y = int(posList[2][1] + 1000*np.sin(theta))
        cv2.line(img1, (startpt_x,startpt_y), (endpt_x,endpt_y), (0, 255, 0), thickness=2, lineType=8)
        van_pts.append(intersectionPt(line1,line2))
    cv2.imshow('image',img1)
    cv2.waitKey(0)
    van_line=findLine(van_pts[0],van_pts[1])
    print("Homogeneous representation of VANISHING LINE in P2: ",van_line/van_line[2])
    centre=np.array([img.shape[1]/2,img.shape[0]/2])
    theta = np.arctan(-van_line[0]/van_line[1])
    endpt_x = int(centre[0] - 1000*np.cos(theta))
    endpt_y = int(centre[1] - 1000*np.sin(theta))
    startpt_x = int(centre[0] + 1000*np.cos(theta))
    startpt_y = int(centre[1] + 1000*np.sin(theta))
    cv2.line(img2, (startpt_x,startpt_y), (endpt_x,endpt_y), (0, 255, 0), thickness=2, lineType=8)
    cv2.imshow('image',img2)
    cv2.waitKey(0)
    return img2, van_line

def homography(img,H):
    img_new=np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ind = np.matmul(H,np.array([j,i,1]))
            ind = ind/ind[2]
            ind = ind.astype(np.int64)
            if ind[1]<img.shape[0] and ind[0]<img.shape[1] and ind[0]>=0 and ind[1]>=0:
                img_new[ind[1],ind[0],:]=img[i,j,:]
    return img_new

def transParallel(img_orig):
    img = copy.deepcopy(img_orig)
    img2, van_line = find_vanishLine(img)
    global posList
    posList = []
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', onMouse)
    cv2.imshow('image',img2)
    cv2.waitKey(0)
    cv2.line(img2, (posList[0][0],-10000), (posList[0][0],10000), (0, 255, 0), thickness=2, lineType=8)
    cv2.imshow('image',img2)
    cv2.waitKey(0)
    vert_line = np.array([1,0,-posList[0][0]])
    vanPoint = intersectionPt(vert_line,van_line)
    l = van_line/van_line[2]
    H = np.eye(3)
    H[2] = l
    idealPoint = np.matmul(H,vanPoint)
    idealPoint = idealPoint/np.abs(idealPoint).max()
    par1 = np.array([idealPoint[1],-idealPoint[0],-20])
    par2 = np.array([idealPoint[1],-idealPoint[0],-1])
    par3 = np.array([idealPoint[1],-idealPoint[0],-50])
    H_t=np.transpose(np.linalg.inv(H))
    l1_new=np.matmul(np.linalg.inv(H_t),par1)
    l2_new=np.matmul(np.linalg.inv(H_t),par2)
    l3_new=np.matmul(np.linalg.inv(H_t),par3)
    x1,x2,x3 = 50, 100, 150
    y1 = [-(l1_new[0]*x1*100+l1_new[2])/l1_new[1],-(-l1_new[0]*x1+l1_new[2])/l1_new[1]]
    y2= [-(l2_new[0]*x2*100+l2_new[2])/l2_new[1],-(-l2_new[0]*x2+l2_new[2])/l2_new[1]]
    y3 = [-(l3_new[0]*x3*100+l3_new[2])/l3_new[1],-(-l3_new[0]*x3+l3_new[2])/l3_new[1]]
    cv2.line(img2, (x1*100,int(y1[0])), (-x1,int(y1[1])), (255, 0,255), thickness=2, lineType=8)
    cv2.line(img2, (x2*100,int(y2[0])), (-x2,int(y2[1])), (255, 255,0), thickness=2, lineType=8)
    cv2.line(img2, (x3*100,int(y3[0])), (-x3,int(y3[1])), (255, 0,0), thickness=2, lineType=8)
    cv2.imshow('image',img2)
    cv2.waitKey(0)

def affine(img):
    img2, van_line = find_vanishLine(img)
    l = van_line/van_line[2]
    H = np.eye(3)
    H[2] = l
    img_new = homography(img2,H)
    return img_new

if __name__=="__main__":
    img = cv2.imread('F:/study material/Advanced CV/assgn2/Garden.JPG')
    parser = argparse.ArgumentParser(description='Assignment 2')
    parser.add_argument('-part', type=int, help='option number')

    args = parser.parse_args()

    if args.part==1:
        draw_line(img)

    elif args.part==2:
        vanished = find_vanishLine(img)

    elif args.part==3:
        transParallel(img)

    else:
        img2 = img[img.shape[0]//2:,:,:]
        img_new = affine(img2)
        cv2.imshow('image',img_new)
        cv2.waitKey(0)
