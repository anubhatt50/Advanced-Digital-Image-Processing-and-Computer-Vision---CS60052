import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse

posList = []
def onMouse(event, x, y, flags, param):
   global posList
   if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))

#utility functions

def conv2CIE(img): #convert to CIE (XYZ)
    bgr2XYZ = np.array([[0.2001,0.1736,0.6067],[0.1143,0.5868,0.2988],[1.1149,0.0661,0]]) #[Z,Y,X]
    img_CIE = bgr2XYZ @ img
    return img_CIE

def convCIE2D(XYZ): #convert to CIE 2D (xy)
    s = np.sum(XYZ)
    x = XYZ[0]/s+1e-6
    y = XYZ[1]/s+1e-6
    return x,y

def conv2lab(img_crop,ind_dom): #convert to lab space
    BGR2lms = np.array([[0.0402,0.5783,0.3811],[0.0782,0.7244,0.1967],[0.8444,0.1288,0.0241]])
    lab1 = np.array([[1.0/(3**0.5),0,0],[0,1.0/(6**0.5),0],[0,0,1.0/(2**0.5)]])
    lab2 = np.array([[1,1,1],[1,1,-2],[1,-1,0]])
    lab_arr = np.zeros((len(ind_dom),3))
    for i in range(len(ind_dom)):
        lms = BGR2lms @ img_crop[ind_dom[i][0],ind_dom[i][1]]
        LMS = np.log(lms)
        lab = lab1 @ lab2 @ LMS
        lab_arr[i,:] = lab
    return lab_arr

def cluster(img,n_clust): #K means clustering to find mode
    kmeans = KMeans(n_clusters = n_clust)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    return colors,labels

def dom_pixel(idx,crop_img,dom_col): #compute and show dominant pixels
    crop_new = copy.deepcopy(crop_img)
    ind_dom = []
    for i in range(crop_img.shape[0]):
        for j in range(crop_img.shape[1]):
            cie = conv2CIE(crop_img[i,j])
            x,y = convCIE2D(cie)
            if np.array([x,y]) in idx:
                if np.allclose(np.array([x,y]),dom_col,atol=1e-3)==True:
                    crop_dom = np.ones((100,100,3),dtype=np.uint8)*crop_img[i,j]
                crop_new[i,j]=np.array([255,255,255])
                ind_dom.append([i,j])
    return ind_dom,crop_new,crop_dom

def estimate_dom_col(img): #Function to estimate dominant colour
    global posList
    posList = []
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', onMouse)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    poslist = np.array(posList)
    img_crop = img[min([posList[0][1],posList[1][1]]):max([posList[0][1],posList[1][1]]),min([posList[0][0],posList[1][0]]):max([posList[0][0],posList[1][0]])]
    
    img_cie = copy.deepcopy(img_crop)
    img_cie = img_cie.reshape(img_cie.shape[0]*img_cie.shape[1],3).T
    img_cie = conv2CIE(img_cie)
    
    index = np.zeros((img_cie.shape[1],2))
    for i in range(img_cie.shape[1]):
        x,y=convCIE2D(img_cie[:,i])
        if x!=x or y!=y: continue
        index[i,:]=np.array([x,y])
    
    col,lbl=cluster(index,3)
    val = np.bincount(lbl).argmax( )
    index2 = index[lbl==val]

    dom_col = col[val]
    
    ind_dom,img_dom, dom_col= dom_pixel(index2,img_crop,dom_col)
    
    return img_crop, img_dom,ind_dom,dom_col,index,index2,[poslist[0],poslist[1]]

def transfer(img_src_crop,lab_src,lab_tgt,ind_dom_src): #Function for dominant colour transfer
    lab_src_mean = np.mean(lab_src,axis=0)
    lab_tgt_mean = np.mean(lab_tgt,axis=0)
    lab_src_std = np.std(lab_src,axis=0)
    lab_tgt_std = np.std(lab_tgt,axis=0)
    lab_star = lab_src - lab_src_mean
    lab_hat = lab_tgt_std*lab_star/lab_src_std
    lab_m = lab_hat + lab_tgt_mean
    
    lms1 = np.array([[1,1,1],[1,1,-1],[1,-2,0]])
    lms2 = np.array([[1.0/(3**0.5),0,0],[0,1.0/(6**0.5),0],[0,0,1.0/(2**0.5)]])
    LMS = lms1 @ lms2 @ lab_m.T
    lms = np.exp(LMS)
    
    lms2RGB = np.array([[4.4679,-3.5873,0.1193],[-1.2186,2.3809,-0.1624],[0.0497,-0.2439,1.2045]])
    RGB = lms2RGB @ lms
    RGB[RGB>255]=255
    
    img_new_crop = copy.deepcopy(img_src_crop)
    for i in range(len(ind_dom_src)):
        img_new_crop[ind_dom_src[i][0],ind_dom_src[i][1]] = np.array([RGB[2,i],RGB[1,i],RGB[0,i]])
    return img_new_crop

#Save results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Assignment 4')
    parser.add_argument('--source_img', type=str, help='Source image name',default = 'IMG_6481.jpg')
    parser.add_argument('--target_img', type=str, help='Target image name',default = 'IMG_6479.jpg')
    args = parser.parse_args()

    #Loading images

    src_dst = './'+args.source_img
    tgt_dst = './'+args.target_img
    img_src = cv2.imread(src_dst) #source
    img_tgt = cv2.imread(tgt_dst) #target

    img_src_crop,img_src_dom,ind_dom_src,dom_src_col,index_src,index2_src,pos_src=estimate_dom_col(img_src)
    cv2.imwrite('outputs/cropped_source.JPG',img_src_crop) #cropped source image
    cv2.imwrite('outputs/cropped_source_dominant_pixels.JPG',img_src_dom) #cropped source image with dominant pixels
    cv2.imwrite('outputs/source_dominant_colour.JPG',dom_src_col) #source image dominant colour
    plt.figure()
    plt.plot(index_src[:,0],index_src[:,1],'ro',markersize=1)
    plt.title('CIE chromaticity diagram of source image') 
    plt.savefig('outputs/CIEdiag_source.jpg') #CIE diagram of source image pixels
    plt.figure()
    plt.plot(index_src[:,0],index_src[:,1],'ro',markersize=1)
    plt.plot(index2_src[:,0],index2_src[:,1],'go',markersize=1)
    plt.title('CIE chromaticity diagram of source image (dominant pixels)')
    plt.savefig('outputs/CIEdiag_source_dominant.jpg') #CIE diagram of source image dominant pixels after finding dominant cluster from K means

    img_tgt_crop,img_tgt_dom,ind_dom_tgt,dom_tgt_col,index_tgt,index2_tgt,pos_tgt=estimate_dom_col(img_tgt)
    cv2.imwrite('outputs/cropped_target.JPG',img_tgt_crop) #cropped target image
    cv2.imwrite('outputs/cropped_target_dominant_pixels.JPG',img_tgt_dom) #cropped target image with dominant pixels
    cv2.imwrite('outputs/target_dominant_colour.JPG',dom_tgt_col) #target image dominant colour
    plt.figure()
    plt.plot(index_tgt[:,0],index_tgt[:,1],'ro',markersize=1)
    plt.title('CIE chromaticity diagram of target image')
    plt.savefig('outputs/CIEdiag_target.jpg') #CIE diagram of target image pixels
    plt.figure()
    plt.plot(index_tgt[:,0],index_tgt[:,1],'ro',markersize=1)
    plt.plot(index2_tgt[:,0],index2_tgt[:,1],'go',markersize=1)
    plt.title('CIE chromaticity diagram of target image (dominant pixels)')
    plt.savefig('outputs/CIEdiag_target_dominant.jpg') #CIE diagram of target image dominant pixels after finding dominant cluster from K means

    lab_src = conv2lab(img_src_crop,ind_dom_src)
    lab_tgt = conv2lab(img_tgt_crop,ind_dom_tgt)

    img_new_crop = transfer(img_src_crop,lab_src,lab_tgt,ind_dom_src) #dominant colour transfer
    cv2.imwrite('outputs/result_cropped.JPG',img_new_crop)
    img_new = copy.deepcopy(img_src)
    img_new[min([pos_src[0][1],pos_src[1][1]]):max([pos_src[0][1],pos_src[1][1]]),min([pos_src[0][0],pos_src[1][0]]):max([pos_src[0][0],pos_src[1][0]])]=img_new_crop
    cv2.imwrite('outputs/result.JPG',img_new) #final result
