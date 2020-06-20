import cv2
import numpy as np
import copy
import argparse

def convolution(filt,img):
    img_new=np.pad(img,(filt.shape[0]-1)//2,'symmetric')
    img2=np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            crop=np.sum(img_new[i:i+filt.shape[0],j:j+filt.shape[1]]*filt)
            img2[i,j]=crop
    return img2

def map_nmf(img): #3x3 depth image patch to 3x3x3 neighborhood
    nmf = np.zeros((img.shape[0],img.shape[1],3,3,3))
    img_new=np.pad(img,1,'symmetric')
    for i in range(1,img.shape[0]+1):
        for j in range(1,img.shape[1]+1):
            nmf[i-1,j-1,1,:,:] = img_new[i-1:i+2,j-1:j+2]
            nmf[i-1,j-1,1,:,:] = nmf[i-1,j-1,1,:,:]-img_new[i,j]
            for a in range(3):
                for b in range(3):
                    if nmf[i-1,j-1,1,a,b] == 1:
                        nmf[i-1,j-1,2,a,b] = 1
                        nmf[i-1,j-1,1,a,b] = 0
                    elif nmf[i-1,j-1,1,a,b] == -1:
                        nmf[i-1,j-1,0,a,b] = 1
                        nmf[i-1,j-1,1,a,b] = 0
                    elif nmf[i-1,j-1,1,a,b] == 0:
                        nmf[i-1,j-1,1,a,b] = 1
                    else:
                        nmf[i-1,j-1,:,a,b] = 0
    return nmf

def find_nps(nmf,k): #compute nps of each pixel
    nps = np.zeros((nmf.shape[0],nmf.shape[1]))
    for i in range(nmf.shape[0]):
        for j in range(nmf.shape[1]):
            if np.sum(nmf[i,j,1,:,:])>=k:
                nps[i,j]+=2**0
            if np.sum(nmf[i,j,:,1,:])>=k:
                nps[i,j]+=2**1
            if np.sum(nmf[i,j,:,:,1])>=k:
                nps[i,j]+=2**2
            if np.sum(nmf[i,j,0,0,:])+np.sum(nmf[i,j,1,1,:])+np.sum(nmf[i,j,2,2,:])>=k:
                nps[i,j]+=2**3
            if np.sum(nmf[i,j,0,:,0])+np.sum(nmf[i,j,1,:,1])+np.sum(nmf[i,j,2,:,2])>=k:
                nps[i,j]+=2**4
            if np.sum(nmf[i,j,:,0,0])+np.sum(nmf[i,j,:,1,1])+np.sum(nmf[i,j,:,2,2])>=k:
                nps[i,j]+=2**5
            if np.sum(nmf[i,j,0,2,:])+np.sum(nmf[i,j,1,1,:])+np.sum(nmf[i,j,2,0,:])>=k:
                nps[i,j]+=2**6
            if np.sum(nmf[i,j,0,:,2])+np.sum(nmf[i,j,1,:,1])+np.sum(nmf[i,j,0,:,2])>=k:
                nps[i,j]+=2**7
            if np.sum(nmf[i,j,:,0,2])+np.sum(nmf[i,j,:,1,1])+np.sum(nmf[i,j,:,2,0])>=k:
                nps[i,j]+=2**8
    return nps

def con_cmp_curv(d_img,curv_img): #connected components
    p = 0
    img_orig_d = np.zeros((d_img.shape[0]+2,d_img.shape[1]+2))-1
    img_orig_d[1:-1,1:-1] = d_img
    orig_curv = np.zeros((d_img.shape[0]+2,d_img.shape[1]+2,3))-1
    orig_curv[1:-1,1:-1] = curv_img
    img_cc = np.zeros_like(img_orig_d)
    for i in range(1,orig_curv.shape[0]-1):
        for j in range(1,orig_curv.shape[1]-1):
            nbr_d = np.array([img_orig_d[i-1,j],img_orig_d[i,j-1],img_orig_d[i-1,j-1],img_orig_d[i-1,j+1]])
            nbr_curv = np.array([orig_curv[i-1,j],orig_curv[i,j-1],orig_curv[i-1,j-1],orig_curv[i-1,j+1]])
            nbr_cc = np.array([img_cc[i-1,j],img_cc[i,j-1],img_cc[i-1,j-1],img_cc[i-1,j+1]])
            nbr_d2=np.where(nbr_d>img_orig_d[i,j]-2)
            nbr_d3=np.where(nbr_d<img_orig_d[i,j]+2)
            nbr_curv2=np.where(nbr_curv==orig_curv[i,j])
            nbr_cc2=np.intersect1d(nbr_d2,np.intersect1d(nbr_d3,nbr_curv2))
            if len(nbr_cc2)==0:
                p += 1
                img_cc[i,j] = p
                continue
            img_cc[i,j] = nbr_cc[nbr_cc2[-1]]
            if len(nbr_cc2)>1:
                img_cc[img_cc == nbr_cc[nbr_cc2[-2]]] = nbr_cc[nbr_cc2[-1]]

    img_cc = img_cc[1:-1,1:-1]
    return img_cc

def con_cmp(d_img,nps_img): #connected components (nps)
    p = 0
    img_orig_d = np.zeros((d_img.shape[0]+2,d_img.shape[1]+2))-1
    img_orig_d[1:-1,1:-1] = d_img
    orig_nps = np.zeros((d_img.shape[0]+2,d_img.shape[1]+2))-1
    orig_nps[1:-1,1:-1] = nps_img
    img_cc = np.zeros_like(img_orig_d)
    for i in range(1,orig_nps.shape[0]-1):
        for j in range(1,orig_nps.shape[1]-1):
            nbr_d = np.array([img_orig_d[i-1,j],img_orig_d[i,j-1],img_orig_d[i-1,j-1],img_orig_d[i-1,j+1]])
            nbr_nps = np.array([orig_nps[i-1,j],orig_nps[i,j-1],orig_nps[i-1,j-1],orig_nps[i-1,j+1]])
            nbr_cc = np.array([img_cc[i-1,j],img_cc[i,j-1],img_cc[i-1,j-1],img_cc[i-1,j+1]])
            nbr_d2=np.where(nbr_d>img_orig_d[i,j]-2)
            nbr_d3=np.where(nbr_d<img_orig_d[i,j]+2)
            nbr_nps2=np.where(nbr_nps==orig_nps[i,j])
            nbr_cc2=np.intersect1d(nbr_d2,np.intersect1d(nbr_d3,nbr_nps2))
            if len(nbr_cc2)==0:
                p += 1
                img_cc[i,j] = p
                continue
            img_cc[i,j] = nbr_cc[nbr_cc2[-1]]
            if len(nbr_cc2)>1:
                img_cc[img_cc == nbr_cc[nbr_cc2[-2]]] = nbr_cc[nbr_cc2[-1]]

    img_cc = img_cc[1:-1,1:-1]
    return img_cc

def smoothen(cc_img): #smoothen segments
    cc_img2 = np.pad(cc_img,7,'symmetric')
    for i in (np.unique(cc_img)):
        if len(np.where(cc_img==i)[0])<=150:
            cc_img2[cc_img2==i]=0
    while len(np.where(cc_img2[7:cc_img.shape[0]+7,7:cc_img.shape[1]+7]==0)[0])!=0:
        x = np.where(cc_img2[7:cc_img.shape[0]+7,7:cc_img.shape[1]+7]==0)[0]
        y = np.where(cc_img2[7:cc_img.shape[0]+7,7:cc_img.shape[1]+7]==0)[1]
        xy = np.stack((x,y),axis=1)
        for i in range(len(xy)):
            x = xy[i][0]
            y = xy[i][1]
            nbr = cc_img2[x:x+15,y:y+15]
            m = np.bincount(nbr.astype(np.int).ravel()).argmax()
            if m==0:
                nbr=nbr[nbr!=m]
                l = np.bincount(nbr.astype(np.int).ravel())
                if len(l)==0:
                    continue
                else:
                    m=l.argmax()
            cc_img2[x+7,y+7] = m
        
    cc_img2 = cc_img2[7:-7,7:-7]
    
    for i in range(len(np.unique(cc_img2))):
        cc_img2[cc_img2==np.unique(cc_img2)[i]] = i
                
    return cc_img2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Assignment 5')
    parser.add_argument('--img_number', type=int, help='Image Number',default = 1)
    args = parser.parse_args()

    img_bgr = cv2.imread('RGBD_dataset/'+str(args.img_number)+'.jpg')
    img_d = cv2.imread('RGBD_dataset/'+str(args.img_number)+'.png')
    img_d = img_d[:,:,0]

    sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    h_x = convolution(sobel_x,img_d)
    h_y = convolution(sobel_y,img_d)
    h_xx = convolution(sobel_x,h_x)
    h_yy = convolution(sobel_y,h_y)
    h_xy = convolution(sobel_y,h_x)

    h_x_sq = np.expand_dims(h_x*h_x,axis=2)
    h_y_sq = np.expand_dims(h_y*h_y,axis=2)

    s = (np.sum(np.concatenate((h_x_sq,h_y_sq,np.ones_like(h_x_sq)),axis=2),axis=2))**0.5
    N = np.concatenate((np.expand_dims(-h_x,axis=2),np.expand_dims(-h_y,axis=2),np.ones_like(np.expand_dims(-h_x,axis=2))),axis=2)/np.expand_dims(s,axis=2)

    E = np.squeeze(1+h_x_sq)
    F = np.squeeze(1+h_x*h_y)
    G = np.squeeze(1+h_y_sq)

    e = -h_xx/s
    f = -h_xy/s
    g = -h_yy/s

    H = (E*g + G*e -2*F*f)/(2*(E*G-F*F)+1e-8) #Mean curvature
    K = (e*g-f*f)/(E*G-F*F+1e-8) #Gaussian curvature

    A = H*H-K
    A[A<0]=0

    k1 = H+A**0.5 #principal curvatures
    k2 = H-A**0.5

    guide_pr = np.zeros((200,300,3))
    guide_pr[0:100,1:100,:] = np.array([0,0,255])
    guide_pr[0:100,100:200,:] = np.array([0,128,128])
    guide_pr[0:100,200:300,:] = np.array([0,255,0])
    guide_pr[100:200,1:100,:] = np.array([128,128,0])
    guide_pr[100:200,100:200,:] = np.array([128,0,128])
    guide_pr[100:200,200:300,:] = np.array([255,0,0])
    cv2.putText(guide_pr, "PEAK", (20,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_pr, "RIDGE", (120,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_pr, "SADDLE", (220,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_pr, "FLAT", (20,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_pr, "VALLEY", (120,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_pr, "PIT", (220,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.imwrite('outputs/output'+str(args.img_number)+'/Principal_curvature_col_chart.JPG',guide_pr)

    curv_pr = np.zeros_like(img_bgr)
    for i in range(curv_pr.shape[0]):
        for j in range(curv_pr.shape[1]):
            if k1[i,j]<0 and k2[i,j]<0: #peak
                curv_pr[i,j,:]=np.array([0,0,255])
            elif (k1[i,j]<0 and k2[i,j]==0) or (k1[i,j]==0 and k2[i,j]<0): #ridge
                curv_pr[i,j,:]=np.array([0,128,128])
            elif (k1[i,j]<0 and k2[i,j]>0) or (k1[i,j]>0 and k2[i,j]<0): #saddle
                curv_pr[i,j,:]=np.array([0,255,0])
            elif k1[i,j]==0 and k2[i,j]==0: #flat
                curv_pr[i,j,:]=np.array([128,128,0])
            elif (k1[i,j]>0 and k2[i,j]==0) or (k1[i,j]==0 and k2[i,j]>0): #valley
                curv_pr[i,j,:]=np.array([128,0,128])
            elif k1[i,j]>0 and k2[i,j]>0: #pit
                curv_pr[i,j,:]=np.array([255,0,0])
                
    cv2.imwrite('outputs/output'+str(args.img_number)+'/Principal_curvature_img.JPG',curv_pr)

    guide_mean_gauss = np.zeros((300,300,3))
    guide_mean_gauss[0:100,1:100,:] = np.array([0,0,255])
    guide_mean_gauss[0:100,100:200,:] = np.array([0,255,0])
    guide_mean_gauss[0:100,200:300,:] = np.array([255,0,0])
    guide_mean_gauss[100:200,1:100,:] = np.array([64,192,64])
    guide_mean_gauss[100:200,100:200,:] = np.array([192,128,64])
    guide_mean_gauss[100:200,200:300,:] = np.array([192,64,255])
    guide_mean_gauss[200:300,1:100,:] = np.array([64,192,255])
    guide_mean_gauss[200:300,100:200,:] = np.array([255,64,128])
    guide_mean_gauss[200:300,200:300,:] = np.array([64,64,192])
    cv2.putText(guide_mean_gauss, "PEAK", (20,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_mean_gauss, "RIDGE", (120,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_mean_gauss, "SADDLE", (220,50), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_mean_gauss, "RIDGE", (220,70), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_mean_gauss, "NONE", (20,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_mean_gauss, "FLAT", (120,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_mean_gauss, "MINIMAL", (220,150), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_mean_gauss, "SURFACE", (220,170), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_mean_gauss, "PIT", (20,250), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_mean_gauss, "VALLEY", (120,250), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_mean_gauss, "SADDLE", (220,250), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.putText(guide_mean_gauss, "VALLEY", (220,270), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
    cv2.imwrite('outputs/output'+str(args.img_number)+'/MeanGauss_curvature_col_chart.JPG',guide_mean_gauss)

    curv_mean_gauss = np.zeros_like(img_bgr)
    for i in range(curv_mean_gauss.shape[0]):
        for j in range(curv_mean_gauss.shape[1]):
            if H[i,j]<0 and K[i,j]<0: #peak
                curv_mean_gauss[i,j,:]=np.array([0,0,255])
            elif H[i,j]<0 and K[i,j]==0: #ridge
                curv_mean_gauss[i,j,:]=np.array([0,255,0])
            elif H[i,j]<0 and K[i,j]>0: #saddle ridge
                curv_mean_gauss[i,j,:]=np.array([255,0,0])
            elif H[i,j]==0 and K[i,j]<0: #none
                curv_mean_gauss[i,j,:]=np.array([64,192,64])
            elif H[i,j]==0 and K[i,j]==0: #flat
                curv_mean_gauss[i,j,:]=np.array([192,128,64])
            elif H[i,j]==0 and K[i,j]>0: #minimal surface
                curv_mean_gauss[i,j,:]=np.array([192,64,255])
            elif H[i,j]>0 and K[i,j]<0: #pit
                curv_mean_gauss[i,j,:]=np.array([64,192,255])
            elif H[i,j]>0 and K[i,j]==0: #valley
                curv_mean_gauss[i,j,:]=np.array([255,64,128])
            elif H[i,j]>0 and K[i,j]>0: #saddle valley
                curv_mean_gauss[i,j,:]=np.array([64,64,192])
                
    cv2.imwrite('outputs/output'+str(args.img_number)+'/MeanGauss_curvature_img.JPG',curv_mean_gauss)

    nmf=map_nmf(img_d)
    nps=find_nps(nmf,5)

    img_cc_pr = con_cmp_curv(img_d,curv_pr)
    img_cc_gauss = con_cmp_curv(img_d,curv_mean_gauss)
    img_cc_nps = con_cmp(img_d,nps)
    cc_pr=smoothen(img_cc_pr)
    cc_gauss=smoothen(img_cc_gauss)
    cc_nps=smoothen(img_cc_nps)
    cc_nps2 = np.stack(((cc_nps.max()-cc_nps)*255/cc_nps.max(),(cc_nps)*255/cc_nps.max(),(cc_nps)*255/cc_nps.max()),axis=2)
    cv2.imwrite('outputs/output'+str(args.img_number)+'/Segmented_nps_img.JPG',cc_nps2)
    cc_pr2 = np.stack(((cc_pr.max()-cc_pr)*255/cc_pr.max(),(cc_pr)*255/cc_pr.max(),(cc_pr)*255/cc_pr.max()),axis=2)
    cv2.imwrite('outputs/output'+str(args.img_number)+'/Segmented_principalCurvature_img.JPG',cc_pr2)
    cc_gauss2 = np.stack(((cc_gauss.max()-cc_gauss)*255/cc_gauss.max(),(cc_gauss)*255/cc_gauss.max(),(cc_gauss)*255/cc_gauss.max()),axis=2)
    cv2.imwrite('outputs/output'+str(args.img_number)+'/Segmented_GaussianCurvature_img.JPG',cc_gauss2)
