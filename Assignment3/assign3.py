import cv2
import numpy as np
import copy
from scipy import linalg as LA
import warnings
warnings.filterwarnings("ignore")

def DLT(idx1_norm,idx2_norm):
    num_match = idx1_norm.shape[0]
    
    A = np.empty((num_match,9))
    
    for i in range(num_match):
        x = np.expand_dims(idx2_norm[i],axis=1) @ np.expand_dims(idx1_norm[i],axis=0)
        x = x.ravel()
        A[i] = x
    
    u,s,vt = np.linalg.svd(A)
    F = np.reshape(vt[-1],(3,3))
    
    return F

def find_F_mat(idx1_norm,idx2_norm,norm1,norm2):
    F = DLT(idx1_norm,idx2_norm)
    u,s,vt = np.linalg.svd(F)
    s[-1] = 0
    F_new = np.dot(u, np.dot(np.diag(s), vt))
    
    F_new = np.dot(np.transpose(norm2), np.dot(F_new, norm1))
    F_new = F_new / F_new[2,2]
    
    return F_new



def epipole(F):
    e_l = np.ones(3)
    e_r = np.ones(3)
    M = F[:2,:2]
    C1 = F[:2,2]
    C2 = F[2,:2]
    e_l[:-1] = -np.linalg.inv(M) @ C1
    e_r[:-1] = -np.linalg.inv(M.transpose()) @ C2 
    
    return e_l, e_r


def find_inter(x,y,m):
    y_int=(y[1]-(m[1]*y[0]/m[0])+m[1]*x[0]-m[1]*x[1])/(1-m[1]/m[0])
    x_int=((y_int-y[1])/m[1])+x[1]
    return x_int,y_int

def find_elines(F,img1,img2,idx1_h,idx2_h):
    img_new = copy.deepcopy(img2)
    x=[]
    y=[]
    m=[]
    inter_x=[]
    inter_y=[]
    for i in range(len(idx1_h)):
        l2 = F @ idx1_h[i]
        theta = np.arctan(-l2[0]/l2[1])
        endpt_x = int(idx2_h[i][0] - 1000*np.cos(theta))
        endpt_y = int(idx2_h[i][1] - 1000*np.sin(theta))
        startpt_x = int(idx2_h[i][0] + 1000*np.cos(theta))
        startpt_y = int(idx2_h[i][1] + 1000*np.sin(theta))
        x.append(idx2_h[i][0])
        y.append(idx2_h[i][1])
        m.append(-l2[0]/l2[1])
        if i>1 and i<len(idx1_h)-1:
            x=x[1:]
            y=y[1:]
            m=m[1:]
            a,b=find_inter(x,y,m)
            if a == a and b == b: #eliminating NAN values
                inter_x.append(a)
                inter_y.append(b)
        cv2.line(img_new, (startpt_x,startpt_y), (endpt_x,endpt_y), (0, 255, 0), thickness=2, lineType=8)
    return (np.mean(np.array(inter_x)),np.mean(np.array(inter_y))),img_new

def find_3D(P1,P2,idx1_h,idx2_h):
    C1 = - np.linalg.inv(P1[:,:-1]) @ P1[:,-1]
    C2 = - np.linalg.inv(P2[:,:-1]) @ P2[:,-1]
    dx1 = np.linalg.inv(P1[:,:-1]) @ idx1_h.T
    dx2 = np.linalg.inv(P2[:,:-1]) @ idx2_h.T
    pts_3D = []
    for i in range(dx1.shape[1]):
        a1 = np.linalg.norm(dx1[:,i])
        b1 = - dx1[:,i].dot(dx2[:,i])
        c1 = dx1[:,i].dot(C1-C2)
        a2 = b1
        b2 = - np.linalg.norm(dx2[:,i])
        c2 = dx2[:,i].dot(C1-C2)
        par = - np.linalg.inv(np.array([[a1,b1],[a2,b2]])) @ np.array([[c1],[c2]])
        pt1 = C1 + par[0]*dx1[:,i]
        pt2 = C2 + par[1]*dx2[:,i]
        pts_3D.append((pt1+pt2)/2)
        
    return pts_3D

if __name__ == '__main__':
    
    img1 = cv2.imread('./Amitava_first.JPG',0)  # queryImage
    img2 = cv2.imread('./Amitava_second.JPG',0) # trainImage

    img1_rgb = cv2.imread('./Amitava_first.JPG')
    img2_rgb = cv2.imread('./Amitava_second.JPG')

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.15*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img_feat = cv2.drawMatchesKnn(img1_rgb,kp1,img2_rgb,kp2,good,None,flags=2)

    cv2.imwrite('outputs/img_feat.JPG',img_feat)

    num_match = len(good)

    idx1 = np.array([kp1[good[i][0].queryIdx].pt for i in range(num_match)])
    idx2 = np.array([kp2[good[i][0].trainIdx].pt for i in range(num_match)])

    shape1 = img1.shape
    shape2 = img2.shape

    idx1_h = np.concatenate((idx1,np.ones((num_match,1))),axis=1)
    idx2_h = np.concatenate((idx2,np.ones((num_match,1))),axis=1)

    mean_1 = np.mean(idx1_h[:2], axis=1)
    S1 = np.sqrt(2) / np.std(idx1_h[:2])
    norm1 = np.array([[S1, 0, -S1 * mean_1[0]],
                        [0, S1, -S1 * mean_1[1]],
                        [0, 0, 1]])

    mean_2 = np.mean(idx2_h[:2], axis=1)
    S2 = np.sqrt(2) / np.std(idx2_h[:2])
    norm2 = np.array([[S2, 0, -S2 * mean_2[0]],
                        [0, S2, -S2 * mean_2[1]],
                        [0, 0, 1]])

    idx1_norm = np.transpose(norm1 @ np.transpose(idx1_h))
    idx2_norm = np.transpose(norm2 @ np.transpose(idx2_h))

    F = find_F_mat(idx1_norm, idx2_norm,norm1,norm2)

    print('Fundamental matrix:\n',F,'\n\n')

    # epipoles obtained from epipolar lines
    e_2, img2_ep = find_elines(F,img1,img2,idx1_h,idx2_h)
    e_1, img1_ep = find_elines(F.transpose(),img2,img1,idx2_h,idx1_h)
    cv2.imwrite('outputs/img2_epipolarLine.JPG',img2_ep)
    cv2.imwrite('outputs/img1_epipolarLine.JPG',img1_ep)
    print('Left epipole from epipolar lines:',e_1)
    print('Right epipole from epipolar lines:',e_2,'\n\n')
    
    # epipoles obtained from F matrix
    e_l, e_r = epipole(F)
    e_l = e_l / e_l[-1]
    e_r = e_r / e_r[-1]
    print('Left epipole from F matrix:',e_l[:-1])
    print('Right epipole from F matrix:',e_r[:-1],'\n\n')
    
    # Distances b/w experimental and calculated values 
    dist1 = np.sqrt((e_l[0]-e_1[0])**2 + (e_l[1]-e_1[1])**2)
    dist2 = np.sqrt((e_r[0]-e_2[0])**2 + (e_r[1]-e_2[1])**2)
    print('Distance for left epipole: {:.4f} \n Distance for right epipole: {:.4f}\n\n'.format(dist1, dist2))

    # Projection matrices
    Ex = np.array([[0,-e_r[2],e_r[1]],[e_r[2],0,-e_r[0]],[-e_r[1],e_r[0],0]])
    P1 = np.concatenate((np.eye(3),np.zeros((3,1))),axis=1)
    P2 = np.concatenate((Ex @ F,np.expand_dims(e_r,1)),axis=1)
    print('Left Projection Matrix:\n',P1)
    print('Right Projection Matrix:\n',P2,'\n\n')

    # 3D coordinates
    print()
    pts_3D = find_3D(P1,P2,idx1_h,idx2_h)
    for i in range(len(pts_3D)):
        print('Image 1 coordinate: {} ; Image 2 coordinate: {} ; 3D coordinate: {}'.format(idx1[i],idx2[i],pts_3D[i]))

    # depth comparison

    img_depth = copy.deepcopy(img1_rgb)
    for i in range(len(idx1)):
        color = np.random.randint(0,255,size=(3))
        l = np.random.randint(20,100)
        r,g,b = int(color[0]),int(color[1]),int(color[2])
        x,y=int(idx1[i][0]),int(idx1[i][1])
        cv2.circle(img_depth, (x,y), 2, (r,g,b), 5)
        label='Z = {:.3e}'.format(-pts_3D[i][2])
        cv2.putText(img_depth, label, (img_depth.shape[1]-140,(i+2)*20), cv2.FONT_HERSHEY_PLAIN, 1, (r,g,b), 1)
    state1 = 'THE DEPTH VALUES CORROBORATE WITH RELATIVE 3D POSITIONS' 
    state2 = 'POINTS NEAR BUDDHA\'S FEET HAVE LESS DEPTH THAN THOSE NEAR HIS HEAD/CHEST'
    cv2.putText(img_depth, state1, (10,img_depth.shape[0]-40), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
    cv2.putText(img_depth, state2, (10,img_depth.shape[0]-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
    cv2.imwrite('outputs/img_depth.JPG',img_depth)
    
