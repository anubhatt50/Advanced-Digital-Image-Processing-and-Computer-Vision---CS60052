import cv2
import numpy as np
import copy

img=cv2.imread('./cavepainting1.JPG')

def conv_gray(img_bgr):
    img_gray=img_bgr[:,:,0]*0.114+img_bgr[:,:,1]*0.587+img_bgr[:,:,2]*0.299
    return img_gray

def convolution(filt,img):
    img_new=np.pad(img,(filt.shape[0]-1)//2,'symmetric')
    img2=np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            crop=np.sum(img_new[i:i+filt.shape[0],j:j+filt.shape[1]]*filt)
            img2[i,j]=crop
    return img2

def G_s(x,y,sig_s):
    return np.exp((-np.linalg.norm(y-x,axis=2)**2)/(2*sig_s**2))

def G_r(img_g,img,sig_r):
    return np.exp((-np.abs(img_g-img)**2)/(2*sig_r**2))

def sc_bil_filt(img,sig_s=4,sig_r=12,sig_g=2,k=(5,5)):
    img_new=np.pad(img,2,'symmetric')
    
    x, y = np.meshgrid(np.linspace(-1,1,5), np.linspace(-1,1,5))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( d**2 / ( 2.0 * sig_g**2 ) ) )    
    g=g/np.sum(g)
        
    img_G=convolution(g,img)   
    img_Gnew=np.pad(img_G,2,'symmetric')
    del img_G
    img=np.zeros((img.shape[0],img.shape[1]))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x=np.array([[[i+a] for a in range(5)]for b in range(5)])
            y=np.array([[[j+b] for a in range(5)]for b in range(5)])
            xy=np.concatenate((y,x),axis=2)
            ij=np.array([[[j+(k[1]-1)//2,i+(k[0]-1)//2] for a in range(5)]for b in range(5)])
            g_s=G_s(xy,ij,sig_s)
            g_r=G_r(img_Gnew[i:i+k[0],j:j+k[1]],img_new[i+(k[0]-1)//2,j+(k[1]-1)//2],sig_r)
            num=np.sum(g_s*g_r*img_new[i:i+k[0],j:j+k[1]])
            den=np.sum(g_s*g_r)
            img[i,j]=num/den
    return img

def sharpen(img):
    filt=-1*np.ones((3,3))
    filt[1,1]=9
    sh_img=convolution(filt,img)
    sh_img[sh_img>255]=255
    return sh_img

def detect_edge(img):
    dog_filter = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
    edge_img = convolution(dog_filter,img)
    return edge_img

def otsu(img):

    img_vect = np.reshape(img,(img.shape[0]*img.shape[1]))

    [hist, _] = np.histogram(img, bins=256, range=(0, 255))
    hist = 1.0*hist/np.sum(hist)

    max_val = -9999
    thresh = -1
    for i in range(1,255):
        n1=np.sum(hist[:i])
        n2=np.sum(hist[i:])
        m1=np.sum(np.array([a for a in range(i)])*hist[:i])/n1
        m2=np.sum(np.array([a for a in range(i,256)])*hist[i:])/n2
        val=n1*(1-n1)*np.power(m1-m2,2)
        if max_val < val:
            max_val = val
            thresh = i
            
    img2=np.zeros((img.shape[0],img.shape[1]))
    img2[img>=thresh]=1
    return img2

def find_connected_comp(img_orig):
    p = 0
    img = np.zeros((img_orig.shape[0]+2,img_orig.shape[1]+2))
    img[1:-1,1:-1] = img_orig
    img_cc = np.zeros_like(img)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            if img[i,j] > 0:
                nbr = [img[i-1,j],img[i,j-1],img[i-1,j-1],img[i-1,j+1]]
                nbr.sort()
                img_cc[i,j] = nbr[-1]
                if nbr[-2] != 0:
                    img_cc[img_cc == nbr[-2]] = nbr[-1]
                if img_cc[i,j] == 0:
                    p += 1
                    img_cc[i,j] = p

    img_cc = img_cc[1:-1,1:-1] 
    return img_cc

def dilation(img,filt=np.ones((3,3))):
    img_new=np.pad(img,(filt.shape[0]-1)//2,'symmetric')
    img2=np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            crop=np.sum(img_new[i:i+filt.shape[0],j:j+filt.shape[1]]*filt)
            if crop>=1:
                img2[i,j]=1
    return img2

def erosion(img,filt=np.ones((3,3))):
    img_new=np.pad(img,(filt.shape[0]-1)//2,'symmetric')
    img2=np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            crop=np.sum(img_new[i:i+filt.shape[0],j:j+filt.shape[1]]*filt)
            if crop<filt.shape[0]*filt.shape[1]:
                img2[i,j]=0
            else:
                img2[i,j]=1
    return img2

def opening(img,filt=np.ones((3,3))):
    ero=erosion(img,filt)
    dil=dilation(ero,filt)
    return dil

def closing(img,filt=np.ones((3,3))):
    dil=dilation(img,filt)
    ero=erosion(dil,filt)
    return ero

def harris(img_otsu):
    d_x=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    d_y=np.transpose(d_x)
    I_x=convolution(d_x,img_otsu)
    I_y=convolution(d_y,img_otsu)
    G=np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273
    I_x2=convolution(G,I_x*I_x)
    I_y2=convolution(G,I_y*I_y)
    I_xy=convolution(G,I_x*I_y)
    det=I_x2*I_y2-I_xy*I_xy
    tr=(I_x2+I_y2)
    R=det-0.05*tr**2
    R[R<np.average(R)]=0

    R2=np.pad(R,2,'symmetric')
    R=np.zeros((R.shape[0],R.shape[1]))
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            x=np.max(R2[i:i+5,j:j+5])
            if R2[i+2,j+2]==x:
                R[i,j]=R2[i+2,j+2]
    corners_x,corners_y = np.nonzero(R)
    img2=np.zeros((img_otsu.shape[0],img_otsu.shape[1],3))
    for i in range(3):
        img2[:,:,i] = copy.deepcopy(img_otsu)
    for i in range(len(corners_x)):
        img2=cv2.circle(img2*255,(corners_y[i],corners_x[i]),radius=4,color=(0,0,255))
    return img2

def skeleton(img,k=5,filt=np.ones((3,3))):
    img2=np.zeros_like(img)
    img_ero=img
    for i in range(k):
        img_ero2=erosion(img_ero,filt)
        img_open=opening(img_ero2,filt)
        s=img_ero2-img_open
        img2=np.logical_or(img2,s)
        img_ero=img_ero2
    return 1*img2

op=int(input("Enter operation number:"))

if op==1:
    img_gray=conv_gray(img)
    cv2.imwrite('grayscale.jpg ',img_gray)

if op==2:
    img_gray=conv_gray(img)
    sc_filt_img=sc_bil_filt(img_gray)
    cv2.imwrite('scaled bilateral filtered image.jpg',sc_filt_img)

if op==3:
    img_gray=conv_gray(img)
    sc_filt_img=sc_bil_filt(img_gray)
    img_sh = sharpen(sc_filt_img)
    cv2.imwrite('sharpened image.jpg',img_sh)

if op==4:
    img_gray=conv_gray(img)
    sc_filt_img=sc_bil_filt(img_gray)
    edge_img = detect_edge(img_gray)
    cv2.imwrite('edge image.jpg',edge_img)

if op==5:
    img_gray=conv_gray(img)
    sc_filt_img=sc_bil_filt(img_gray)
    img_sh = sharpen(sc_filt_img)
    img_otsu = otsu(img_sh)
    cv2.imwrite('otsu thresholded img.jpg',img_otsu*255.)
    
if op==6:
    img_gray=conv_gray(img)
    sc_filt_img=sc_bil_filt(img_gray)
    img_sh = sharpen(sc_filt_img)
    img_otsu = otsu(img_sh)
    img_cc = find_connected_comp(img_otsu)
    img_cc2=np.zeros((img_cc.shape[0],img_cc.shape[1],3))
    img_cc2[:,:,0] = (img_cc%20)*10
    img_cc2[:,:,1] = 255-(img_cc%20)*10
    img_cc2[:,:,2] = (img_cc%20)*10
    cv2.imwrite('connected component image.jpg',img_cc2)
    img_skel = skeleton(img_otsu)
    cv2.imwrite('connected component image as line segments (skeleton).jpg',img_skel*255.)

if op==7:
    img_gray=conv_gray(img)
    sc_filt_img=sc_bil_filt(img_gray)
    img_sh = sharpen(sc_filt_img)
    img_otsu = otsu(img_sh)
    img_ero = erosion(img_otsu)
    img_dil = dilation(img_otsu)
    img_open = opening(img_otsu)
    img_cl = closing(img_otsu)
    cv2.imwrite('eroded image.jpg',img_ero*255.)
    cv2.imwrite('dilated image.jpg',img_dil*255.)
    cv2.imwrite('opening.jpg',img_open*255.)
    cv2.imwrite('closing.jpg',img_cl*255.)

if op==8:
    img_gray=conv_gray(img)
    sc_filt_img=sc_bil_filt(img_gray)
    img_sh = sharpen(sc_filt_img)
    img_otsu = otsu(img_sh)
    img_harris = harris(img_otsu)
    cv2.imwrite('Harris corner image.jpg',img_harris)
