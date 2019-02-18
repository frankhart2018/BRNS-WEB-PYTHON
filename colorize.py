import pywt
import time
import numpy as np
from numpy.fft import fft2,ifft2
from  scipy.signal import wiener
from scipy.io import loadmat
from scipy.misc import imsave
import pandas
import matplotlib.pyplot as plt

def lpfilter(filt_type,img):
    F=fft2(img)
    m,n=img.shape
    u,v=np.arange(m),np.arange(n)
    idx,idy=np.argwhere(u>m/2),np.argwhere(v>n/2)
    u[idx],v[idy]=u[idx]-m,v[idy]-n
    V,U=np.meshgrid(v,u)
    d,d0=np.hypot(V,U),0.12*n

    res_img={
        'ideal':
            ifft2(F*np.reshape((d<=d0).astype(np.uint8),F.shape)),
        'butterworth':
            ifft2(F*np.reshape(1/(1+(d/d0)**8),F.shape)),
        'gaussian':
            ifft2(F*np.reshape(np.exp(-d**2/(2*d0**2)),F.shape))
    }
    return res_img[filt_type].astype(np.uint8)

def colorize(im, all=False, fused=False):

    data_folders='./images/'
    print(im)
    exp1=np.load(im)
    noobj20082018=np.load(data_folders+'noobject.npy')

    A=exp1[:512,:640]
    B=exp1[:512,640:1280]

    M=noobj20082018[:512,:640]
    N=noobj20082018[:512,640:1280]

    L=A/M
    H=B/N

    LE=0.5*L**3.8
    HE=0.6*H**0.2
    a1,(h1,v1,d1)=pywt.dwt2(LE,'haar')
    a2,(h2,v2,d2)=pywt.dwt2(HE,'haar')

    k1,k2=a1.shape
    ahvd3=(a1+a2,(h1+h2,v1+v2,d1+d2))

    c=pywt.idwt2(ahvd3,'haar')
    f=(((c-c.min())/(c.max()-c.min()))*255).astype(np.uint8)

    f=np.clip(c,0,1)

    f=(f*255).astype(np.uint8)

    if(fused):
        return f

    L1=wiener(L,[5,5])
    H1=wiener(H,[5,5])

    imfused=f
    le=LE
    he=HE

    x_ax=(L1+H1)/2

    y_ax=H1-L1

    y_s=0.17213*(x_ax**3)-1.399*(x_ax**2)+1.2392*(x_ax)-0.0027535;
    y_p=0.228*(x_ax**3)-0.51622*(x_ax**2)+0.30413*(x_ax)+0.0053274;
    y_al=-0.409873*(x_ax**4)+0.90975*(x_ax**3)-1.2392*(x_ax**2)+0.81073*(x_ax)+0.0018109;

    y_a=(y_p+y_al)/2;
    y_b=(y_al+y_s)/2;

    choice_v1=np.zeros(y_ax.shape)
    choice_v1[y_ax>y_b]=1
    choice_v1[y_ax<y_a]=3
    choice_v1[choice_v1==0]=2
    choice_v1[np.logical_and(x_ax<0.06, y_ax<0.06)]=1

    res=np.zeros((y_ax.size,1,3))
    choice_v1=(choice_v1-1).astype(np.uint8)

    clut_p=pandas.read_excel(data_folders+'clut.xlsx')
    c_clut=[clut_p.iloc[1:,8:8+3].values,clut_p.iloc[1:,4:4+3].values,clut_p.iloc[1:,:3].values]
    chc=[np.expand_dims(i,axis=2) for i in c_clut]

    clut_p=np.concatenate(tuple(chc),axis=2)
    clut_p[67,:,:]=clut_p[66,:,:]
    clut_p[74,:,:]=clut_p[73,:,:]
    clut_p=clut_p.astype(np.uint8)

    clut=loadmat('clutup1.mat')['CLUT'].astype(np.uint8)

    r,c=imfused.shape
    res=np.zeros(f.shape+(3,))
    res=np.array([[clut[imfused[i,j],:,choice_v1[i,j]] for j in range(c)] for i in range(r)])

    mask_res=(choice_v1*0.5).reshape(f.shape)
    for i in range(3):
        res[:,:,i]=lpfilter('gaussian',res[:,:,i])

    if(not all):
        return res

    return res, le, he
