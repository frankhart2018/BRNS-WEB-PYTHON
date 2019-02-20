import pywt
import numpy as np
from  scipy.signal import wiener
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2


def loadimgfile(fpath):
    extension=fpath.split('.')[1]
    if extension=='txt':
        return np.loadtxt(fpath)
    elif extension=='npy':
        return np.load(fpath)



def getAllFilesInFolder(folderpath):
    return [os.path.join(root,file) for root,dir,files in os.walk(folderpath) for file in files]

class BRNSProcessing:
    clut=loadmat('eclut.mat')['CLUT']
    noobj=loadimgfile('NOOBJECT_20-08-2018.txt')
    def __init__(self,imgfpath):
        self.loadLH(imgfpath)
        self.generateFusion()
        self.generateChc()
        self.generateFColor()

    def loadLH(self,imgfpath):
        img=loadimgfile(imgfpath)
        M=BRNSProcessing.noobj[:512,:640]
        N=BRNSProcessing.noobj[:512,640:1280]
        A=img[:512,:640]
        B=img[:512,640:1280]
        self.L=A/M
        self.H=B/N

    def generateFusion(self):
        LE=self.L**3.2
        HE=self.H**0.2
        a1,(h1,v1,d1)=pywt.dwt2(LE,'haar')
        a2,(h2,v2,d2)=pywt.dwt2(HE,'haar')
        self.imfused=(np.clip(pywt.idwt2((0.6*a1+a2/2,(h1+h2,v1+v2,d1+d2)),'haar'),0,1)*255).astype(np.uint8)


    def generateChc(self):
        L1=wiener(self.L,[5,5])
        H1=wiener(self.H,[5,5])
        
        x_ax=(L1+H1)/2
        y_ax=H1-L1
            
        y_s=0.17213*(x_ax**3)-1.399*(x_ax**2)+1.2392*x_ax-0.0027535
        y_p=0.228*(x_ax**3)-0.51622*(x_ax**2)+0.30413*x_ax+0.0053274
        y_al=-0.409873*(x_ax**4)+0.90975*(x_ax**3)-1.298*(x_ax**2)+0.81073*x_ax+0.0018109
        y_a=(y_p+y_al)/2
        y_b=(y_al+y_s)/2
            
        self.choice=np.zeros(y_ax.shape)
        self.choice[y_ax>y_b]=1
        self.choice[self.choice==0]=2
        self.choice[y_ax<y_a]=3
        
        a=L1;b=H1;c=np.log(a);d=np.log(b)
        q=c/d
        self.choice[(q<1.17)&(self.H<0.19)&(self.L<0.16)]=4
        self.choice[(q<1.24)&(self.H<0.42)&(self.L<0.3)]=4
        self.choice[(x_ax<0.06) & (y_ax<0.06)]=1;
        self.choice= (self.choice-1).astype(np.uint8)

    def generateFColor(self):
        r,c=self.imfused.shape
        self.pc_img=np.array([[BRNSProcessing.clut[self.imfused[i,j],:,self.choice[i,j]] for j in range(c)] for i in range(r)])
        self.pc_img=cv2.bilateralFilter(np.uint8(self.pc_img),6,157,157)/255

    def genGrayImg(self):
        return np.stack((cv2.cvtColor(np.uint8(self.pc_img*255),cv2.COLOR_RGB2GRAY),)*3,axis=2)/255

    def genVCplus(self):
        res1=cv2.cvtColor(np.uint8(self.pc_img*255),cv2.COLOR_RGB2HSV);
        ress=res1[...,1];
        resh=res1[...,0];
        ress[(resh<95)]=0;
        return cv2.cvtColor(np.stack((resh,ress,res1[...,2]),axis=2),cv2.COLOR_HSV2RGB);

    def genVCminus(self):
        res1=cv2.cvtColor(np.uint8(self.pc_img*255),cv2.COLOR_RGB2HSV);
        ress=res1[...,1];
        resh=res1[...,0];
        ress[(resh>95)]=0;
        return cv2.cvtColor(np.stack((resh, ress,res1[...,2]),axis=2),cv2.COLOR_HSV2RGB);

    def adjust_gamma(self, gamma):
       invGamma = 1.0 / gamma
       
       gamma_table = np.array([((i / 255.0) ** invGamma) * 255
          for i in np.arange(0, 256)]).astype("uint8")
       return cv2.LUT(np.uint8(self.pc_img*255), gamma_table)

    def genVDImg(self,val):
        if val>1:
            val=2-val;
            return self.adjust_gamma(pc_img, val)
        else:
            res_g=np.stack((cv2.cvtColor(np.uint8(self.pc_img*255),cv2.COLOR_RGB2GRAY),)*3,axis=2)
            res=(255 - cv2.threshold(res_g,(1-val)*255,255,cv2.THRESH_BINARY)[1])/255
            return self.pc_img+res


    def genIMImg(self):
        res=np.array(self.pc_img,copy=True)
        res[self.choice!=1]=1
        return res

    def genOMImg(self):
        res=np.array(self.pc_img,copy=True)
        res[self.choice!=2]=1
        return res

    def genVEImg(self,scale_factor):
        res_g=np.stack((cv2.cvtColor(np.uint8(self.pc_img*255),cv2.COLOR_RGB2GRAY),)*3,axis=2)
        resve_ul=4**self.pc_img*((255-res_g)/255) \
                 +2*self.pc_img*(255 - cv2.threshold(res_g,0.52*255,255,cv2.THRESH_BINARY)[1])/255
        resve_ll=0.8*self.pc_img \
                 +0.2*self.pc_img*(255 - cv2.threshold(res_g,0.95*255,255,cv2.THRESH_BINARY)[1])/255
        resve_diff=resve_ul-resve_ll
        return resve_ll+scale_factor*resve_diff

    def genOvsBImg(self):
        res=np.stack((cv2.cvtColor(np.uint8(self.pc_img*255),cv2.COLOR_RGB2GRAY),)*3,axis=2)/255
        return cv2.cvtColor(np.uint8(self.pc_img*(4*(1-res))*255),cv2.COLOR_RGB2GRAY) #cmap='pink'

    def genInvImg(self):
        res_g1=np.stack((cv2.cvtColor(np.uint8(self.pc_img*255),cv2.COLOR_RGB2GRAY),)*3,axis=2)
        return 4**self.pc_img*((255-res_g1)/255)

    def genCCImg(self):
        LE=self.L**3.2
        HE=self.H**0.2
        LE=np.uint8(np.clip(LE*255,0,255))
        HE=np.uint8(np.clip(HE*255,0,255))
        return np.concatenate((generateFColor(LE,BRNSProcessing.clut,self.choice),generateFColor(HE,BRNSProcessing.clut,self.choice)),axis=1)

    def genHSIImg(self):
        R=pc_img[...,0]
        G=pc_img[...,1]
        B=pc_img[...,2]
        RGB=pc_img.sum(axis=2)
        r=R/RGB
        g=G/RGB
        b=B/RGB
        H=np.zeros(r.shape)
        num=(0.8*R+0.88*G+0.000000001*B);
        den=np.sqrt(2*(R**2+G**2+B**2-(R*G)+(R*B)+(G*B)));
        th=num/den;
        the=np.arccos(th);
        H=(B>G)*(360-the)+(B<=G)*the;
        a=pc_img.max(axis=2); b=pc_img.min(axis=2);
        S=a+b;
        I=R+G+B/3;
        return np.stack(( H, S, I),axis=2)



"""
import brns_processing as bp
bp.BRNSProcessing('STTEST1.txt')
#"""