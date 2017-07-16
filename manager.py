import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import regex as re
from PIL import Image,ImageEnhance
import requests
from io import BytesIO
import pytesseract
import urllib.request
import pickle




class doc_name:
    
    def __init__(self,url):
        
        self.url = url
        self.ns = pickle.load(open('doc_name_surname.p','rb'))
        
        
    def url_to_image(self,url):
        #converts url to image

        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
        return (image)
    
    def cleaned(self,text):

        #detects doctor's name from the text extracted, matches strings and sub-strings,returns entire name
  
        text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
        text=' '.join(text.split())
    
        line=text.strip().split(' ')
   
        for i in range(1,len(line)-1):
        #print (i,line[i])
            for j in range(i+1,i+2):
        
                if ((line[i],line[j]) in self.ns) and ( len(line[j])!=1):
           
                    return (line[i] + ' '+ line[j])
                if ((((line[i-1]+' '+line[i]),line[j]) in self.ns) and len(line[j])!=1):
                
                    return (line[i-1]+'.'+line[i]+ ' '+ line[j])
                elif (line[i] in [x[0] for x in self.ns]):
                
                    if len(line[j])>1:
                        #print ('place 1',line[i])
                    
                        for v in [x[1] for x in self.ns]:
                            if ((len(line[i])==1 and len(line[i-1])==1)):
                            
                            #print (line[j])
                            
                                if ((line[j][:2]==v[:2]) and (line[j][-1]==v[-1])) and ((line[i-1]+' '+line[i],v) in self.ns):
                                
                                    return (line[i-1]+' '+line[i] + ' '+ v)
                            
                            if len(line[i])>1:
                            
                                if (line[j][:3]==v[:3]) and ((line[i],v) in self.ns):
                                
                                
                                    return (line[i] + ' '+ v)
                                
                elif (line[j] in [x[1] for x in self.ns]):
            
                    if len(line[j])>2:
                        #print ('place 2', line[j])
                    
                        for v in [x[0] for x in self.ns]:
                  
                            
                            if (line[i][-4:]==v[-4:]) and ((v,line[j]) in self.ns):
                            
                                if line[i-1] in [x[0] for x in self.ns]:
                                
                                
                                    return (line[i-1]+' '+v + ' '+ line[j])
                                
                                else:
                                
                                    return (v+' '+line[j])
                                
    def wt(self,text):
        t1=[]
        while self.cleaned(text):
    
            c = self.cleaned(text).split(' ')
            for i in range(len(c)):
                text=text.lower().replace(c[i],'')
            t1.append(' '.join(c))
        return (t1)
    
    def rotate_bound(self,image, angle):
    # grab the dimensions of the image and then determine the
    # center
    #image = cv2.imread(image)
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))
                                
                                
    def image2text(self,url,a,n,cont):
        #extracting text from images using tesseract, with variations in rotation,crop ratio and contrast to achieve highest recall
    
        img = self.url_to_image(url)
        
        #print (img)
    
        if img is not None:
     
            if img.shape[0]<img.shape[1]:
        
                img=self.rotate_bound(img,90)
                img = Image.fromarray(img)
                img2 = img.crop((0, a,img.size[0],(img.size[1])))
                img3 = img2.crop((0, 0,img2.size[0],(img2.size[1])/n))
        
                contrast = ImageEnhance.Contrast(img3)
                img4=contrast.enhance(cont)
                text = pytesseract.image_to_string(img4)
        #print (text)
                if not self.wt(text):
                
                    img = cv2.imread(path,0)
                    img=self.selrotate_bound(img,270)
                    img = Image.fromarray(img)
                    img2 = img.crop((0, a,img.size[0],(img.size[1])))
                    img3 = img2.crop((0, 0,img2.size[0],(img2.size[1])/n))
        
                    contrast = ImageEnhance.Contrast(img3)
                    img4=contrast.enhance(cont)
                    text = pytesseract.image_to_string(img4)
            #print (text)
                    if self.wt(text):
                        return (self.wt(text))
                    else:
                        return ([])
                else:
                    return (self.wt(text))
            else:
                img = Image.fromarray(img)
   
                img2 = img.crop((0, a,img.size[0],(img.size[1])))
                img3 = img2.crop((0, 0,img2.size[0],(img2.size[1])/n))
        
                contrast = ImageEnhance.Contrast(img3)
                img4=contrast.enhance(cont)
                text = pytesseract.image_to_string(img4)
        #print (text)
                return (self.wt(text))
        
        else:
            
            return([])
        
    
    def image2text2(self,url,cont):
    
        img = self.url_to_image(url)
    
        if img is not None: 
    
            if img.shape[0]<img.shape[1]:
        
                img=self.rotate_bound(img,90)
                img = Image.fromarray(img)
                width = img.size[0]
                height = img.size[1]
        
                img3 = img.crop((0,height/2 ,img.size[0],(img.size[1])))
        #img3 = img2.crop((0, 0,img2.size[0],(img2.size[1])/n))
        
                contrast = ImageEnhance.Contrast(img3)
                img4=contrast.enhance(cont)
                text = pytesseract.image_to_string(img4)
        #print (text)
                if not self.wt(text):
                
                    img = cv2.imread(path,0)
                    img=self.rotate_bound(img,270)
                    img = Image.fromarray(img)
                    img3 = img.crop((0, height/2,img.size[0],(img.size[1])))
            #img3 = img2.crop((0, 0,img2.size[0],(img2.size[1])/n))
                    contrast = ImageEnhance.Contrast(img3)
                    img4=contrast.enhance(cont)
                    text = pytesseract.image_to_string(img4)
            #print (text)
                    if self.wt(text):
                        return (self.wt(text))
                    else:
                        return ([])
                else:
                    return (self.wt(text))
            else:
                    img = Image.fromarray(img)
                    width = img.size[0]
                    height = img.size[1]

                    img3 = img.crop((0, height/2,img.size[0],(img.size[1])))
            #img3 = img2.crop((0, 0,img2.size[0],(img2.size[1])/n))

                    contrast = ImageEnhance.Contrast(img3)
                    img4=contrast.enhance(cont)
                    text = pytesseract.image_to_string(img4)
            #print (text)
                    return (self.wt(text))

        else:
            return([])
        
        


    def final(self):
        #checking combinations of contrast and crop ratio for better recall
        pass
     
        a=self.image2text (self.url,0,3.8,4)
    

        if not a:
            #print ('two')
            b=self.image2text(self.url,0,3.8,2)
            if not b:
                #print ('three')
                c= self.image2text(self.url,0,3.8,1)
                if not c:
                
                    #print ('four')
                    e=self.image2text(self.url,60,3.8,4)
                #e=image2text2(path,1)
                    if not e:
                        #print ('five')
                        return (self.image2text2(self.url,1))
                    else:
                        return (e)
                else:
                    return (c)
            else:
                return (b)
        else:
            return (a)
        
    def Final(self):
        
        b= self.final()
        s = ''
        if len(b)==1:
            s=s+str(b[0])
            return (s)
        else:
            for i in range(len(b)):
            
                s=s+str(b[i])+','
            
            return (s)
            
