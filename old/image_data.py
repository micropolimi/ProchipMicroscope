import numpy as np
import cv2



class ImageManager:
    '''
    Class to be used to store the acquired images split in two channels and methods useful for cell identification and roi creation
    '''

    def __init__(self,dim_h, dim_v, half_side, min_cell_size):

        zeros_im = np.zeros((dim_v,dim_h),dtype=np.uint16) 
        
        self.image = [zeros_im, zeros_im]    # original 16 bit images from the two channels
        self.dim_h = dim_h
        self.dim_v = dim_v
        
        self.contours = []        # list of contours of the detected cells
        self.cx = []             # list of the x coordinates of the centroids of the detected cells
        self.cy = []             # list of the y coordinates of the centroids of the detected cells
         
        self.roi_half_side = half_side        # half dimension of the roi
        self.min_cell_size = min_cell_size    # minimum area that the object must have to be recognized as a cell
    

        
    def find_cell(self, ch):    # ch: selected channel       
        """ Input: 
             ch: channel to use to create the 8 bit image to process
        Determines if a region avove thresold is a cell, generates contours of the cells and their centroids cx and cy      
        """          
    
    
        # level_min = np.amin(self.image[ch])
        # level_max = np.amax(self.image[ch])
        # img_thres = np.clip(self.image[ch], level_min, level_max)
        # image8bit = ((img_thres-level_min+1)/(level_max-level_min+1)*255).astype('uint8') 


        image8bit = (self.image[ch]/256).astype('uint8')
        
        _ret,thresh_pre = cv2.threshold(image8bit,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # ret is the threshold that was used, thresh is the thresholded image.     
        kernel  = np.ones((3,3),np.uint8)
        thresh = cv2.morphologyEx(thresh_pre,cv2.MORPH_OPEN, kernel, iterations = 2)
        # morphological opening (remove noise)
        cnts, _hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cx = []
        cy = []            
        contours = []
        roi_half_side = self.roi_half_side
        l = image8bit.shape
        
       
        for cnt in cnts:
            
            M = cv2.moments(cnt)
            if M['m00'] >  int(self.min_cell_size):    # (M['m00'] gives the contour area, also as cv2.contourArea(cnt)
                #extracts image center
            
                x0 = int(M['m10']/M['m00']) 
                y0 = int(M['m01']/M['m00'])
                x = int(x0 - roi_half_side) 
                y = int(y0 - roi_half_side)
                w = h = roi_half_side*2
                    
        
                if x>0 and y>0 and x+w<l[1]-1 and y+h<l[0]-1:    # only rois far from edges are considered
                    cx.append(x0)
                    cy.append(y0)
                    contours.append(cnt)
        
        self.cx = cx
        self.cy = cy 
        self.contours = contours  
        # found_cell_num = len(contours)
        # if found_cell_num > 0:
        #     print('found this number of cells:',len(contours))
        
            
    
    
    def draw_contours_on_image(self, image8bit):        
        """ Input: 
        img8bit: monochrome image, previously converted to 8bit
            Output:
        displayed_image: RGB image with annotations
        """  
        
        cx = self.cx
        cy = self.cy 
        roi_half_side = self.roi_half_side
        contours = self.contours
      
        displayed_image = cv2.cvtColor(image8bit,cv2.COLOR_GRAY2RGB)      
        
        for indx, _val in enumerate(cx):
            
    
            x = int(cx[indx] - roi_half_side) 
            y = int(cy[indx] - roi_half_side)
         
            w = h = roi_half_side*2
            
            displayed_image = cv2.drawContours(displayed_image, [contours[indx]], 0, (0,256,0), 2) 
            
            if indx == 0:
                color = (256,0,0)
            else: 
                color = (0,0,256)
                
            cv2.rectangle(displayed_image,(x,y),(x+w,y+h),color,1)
            
        return displayed_image
    
    
    
    def roi_creation(self, ch, cx, cy):
        """ Input: 
        ch: selected channel
        args: centroids cx and cy, if specified
            Output:
        rois: list of rois in the frame
        """          
        image16bit = self.image[ch]
    
        roi_half_side = self.roi_half_side
        rois = []
        
        for indx, _val in enumerate(cx):
            x = int(cx[indx] - roi_half_side) 
            y = int(cy[indx] - roi_half_side)
            w = h = roi_half_side*2
            detail = image16bit [y:y+w, x:x+h]
            rois.append(detail)
                    
        return rois
    
    
    
    
    def highlight_channel(self,displayed_image):
        
         cv2.rectangle(displayed_image,(0,0),(self.dim_h-1,self.dim_v-1),(255,255,0),3) 
   
            
            
        