import numpy as np
import cv2

class image_data:

    def __init__(self,dim_h, dim_v, half_side, min_cell_size):

        zeros_im = np.zeros((dim_v,dim_h),dtype=np.uint16) 
        
        self.image = [zeros_im, zeros_im]    # original 16 bit images from the two channels
        
        self.contour = []    # list of contours of the detected cells
        self.cx = []         # list of the coordinates of the centroids of the detected cells
        self.cy = []
        
        self.roi_half_side = half_side        # half dimension of the roi
        self.min_cell_size = min_cell_size    # minimum area that the object must have to be recognized as a cell
    

        
    def find_cell(self, ch):    # ch: selected channel       
            
            image8bit = (self.image[ch]/256).astype('uint8')
            
            _ret,thresh_pre = cv2.threshold(image8bit,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # ret is the threshold that was used, thresh is the thresholded image.     
            kernel  = np.ones((3,3),np.uint8)
            thresh = cv2.morphologyEx(thresh_pre,cv2.MORPH_OPEN, kernel, iterations = 2)
            # morphological opening (remove noise)
            contours, _hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cx = []
            cy = []            
            contour =[]
           
            for cnt in contours:
                
                M = cv2.moments(cnt)
                if M['m00'] >  int(self.min_cell_size):    # (M['m00'] gives the contour area, also as cv2.contourArea(cnt)
                    #extracts image center
                    cx.append(int(M['m10']/M['m00']))
                    cy.append(int(M['m01']/M['m00']))
                    contour.append(cnt)
            
            self.cx = cx
            self.cy = cy 
            self.contour = contour  
            
            
    
    
    def draw_contours_on_image(self, image8bit):        
        """ Input: 
        img8bit: monochrome image, previously converted to 8bit
            Output:
        displayed_image: RGB image with annotations
        """  
        
        cx = self.cx
        cy = self.cy 
        roi_half_side = self.roi_half_side
        contour = self.contour
      
        displayed_image = cv2.cvtColor(image8bit,cv2.COLOR_GRAY2RGB)      
        
        for indx, _val in enumerate(cx):
            
    
            x = int(cx[indx] - roi_half_side) 
            y = int(cy[indx] - roi_half_side)
         
            
            w = h = roi_half_side*2
            
            displayed_image = cv2.drawContours(displayed_image, [contour[indx]], 0, (0,256,0), 2) 
            
            if indx == 0:
                color = (256,0,0)
            else: 
                color = (0,0,256)
                
            cv2.rectangle(displayed_image,(x,y),(x+w,y+h),color,1)    
        
            
        return displayed_image
    
    
    
    def roi_creation(self, ch):    # ch: selected channel
        
        image16bit = self.image[ch]
            
        cx = self.cx
        cy = self.cy 
        roi_half_side = self.roi_half_side
        
        l = image16bit.shape
        rois = []    # list of rois in the frame considered
        
        for indx, _val in enumerate(cx):
            x = int(cx[indx] - roi_half_side) 
            y = int(cy[indx] - roi_half_side)
            w = h = roi_half_side*2
            
            if x>0 and y>0 and x+w<l[1]-1 and y+h<l[0]-1:    # only rois far from edges are considered
                    detail = image16bit [y:y+w, x:x+h]
                    rois.append(detail)
                    
        return rois
   
            
            
        