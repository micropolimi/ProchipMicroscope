from __future__ import division
from ScopeFoundry import Measurement
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file
from ScopeFoundry import h5_io
import pyqtgraph as pg
import numpy as np
import os
import time
from PROCHIP_Microscope.image_data import ImageManager


class PROCHIP_Single_Color_Measurement(Measurement):
    
    name = "PROCHIP Single Color"    #PROCHIP_Measurement
    
    def setup(self):
        
        "..." 

        self.ui_filename = sibling_path(__file__, "SingleColor.ui")
        self.ui = load_qt_ui_file(self.ui_filename)
        
        # settings creation
        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals = 4, initial=0.08, vmin = 0 ,vmax=10)
        self.settings.New('auto_range_0', dtype=bool, initial=True)
        self.settings.New('auto_levels_0', dtype=bool, initial=True)
        self.settings.New('level_min_0', dtype=int, initial=60)
        self.settings.New('level_max_0', dtype=int, initial=150)
        #self.settings.New('auto_range_1', dtype=bool, initial=True)
        #self.settings.New('auto_levels_1', dtype=bool, initial=True)
        #self.settings.New('level_min_1', dtype=int, initial=60)
        #self.settings.New('level_max_1', dtype=int, initial=150)
        self.settings.New('save_h5', dtype=bool, initial=False)
        self.settings.New('save_roi_h5', dtype=bool, initial=False)
        self.settings.New('roi_half_side', dtype=int, initial=100)
        self.settings.New('min_cell_size', dtype=int, initial=1600)
        self.settings.New('selected_channel', dtype=int, initial=0, vmin = 0, vmax = 1)
        self.settings.New('captured_cells', dtype=int, initial=0)
        
        self.settings.New('xsampling', dtype=float, unit='um', initial=0.11)
        self.settings.New('ysampling', dtype=float, unit='um', initial=0.11)
        self.settings.New('zsampling', dtype=float, unit='um', initial=1.0)
        self.settings.New('flowrate', dtype=float, unit='nl/min', initial=50.0)
        
        self.camera = self.app.hardware['HamamatsuHardware']
        
        self.display_update_period = self.settings.refresh_period.val
        
        self.channels = [0]
        
          
    def setup_figure(self):
        """
        Runs once during App initialization, after setup()
        This is the place to make all graphical interface initializations,
        build plots, etc.
        """
                
        # connect ui widgets to measurement/hardware settings or functions
        self.ui.start_pushButton.clicked.connect(self.start)
        self.ui.interrupt_pushButton.clicked.connect(self.interrupt)
        self.settings.save_h5.connect_to_widget(self.ui.save_h5_checkBox)
        self.settings.save_roi_h5.connect_to_widget(self.ui.save_ROI_h5_checkBox)
        
        self.settings.auto_levels_0.connect_to_widget(self.ui.autoLevels_checkBox0)
        self.settings.auto_range_0.connect_to_widget(self.ui.autoRange_checkBox0)
        self.settings.level_min_0.connect_to_widget(self.ui.min_doubleSpinBox0) 
        self.settings.level_max_0.connect_to_widget(self.ui.max_doubleSpinBox0) 
        
        self.settings.captured_cells.connect_to_widget(self.ui.captured_doubleSpinBox) 
        
        self.settings.flowrate.connect_to_widget(self.ui.flowrate_doubleSpinBox)
        
    def pre_run(self):    
        '''
        Initialization of the ImageManager class and of figures
        '''
        
        self.display_update_period = self.settings.refresh_period.val
               
        eff_subarrayh = self.eff_subarrayh = int(self.camera.subarrayh.val/self.camera.binning.val)
        eff_subarrayv = self.eff_subarrayv = int(self.camera.subarrayv.val/self.camera.binning.val)
                
        self.im = ImageManager(eff_subarrayh,
                             eff_subarrayv, 
                             self.settings.roi_half_side.val,
                             self.settings.min_cell_size.val
                             )
        
        plot0 = pg.PlotItem(title="channel0")
        self.imv0 = pg.ImageView(view = plot0)
        self.imv0.ui.histogram.hide()
        self.imv0.ui.roiBtn.hide()
        self.imv0.ui.menuBtn.hide()
        self.imv0.show()
       
        
        self.settings['captured_cells'] = 0
    
    
    
    def run(self):
        
        eff_subarrayh = self.eff_subarrayh
        eff_subarrayv = self.eff_subarrayv
         
        try:
            
            self.camera.read_from_hardware()
            
            
            self.start_Acquisition()
            
            first_cycle = True          # it will become False when the roi_h5_file will be created
           
            z_index_roi = [0]*len(self.channels)    # list of z indexes in which each element is the z index corresponding to a different channel
            roi_index = 0      # absolute index of rois
            num_rois = 0       # number of rois detected in the current image
            num_active_rois = 0    # number of rois detected in the previous image
            self.roi_h5 = []      # content to be saved in the roi_h5_file
            self.image_h5 = [None]*len(self.channels)    # prepare list to contain the image data to be saved in the h5file
            active_rois =[]
            active_cx = []
            active_cy = []
            
            while not self.interrupt_measurement_called:    # "run till abort" mode
            
                [frames, _dims] = self.camera.hamamatsu.getFrames()

                # processing of each acquired image
                for frame_index, frame in enumerate(frames):
                    
                    # define the correct channel in use for cells detection
                    channel_index = 0 #(self.camera.hamamatsu.buffer_index - self.camera.hamamatsu.backlog + frame_index + 1) % 2
                    
                    self.np_data = frame.getData()
                    self.im.image[channel_index] = np.reshape(self.np_data, (eff_subarrayv, eff_subarrayh))
                    
                    #last_cx = self.im.cx
                    #last_cx = self.im.cy
                    
                    self.im.find_cell(self.settings.selected_channel.val)
                    
                    if self.settings['save_roi_h5']:
                        
                        # create and initialize the roi h5 file if it does not exist yet
                        if first_cycle:
                            self.init_roi_h5()
                            first_cycle = False
                        
                        num_rois = len(self.im.contours)
                        num_active_rois = len(active_rois)
                        
                        if num_rois == num_active_rois:
                            active_rois = self.im.roi_creation(channel_index, active_cx, active_cy)
                        
                        else:
                            active_rois = self.im.roi_creation(channel_index, self.im.cx, self.im.cy)
                            active_cx = self.im.cx
                            active_cy = self.im.cy
                            
                            
                        for i, roi in enumerate(active_rois):
                            
                            # create a new dataset if we are dealing with a new cell
                            self.roi_h5_dataset(roi_index, channel_index)
                            
                            # dynamically increment the dimension of the "box" in which we are going to put a new roi image
                            if z_index_roi[channel_index] != 0:   
                                self.roi_h5[channel_index].resize(self.roi_h5[channel_index].shape[0]+1, axis = 0)
                            
                            self.roi_h5[channel_index][z_index_roi[channel_index], :, :] = roi
                            self.h5_roi_file.flush()    # this allow us to open the h5 file also while it is not completely created yet
                            z_index_roi[channel_index] += 1
                        
                        if num_rois < num_active_rois:
                           roi_index += 1
                           z_index_roi = [0]*len(self.channels) 
                        # one roi is disapperead: update indexes, ready for a new cell
                        # this is thought for a single cell, multi rois are not always managed
                                        
                        self.settings['captured_cells'] = roi_index
                    
                    
                    
                if self.settings['save_h5']:
                    
                    progress_index = 0
                    
                    # temporarily stop the acquisition in order not to overwrite the camera buffer
                    self.pause_Acquisition()
                                                    
                    [frames, _dims] = self.camera.hamamatsu.getLastTotFrames()               
                    
                    # create and initialize h5file
                    self.initH5()
                    
                    z_index_h5 = [0]*len(self.channels)    # list of indexes in which each element is the z index corresponding to a different channel
                    buffer_index = self.camera.hamamatsu.buffer_index + 1
                    
                    
                    for aframe in frames:                                

                        self.np_data = aframe.getData()
                        image_on_the_run = np.reshape(self.np_data, (eff_subarrayv, eff_subarrayh))
                                                
                        ch_on_the_run = 0 # 0 if the image is even, 1 if the image is odd, in the image stack
                        
                        self.image_h5[ch_on_the_run][z_index_h5[ch_on_the_run], :, :] = image_on_the_run  # saving to the h5 dataset
                        z_index_h5[ch_on_the_run] += 1
                           
                        self.h5file.flush()
                        self.settings['progress'] = progress_index*100./self.camera.hamamatsu.number_image_buffers
                        progress_index += 1
                        buffer_index += 1

                    self.h5file.close()    # finally save and close the h5 file created
                    self.settings['save_h5'] = False    # update the value to False, so the save_h5 can be called again
                    self.settings['progress'] = 0
                    
                    print("\n \n ******* \n \n Stack saved :D !\n \n *******")
  
                    # restart the acquisition
                    self.restart_Acquisition()
                
        finally:
            
            self.stop_Acquisition()

            # close all the h5 file still open    
            if self.settings['save_h5']:
                self.h5file.close()  
                self.settings['save_h5'] = False
                
            if self.settings['save_roi_h5']:
                self.h5_roi_file.close()
                self.settings['save_roi_h5'] = False
                
             
    def post_run(self):
        '''
        Close all the figures after the run ended
        '''
        self.imv0.close()
            
            
            
    def update_display(self):
        """
        Displays the numpy array called displayed_image for each channel.
        This function runs repeatedly and automatically during the measurement run.
        Its update frequency is defined by self.display_update_period.
        """
        
        for ch in self.channels:
            
            # choose the setting keys according to the channel to update
            autorange_key = f'auto_range_{ch}'
            autolevel_key = f'auto_levels_{ch}'
            level_min_key = f'level_min_{ch}'
            level_max_key = f'level_max_{ch}'
            
            if self.settings[autolevel_key]:
                # if autolevel is ON, normalize the image to its max and min     
                level_min = np.amin(self.im.image[ch])
                level_max = np.amax(self.im.image[ch])
                self.settings[level_min_key] = level_min    
                self.settings[level_max_key] = level_max
            
            else:
                # if autolevel is OFF, normalize the image to the choosen values     
                level_min = self.settings[level_min_key]
                level_max = self.settings[level_max_key]
            
            # note that these levels are uint16, but the visulaized image is uint8, for compatibility with opencv processing (contours and rectangles annotations) 
            
            # thresolding is required if autolevel is OFF; it could be avoided if autolevel is ON
            img_thres = np.clip(self.im.image[ch], level_min, level_max)
            
            # conversion to 8bit is done here for compatibility with opencv    
            image8bit_normalized = ((img_thres-level_min+1)/(level_max-level_min+1)*255).astype('uint8') 
            
            # creation of the image with open cv annotations, ready to be displayed
            displayed_image = self.im.draw_contours_on_image(image8bit_normalized)
            
            
            # display the image with a frame around the figure corresponding to the channel selected to do the find_cell operation (selected_channel)
            #if ch == self.settings.selected_channel.val:
            #    self.im.highlight_channel(displayed_image)
            
            imv_key = f'imv{ch}'
            imv = getattr(self, imv_key)
            imv.setImage(displayed_image, autoLevels=False, autoRange=self.settings[autorange_key], levels=(0,255))                
               


    def updateIndex(self, last_frame_index):
        """
        Update the index of the image to fetch from buffer. 
        If we reach the end of the buffer, we reset the index.
        """
        last_frame_index += 1
        
        if last_frame_index > self.camera.hamamatsu.number_image_buffers - 1:    # if we reach the end of the buffer
            last_frame_index = 0    # reset
        
        return last_frame_index
          
    
    def start_Acquisition(self):
        ''' The camera is operated with external start trigger mode and will be triggered by the digital output of the DAQ '''
        
        self.camera.settings.trigger_source.update_value(new_val = 'internal')
        self.camera.settings.acquisition_mode.update_value(new_val = 'run_till_abort')
        self.camera.trmode.val='normal'
        self.camera.trmode.write_to_hardware()
        # self.camera.tractive.val='syncreadout'
        # self.camera.tractive.write_to_hardware()     
        self.camera.read_from_hardware()
        self.camera.hamamatsu.startAcquisition() 
        #t_readout = self.camera.hamamatsu.getPropertyValue("internal_line_interval")[0]
        
        
        
    def pause_Acquisition(self):
        self.camera.hamamatsu.stopAcquisitionNotReleasing()
    
    
    
    def restart_Acquisition(self):

        self.camera.hamamatsu.startAcquisitionWithoutAlloc()
        
        
    def stop_Acquisition(self):       
         
        self.camera.hamamatsu.stopAcquisition() 
        
        
    def create_saving_directory(self):
        
        if not os.path.isdir(self.app.settings['save_dir']):
            os.makedirs(self.app.settings['save_dir'])
        
        
        
    def initH5(self):
        """
        Initialization operations for the h5 file
        """
        self.create_saving_directory()
        
        # file name creation
        timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
        sample = self.app.settings['sample']
        #sample_name = f'{timestamp}_{self.name}_{sample}.h5'
        if sample == '':
            sample_name = '_'.join([timestamp, self.name])
        else:
            sample_name = '_'.join([timestamp, self.name, sample])
        fname = os.path.join(self.app.settings['save_dir'], sample_name + '.h5')
        
        # file creation
        self.h5file = h5_io.h5_base_file(app=self.app, measurement=self, fname = fname)
        self.h5_group = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5file)
        
        img_size=self.im.image[0].shape    # both image[0] and image[1] are valid, since they have the same shape
        
        number_of_channels = len(self.channels)
        
        # take as third dimension of the file the total number of images collected in the buffer
        if self.camera.hamamatsu.last_frame_number < self.camera.hamamatsu.number_image_buffers:
            length = int((self.camera.hamamatsu.last_frame_number+1)/number_of_channels)
        else:
            length=self.camera.hamamatsu.number_image_buffers/number_of_channels 
            
        # dataset creation
        
        
        for ch_index  in self.channels:
            
            name = f't0/c{ch_index}/image'
            
            self.image_h5[ch_index] = self.h5_group.create_dataset( name  = name, 
                                                      shape = ( length, img_size[0], img_size[1]),
                                                      dtype = self.im.image[0].dtype, chunks = (1, img_size[0], img_size[1])
                                                      )
               
            self.image_h5[ch_index].attrs['element_size_um'] =  [self.settings['zsampling'],self.settings['ysampling'],self.settings['xsampling']]
            self.image_h5[ch_index].attrs['acq_time'] =  timestamp
            self.image_h5[ch_index].attrs['flowrate'] = self.settings['flowrate']

    
        
    def init_roi_h5(self):
        """
        Initialization operations for the h5 file
        """
        self.create_saving_directory()
        
        # file name creation
        timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
        sample = self.app.settings['sample']
        #sample_name = f'{timestamp}_{self.name}_{sample}_ROI.h5'
        if sample == '':
            sample_name = '_'.join([timestamp, self.name, 'ROI.h5'])
        else:
            sample_name = '_'.join([timestamp, self.name, sample, 'ROI.h5'])
        fname = os.path.join(self.app.settings['save_dir'], sample_name)
        
        # file creation
        self.h5_roi_file = h5_io.h5_base_file(app=self.app, measurement=self, fname = fname)
        self.h5_roi_group = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5_roi_file)
        
       
        
    def roi_h5_dataset(self, t_index, c_index):   
        """
        Dataset creation
        """
                
        roi_size = self.settings.roi_half_side.val*2    # roi dimension
        
        timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
        
        
        if len(self.roi_h5) == 0:    # creation of the first two datasets (one for each channel)
            
            for ch in self.channels:
                             
                name = f't{0:04d}/c{ch}/roi' # data set name, initially with t index 0000
                
                self.roi_h5.append(self.h5_roi_group.create_dataset( name  = name, 
                                                              shape = (1, roi_size, roi_size),
                                                              maxshape = ( None, roi_size, roi_size),
                                                              dtype = np.uint16, chunks = (1, roi_size, roi_size)
                                                              ) 
                                   )
                
                # dataset attributes
                self.roi_h5[ch].attrs['element_size_um'] =  [self.settings['zsampling'],self.settings['ysampling'],self.settings['xsampling']]
                self.roi_h5[ch].attrs['flowrate'] = self.settings['flowrate']
                self.roi_h5[ch].attrs['acq_time'] =  timestamp
                # self.roi_h5[c_index].attrs['centroid_x'] =  self.im.cx[0] # to be updated when multiple rois are saved
                # self.roi_h5[c_index].attrs['centroid_y'] =  self.im.cy[0]
                self.roi_h5[ch].attrs['centroid_x'] =  self.im.cx[0] # to be updated when multiple rois are saved
                self.roi_h5[ch].attrs['centroid_y'] =  self.im.cy[0]
                
        else:
             
            name = f't{t_index:04d}/c{c_index}/roi' # dataset name  
            
            fullname = self.h5_roi_group.name + '/' + name
            
            if self.roi_h5[c_index].name != fullname:    # create a new dataset only if the name does not exist yet
                        
                self.roi_h5[c_index] = self.h5_roi_group.create_dataset( name  = name, 
                                                                  shape = (1, roi_size, roi_size),
                                                                  maxshape = ( None, roi_size, roi_size),
                                                                  dtype = np.uint16, chunks = (1, roi_size, roi_size)
                                                                  ) 
                
                # dataset attributes
                self.roi_h5[c_index].attrs['element_size_um'] =  [self.settings['zsampling'],self.settings['ysampling'],self.settings['xsampling']]
                self.roi_h5[c_index].attrs['acq_time'] =  timestamp
                self.roi_h5[c_index].attrs['flowrate'] = self.settings['flowrate']
                self.roi_h5[c_index].attrs['centroid_x'] =  self.im.cx[0]    # to be updated when multiple rois are saved
                self.roi_h5[c_index].attrs['centroid_y'] =  self.im.cy[0]