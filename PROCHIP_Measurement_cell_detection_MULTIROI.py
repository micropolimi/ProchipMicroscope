from __future__ import division
from ScopeFoundry import Measurement
from ScopeFoundry.helper_funcs import sibling_path, load_qt_ui_file
from ScopeFoundry import h5_io
import pyqtgraph as pg
import numpy as np
import os
import time
from PROCHIP_Microscope.image_data import ImageManager


class PROCHIP_Measurement(Measurement):
    
    name = "PROCHIP"    #PROCHIP_Measurement
    
    def setup(self):
        
        "..." 

        self.ui_filename = sibling_path(__file__, "DualColor.ui")
        self.ui = load_qt_ui_file(self.ui_filename)
        
        # settings creation
        self.settings.New('refresh_period', dtype=float, unit='s', spinbox_decimals = 4, initial=0.08, vmin = 0 ,vmax=10)
        self.settings.New('auto_range_0', dtype=bool, initial=True)
        self.settings.New('auto_levels_0', dtype=bool, initial=True)
        self.settings.New('level_min_0', dtype=int, initial=60)
        self.settings.New('level_max_0', dtype=int, initial=150)
        self.settings.New('auto_range_1', dtype=bool, initial=True)
        self.settings.New('auto_levels_1', dtype=bool, initial=True)
        self.settings.New('level_min_1', dtype=int, initial=60)
        self.settings.New('level_max_1', dtype=int, initial=150)
        self.settings.New('save_h5', dtype=bool, initial=False)
        self.settings.New('save_roi_h5', dtype=bool, initial=False)
        self.settings.New('roi_half_side', dtype=int, initial=100)
        self.settings.New('min_cell_size', dtype=int, initial=1600)
        self.settings.New('selected_channel', dtype=int, initial=0, vmin = 0, vmax = 1)
        self.settings.New('captured_cells', dtype=int, initial=0)
        
        self.settings.New('acq_freq', dtype=float, unit='Hz', initial=200)
        self.settings.New('xsampling', dtype=float, unit='um', initial=0.11)
        self.settings.New('ysampling', dtype=float, unit='um', initial=0.11)
        self.settings.New('zsampling', dtype=float, unit='um', initial=1.0)
        
        
        self.camera = self.app.hardware['HamamatsuHardware']
        
        self.display_update_period = self.settings.refresh_period.val
        
        self.laser_0 = self.app.hardware['Laser_0']
        self.laser_1 = self.app.hardware['Laser_1']
        self.ni_co_0 = self.app.hardware['Counter_Output_0']
        self.ni_co_1 = self.app.hardware['Counter_Output_1']
        self.ni_do_0 = self.app.hardware['Digital_Output_0']
        
        self.channels = [0,1]
        
        
          
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
        

        self.settings.auto_levels_1.connect_to_widget(self.ui.autoLevels_checkBox1)
        self.settings.auto_range_1.connect_to_widget(self.ui.autoRange_checkBox1)
        self.settings.level_min_1.connect_to_widget(self.ui.min_doubleSpinBox1)
        self.settings.level_max_1.connect_to_widget(self.ui.max_doubleSpinBox1) 
        
        self.settings.selected_channel.connect_to_widget(self.ui.ch_doubleSpinBox)
        self.settings.captured_cells.connect_to_widget(self.ui.captured_doubleSpinBox) 
        
        
        
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
       
        plot1 = pg.PlotItem(title="channel1")
        self.imv1 = pg.ImageView(view = plot1)
        self.imv1.ui.histogram.hide()
        self.imv1.ui.roiBtn.hide()
        self.imv1.ui.menuBtn.hide()
        self.imv1.show()
        
        self.settings['captured_cells'] = 0
    
    
    
    def run(self):
        
        eff_subarrayh = self.eff_subarrayh
        eff_subarrayv = self.eff_subarrayv
        
        RANGE = 20    # TEMPORARY CONSTANT
        
        try:
            
            self.camera.read_from_hardware()
            freq1 = self.settings['acq_freq']
            
            self.start_triggered_Acquisition(freq1)
            
            first_cycle = True          # it will become False when the roi_h5_file will be created
                        
            z_index = []                # dynamic list of z indexes corresponding to the number of frames contained in each dataset for each channel
            old_z_index = []            # z_index at the previous cycle
            
            number_of_saved_roi = 0     # total number of saved cells
            num_rois = 0                # number of valid rois in the specific frame at that cycle
            active_rois = 0             # num_rois at the previous cycle
        
            self.roi_h5 = []            # dynamic list containing all the dataset under saving procedure
            
            centroid_x_position = []    # dynamic list containing the centroid positions in x of the cells under saving procedure
            centroid_y_position = []    # dynamic list containing the centroid positions in y of the cells under saving procedure
            
            number_of_channels = len(self.channels)
            
            self.image_h5 = [None]*number_of_channels    # prepare list to contain the image data to be saved in the h5file  
            
            
            
            while not self.interrupt_measurement_called:    # "run till abort" mode
            
                [frames, _dims] = self.camera.hamamatsu.getFrames()
                
                # processing of each acquired image
                for frame_index, frame in enumerate(frames):
                    
                    # define the correct channel in use for cells detection
                    channel_index = (self.camera.hamamatsu.buffer_index - self.camera.hamamatsu.backlog + frame_index + 1) % 2
                    
                    self.np_data = frame.getData()
                    self.im.image[channel_index] = np.reshape(self.np_data, (eff_subarrayv, eff_subarrayh))
                    
                    # detect all the cells present in this frame (do it only on the selected channel and not both)
                    if channel_index == self.settings.selected_channel.val:
                        self.im.find_cell(self.settings.selected_channel.val)
                     
                        
                     
                    if self.settings['save_roi_h5']:
                        
                        # H5 file creation                        
                        if first_cycle:
                            self.init_roi_h5()
                            first_cycle = False
                            
                        rois, selected_cx, selected_cy = self.im.roi_creation(channel_index)
                                                
                        num_rois = len(rois)                         
                        contained_rois = 0    # variable useful to allow the correct incrementation of the sample index in the dataset creation                       
                        new_cell = 0          # number of new cells recognized in the current frame
                                              
                        # creation of the needed elements in the lists
                        while len(z_index) < num_rois*number_of_channels:
                            z_index.append(0)
                        while len(centroid_x_position) < num_rois:
                            centroid_x_position.append(None)
                            centroid_y_position.append(None)
                                 
                            
                        for roi_index, roi in enumerate(rois):
                            
                            comp_index = 0         # position of the dataset whose centroid position matches with the centroid position of the current cell
                            empty_check = False    # True when the first roi frame of the current channel must be put into a dataset
                            comp_check = False     # True if there is a match between centroid positions
                            
                            # centroid comparison                           
                            for comp_index in range(active_rois):
                                if z_index[number_of_channels * comp_index + channel_index] == 0:
                                    empty_check = True
                                    break
                                if selected_cx[roi_index] in range(centroid_x_position[comp_index]-RANGE, centroid_x_position[comp_index]+RANGE) and selected_cy[roi_index] in range(centroid_y_position[comp_index]-RANGE, centroid_y_position[comp_index]+RANGE):    
                                    comp_check = True
                                    insert_position = number_of_channels * comp_index + channel_index
                                    break                                
                                
                            if comp_check == False:                                
                                if comp_index == (active_rois - 1) and empty_check == False:
                                    new_cell += 1
                                    contained_rois = new_cell + comp_index                                   
                                    self.roi_h5_dataset(number_of_saved_roi + contained_rois, roi_index)
                                
                                elif empty_check == True:
                                    pass
                                
                                else:                                   
                                    contained_rois = roi_index                                    
                                    self.roi_h5_dataset(number_of_saved_roi + contained_rois, roi_index)
                                    
                                number_of_dataset = len(self.roi_h5)   
                                    
                                # check again if there are all the needed elements in the lists, otherwise add them
                                while len(z_index) < number_of_dataset:
                                    z_index.append(0)                                    
                                while len(centroid_x_position) < (number_of_dataset/number_of_channels):
                                    centroid_x_position.append(None)   
                                    centroid_y_position.append(None)

                                if empty_check == True:
                                    insert_position = number_of_channels * comp_index + channel_index
                                else:
                                    insert_position = number_of_dataset - number_of_channels + channel_index
                                
                            # add an empty element where put the next roi frame    
                            if z_index[insert_position] != 0:
                                self.roi_h5[insert_position].resize(self.roi_h5[insert_position].shape[0]+1, axis = 0)
                            
                            self.roi_h5[insert_position][z_index[insert_position], :, :] = roi
                            self.h5_roi_file.flush()
                            z_index[insert_position] += 1
                            
                            centroid_x_position[int(insert_position/number_of_channels)] = selected_cx[roi_index]
                            centroid_y_position[int(insert_position/number_of_channels)] = selected_cy[roi_index]
                                
                        
                        # past dimensions comparison to see if we have new cells. If not, delate these elements
                        for position in range(active_rois):    # can be done only if num_rois < active_rois ??? IT SEEMS NOT !!!
 
                            check_pos = number_of_channels * (active_rois - position -1)
                                
                            if old_z_index[check_pos + channel_index] == z_index[check_pos + channel_index]:
                                del z_index[check_pos]
                                del z_index[check_pos]
                                del self.roi_h5[check_pos]
                                del self.roi_h5[check_pos]
                                del centroid_x_position[int(check_pos/number_of_channels)]
                                del centroid_y_position[int(check_pos/number_of_channels)]
                                   
                                number_of_saved_roi += 1
                                
                        
                        # update useful parameters for the next cycle
                        active_rois = num_rois
                        old_z_index = list(z_index)
                        
                        self.settings['captured_cells'] = number_of_saved_roi
                    
                    
                    
                if self.settings['save_h5']:
                    
                    progress_index = 0
                    
                    # temporarily stop the acquisition in order not to overwrite the camera buffer
                    self.pause_triggered_Acquisition()
                                                    
                    [frames, _dims] = self.camera.hamamatsu.getLastTotFrames()               
                    
                    # create and initialize h5file
                    self.initH5()
                    
                    z_index_h5 = [0]*number_of_channels    # list of indexes in which each element is the z index corresponding to a different channel
                    buffer_index = self.camera.hamamatsu.buffer_index + 1
                    
                    
                    for aframe in frames:                                

                        self.np_data = aframe.getData()
                        image_on_the_run = np.reshape(self.np_data, (eff_subarrayv, eff_subarrayh))
                                                
                        ch_on_the_run = buffer_index%2 # 0 if the image is even, 1 if the image is odd, in the image stack
                        
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
                    self.restart_triggered_Acquisition(freq1)
                
                    
                    
        finally:
            
            self.stop_triggered_Acquisition()

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
        self.imv1.close()
            
            
            
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
            if ch == self.settings.selected_channel.val:
                #cv2.rectangle(displayed_image,(0,0),(self.eff_subarrayh-1,self.eff_subarrayv-1),(255,255,0),3) 
                self.im.highlight_channel(displayed_image)
            
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
    

           
    def start_laser(self, laserHW):
        ''' Laser is prepared for digital modulation at the power specified. Laser is turned OFF before''' 
        
        if laserHW.laser_status.val == 'ON':
            self.stop_laser(laserHW)
        if laserHW.connected.val:    # do this only if the laser was connected by the user!
            if laserHW.operating_mode.val != 'DIGITAL':
                laserHW.operating_mode.val = 'DIGITAL'
                laserHW.operating_mode.write_to_hardware()
            
            laserHW.laser_status.val = 'ON'
            laserHW.laser_status.write_to_hardware()
            laserHW.read_from_hardware()
            
            
        
    def stop_laser(self, laserHW):
        ''' Laser is turned off '''
        
        if laserHW.connected.val:    # do this only if the laser was connected by the user!
            laserHW.laser_status.val = 'OFF' 
            laserHW.laser_status.write_to_hardware()
            
        
        
    def start_digital_rising_edge(self, digitalHW):
        '''The digital output of the DAQ start a rising edge procedure at the channel set by the user '''
        
        digitalHW.value.val=0
        digitalHW.write_value()
        
        digitalHW.value.val=1
        digitalHW.write_value()    # Trigger
        digitalHW.read_from_hardware()  
        
    
    
    def start_triggered_counter_task(self, counterHW, initial_delay, freq, duty_cycle):
        '''
        DAQ Counter set to be at 5V and 0V with a frequency of half the exposure time of the camera (1 frame OFF and 1 frame ON).
        It will be triggered by the digital output of the DAQ at the channel set by the user (please check that is equal to the one used for the ni_do_HW)
        '''
        
        if counterHW.connected.val:    # do this only if the counter was connected by the user!
            counterHW.freq.val=freq
            counterHW.freq.write_to_hardware()
            # if exposure time is less 1/fps then the laser should not be ON
            # for all the period but only when the camera is exposing pixels
            
            #===================================================================
            # if self.camera.exposure_time.val < 1/self.camera.internal_frame_rate.val:
            #     counterHW.duty_cycle.val=self.camera.exposure_time.val*counterHW.freq.val
            #     counterHW.duty_cycle.write_to_hardware()
            #===================================================================
            counterHW.duty_cycle.val=duty_cycle
            counterHW.duty_cycle.write_to_hardware()
            counterHW.trigger.val=True
            counterHW.trigger.write_to_hardware()
            counterHW.initial_delay.val=initial_delay    # probably in start trigger mode there is no delay so counter 1 should have an initial delay = 0
            #counterHW.initial_delay.val=0.00003896
            #counterHW.initial_delay.val=0.0000877 # initial delay to be synchronized with the camera( check camera documentation at start trigger mode...)
            counterHW.initial_delay.write_to_hardware()
            counterHW.start()
            counterHW.read_from_hardware()
            
            
            
    def stop_counter_task(self, counterHW):
        ''' Stop counter output task '''
        
        if counterHW.connected.val:    # do this only if the counter was connected by the user!
            counterHW.stop()
            
            
    
    def start_triggered_Acquisition(self,freq1):
        ''' The camera is operated with external start trigger mode and will be triggered by the digital output of the DAQ '''
        
        self.camera.settings.trigger_source.update_value(new_val = 'external')
        self.camera.settings.acquisition_mode.update_value(new_val = 'run_till_abort')
        self.camera.trmode.val='normal'
        self.camera.trmode.write_to_hardware()
        self.camera.tractive.val='syncreadout'
        self.camera.tractive.write_to_hardware()     
        self.camera.read_from_hardware()
        self.camera.hamamatsu.startAcquisition() 

        self.start_laser(self.laser_0)
        self.start_laser(self.laser_1)
        
        # dutycycle for advanced laser switch hardware
        
        t_readout = self.camera.hamamatsu.getPropertyValue("internal_line_interval")[0]
        camera_dutycycle = freq1*t_readout*self.camera.subarrayv.val
        
        # dutycycle for simple laser switch hardware
        
        camera_dutycycle=0.5 
        self.start_triggered_counter_task(self.ni_co_0, initial_delay=0.0000, freq=freq1, duty_cycle=camera_dutycycle)
        # counterOutput2 used to control 2 lasers via 1 signal and a duplication port with a buffer and a NOT. 
        self.start_triggered_counter_task(self.ni_co_1, initial_delay=0.00003896, freq=freq1/2, duty_cycle=0.5)
        
        self.start_digital_rising_edge(self.ni_do_0)
        
        
        
    def pause_triggered_Acquisition(self):
        
        self.camera.hamamatsu.stopAcquisitionNotReleasing()
        self.stop_counter_task(self.ni_co_0)
        self.stop_counter_task(self.ni_co_1)
    
    
    
    def restart_triggered_Acquisition(self, freq1):

        self.camera.hamamatsu.startAcquisitionWithoutAlloc()
        
        # dutycycle for advanced laser switch hardware
        
        t_readout = self.camera.hamamatsu.getPropertyValue("internal_line_interval")[0]
        camera_dutycycle = freq1*t_readout*self.camera.subarrayv.val
        
        # dutycycle for simple laser switch hardware
        
        camera_dutycycle=0.5 
        self.start_triggered_counter_task(self.ni_co_0, initial_delay=0.0000, freq=freq1, duty_cycle=camera_dutycycle)
        # counterOutput2 used to control 2 lasers via 1 signal and a duplication port with a buffer and a NOT. 
        self.start_triggered_counter_task(self.ni_co_1, initial_delay=0.00003896, freq=freq1/2, duty_cycle=0.5)
        
        self.start_digital_rising_edge(self.ni_do_0)
        
        
        
    def stop_triggered_Acquisition(self):
        ''' Acquisition is terminated, laser and counters turned off '''        
         
        self.camera.hamamatsu.stopAcquisition() 
        self.stop_laser(self.laser_0)
        self.stop_laser(self.laser_1)
        self.stop_counter_task(self.ni_co_0)
        self.stop_counter_task(self.ni_co_1)
        self.camera.settings['trigger_source'] = 'internal'
        self.camera.trsource.write_to_hardware()
        self.camera.read_from_hardware()
        
        
        
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
            length=self.camera.hamamatsu.number_image_buffers/number_of_channels #divided by two since we have the two channels
            
        # dataset creation
        
        
        for ch_index  in self.channels:
            
            name = f't0/c{ch_index}/image'
            
            self.image_h5[ch_index] = self.h5_group.create_dataset( name  = name, 
                                                      shape = ( length, img_size[0], img_size[1]),
                                                      dtype = self.im.image[0].dtype, chunks = (1, img_size[0], img_size[1])
                                                      )
               
            self.image_h5[ch_index].attrs['element_size_um'] =  [self.settings['zsampling'],self.settings['ysampling'],self.settings['xsampling']]
            self.image_h5[ch_index].attrs['acq_time'] =  timestamp

    
        
    def init_roi_h5(self):
        """
        Initialization operations for the h5 file
        """
        self.create_saving_directory()
        
        # file name creation
        timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
        sample = self.app.settings['sample']
        #sample_name = f'{timestamp}_{self.name}_{sample}_ROI.h5'
        sample_name = '_'.join([timestamp, self.name, sample, 'ROI.h5'])
        fname = os.path.join(self.app.settings['save_dir'], sample_name)
        
        # file creation
        self.h5_roi_file = h5_io.h5_base_file(app=self.app, measurement=self, fname = fname)
        self.h5_roi_group = h5_io.h5_create_measurement_group(measurement=self, h5group=self.h5_roi_file)
        
       
        
    def roi_h5_dataset(self, t_index, roi_index):   
        """
        Dataset creation
        """
                
        roi_size = self.settings.roi_half_side.val * 2
        number_of_channels = len(self.channels)
        number_of_dataset = len(self.roi_h5)
        
        # creation of one dataset for each channel
        for ch in range(number_of_channels):
        
            name = 't' + str(t_index) + '/c' + str(ch) + '/roi'
                
            fullname = self.h5_roi_group.name +'/'+ name
                
            #if len(self.roi_h5) > t_index + ch:
            for name_check_position in range (int(number_of_dataset/number_of_channels)): # can we do it only if len(self.roi_h5) > t_index + ch ??? MAYBE YES...TRY !!!
                if self.roi_h5[name_check_position * number_of_channels + ch].name == fullname:
                    return
                                        
            self.roi_h5.append(self.h5_roi_group.create_dataset( name  = name, 
                                                              shape = (1, roi_size, roi_size),
                                                              maxshape = ( None, roi_size, roi_size),
                                                              dtype = np.uint16, chunks = (1, roi_size, roi_size)
                                                              ) 
                               )
            
            last_position = number_of_dataset - 1
            
            # assign attributes
            self.roi_h5[last_position].dims[0].label = "z"
            self.roi_h5[last_position].dims[1].label = "y"
            self.roi_h5[last_position].dims[2].label = "x"
            self.roi_h5[last_position].attrs['element_size_um'] =  [self.settings['zsampling'],self.settings['ysampling'],self.settings['xsampling']]
            self.roi_h5[last_position].attrs['acq_time'] =  time.time()
            self.roi_h5[last_position].attrs['centroid_x'] =  self.im.cx[roi_index]    # TO BE CHANGED with selected_cx...
            self.roi_h5[last_position].attrs['centroid_y'] =  self.im.cy[roi_index]    # TO BE CHANGED with selected_cy...
    