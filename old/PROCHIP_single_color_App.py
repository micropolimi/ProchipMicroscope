from ScopeFoundry import BaseMicroscopeApp



class PROCHIP_App(BaseMicroscopeApp):
    
    name = 'PROCHIP_single_color_App'
    
    def setup(self):
        
        from Hamamatsu_ScopeFoundry.CameraHardware import HamamatsuHardware
        from OBIS_ScopeFoundry.laser_hardware import LaserHW
        
        self.add_hardware(HamamatsuHardware(self))
        self.add_hardware(LaserHW(self, name='Laser_0'))
        self.add_hardware(LaserHW(self, name='Laser_1'))
        print("Adding Hardware Components")
        
        from Hamamatsu_ScopeFoundry.CameraMeasurement import HamamatsuMeasurement
        self.add_measurement(HamamatsuMeasurement(self))
       
        from PROCHIP_Microscope.PROCHIP_Measurement_cell_detection_SINGLE_COLOR import PROCHIP_Measurement

        self.add_measurement(PROCHIP_Measurement(self))
        
        print("Adding measurement components")
        
        self.ui.show()
        self.ui.activateWindow()
        
if __name__ == '__main__':
            
    import sys
    app = PROCHIP_App(sys.argv)
    
    ################### for debugging only ##############
    app.settings_load_ini(".\\Settings\\settingsPROCHIP_single_color.ini")
    for hc_name, hc in app.hardware.items():

       hc.settings['connected'] = True    # connect all the hardwares   
    #####################################################    
    
    sys.exit(app.exec_())
        
    