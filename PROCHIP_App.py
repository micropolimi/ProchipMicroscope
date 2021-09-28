from ScopeFoundry import BaseMicroscopeApp


class PROCHIP_App(BaseMicroscopeApp):
    
    name = 'PROCHIP_App'
    
    def setup(self):
        
        from Hamamatsu_ScopeFoundry.CameraHardware import HamamatsuHardware
        from OBIS_ScopeFoundry.laser_hardware import LaserHW
        from NIdaqmx_ScopeFoundry.ni_do_hardware import NI_DO_hw
        from NIdaqmx_ScopeFoundry.ni_co_hardware import NI_CO_hw
        from NIdaqmx_ScopeFoundry.ni_ao_hardware import NI_AO_hw

        self.add_hardware(HamamatsuHardware(self))
        self.add_hardware(LaserHW(self, name='Laser_0'))
        self.add_hardware(LaserHW(self, name='Laser_1'))
        self.add_hardware(NI_CO_hw(self, name='Counter_Output_0'))
        self.add_hardware(NI_CO_hw(self, name='Counter_Output_1'))
        self.add_hardware(NI_DO_hw(self, name='Digital_Output_0'))
        self.add_hardware(NI_AO_hw(self, name='Analog_Output_0'))
        self.add_hardware(NI_AO_hw(self, name='Analog_Output_1'))
       
        print("Adding Hardware Components")
        
        from Hamamatsu_ScopeFoundry.CameraMeasurement import HamamatsuMeasurement
        self.add_measurement(HamamatsuMeasurement(self))
       
        from PROCHIP_Microscope.PROCHIP_Measurement_multichannel import PROCHIP_Multichannel_Measurement
        self.add_measurement(PROCHIP_Multichannel_Measurement(self))
        
        from PROCHIP_Microscope.PROCHIP_Measurement_SIM import PROCHIP_SIM_Measurement
        self.add_measurement(PROCHIP_SIM_Measurement(self))
        
        from PROCHIP_Microscope.PROCHIP_Measurement_hexSIM import PROCHIP_HexSIM_Measurement
        self.add_measurement(PROCHIP_HexSIM_Measurement(self))
        
        
        from HexSimAnalyser.HexSimAnalyser_measurement import HexSimAnalysis
        self.add_measurement(HexSimAnalysis(self))
        
        from PROCHIP_Microscope.PROCHIP_Measurement_cell_detection import PROCHIP_Measurement
        self.add_measurement(PROCHIP_Measurement(self))
        
        from PROCHIP_Microscope.PROCHIP_Measurement_cell_detection_SINGLE_COLOR import PROCHIP_Single_Color_Measurement
        self.add_measurement(PROCHIP_Single_Color_Measurement(self))
       
        print("Adding measurement components")
        
        self.ui.show()
        self.ui.activateWindow()
        
if __name__ == '__main__':
            
    import sys
    app = PROCHIP_App(sys.argv)
    
    ################### for debugging only ##############
    app.settings_load_ini(".\\Settings\\settingsPROCHIP.ini")
    #app.settings_load_ini(".\\Settings\\settingsSIM.ini")
    
    for hc_name, hc in app.hardware.items():

       hc.settings['connected'] = True    # connect all the hardwares   
    #####################################################    
    
    sys.exit(app.exec_())
        
    