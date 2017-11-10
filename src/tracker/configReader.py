'''
'''

import json

default_config = \
{
	"multiCam": False,
	"captureDevice0": 0,
	"captureDevice1": 1,
	"imageWidth": 640,
	"imageHeight": 480,
	"fps": 60,
	"quitAppByGPIO": False,
	"minArea": 25,
	"maxArea": 5000,
	"thresHold": 33,
	"minMotion": 5,
	"maxMotion": 100,
	"maxTraces" : 20,
	"maxTrackingAngle" : 35.0,
	"streamServer": True
}

class configuration:
    def __init__(self, file_name=None):
        self.conf = default_config
        self.configFileName = None
        if file_name is not None:
            self.read(file_name)


    def read(self, config_file):
        try:
            self.conf = json.load(open(config_file))
            self.configFileName = config_file
        except:
            pass

    def write(self, config_file=None):
        if config_file is None:
            fn = self.configFileName
        else:
            fn = config_file

        try:
            json.dump(self.conf, open(fn,'w'), indent=1, sort_keys=True)
        except:
            raise

if __name__ == '__main__':
    cc = configuration('does-not-exist.json')
    print cc.conf
    cc.conf['fps'] = 77
    cc.conf['AAAAA'] = 'text'
    cc.write('test.json')
    cc.read('test.json')
    print cc.conf
