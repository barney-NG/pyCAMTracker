from ConfigParser import ConfigParser
from os.path import expanduser

class Config(ConfigParser):
	def __init__(self, filename='.tracker.conf'):
		ConfigParser.__init__(self)
		self.configFile = "%s/%s" % ( expanduser("~"), filename )
		self.hasChanged = False

	def setVal(self, section, option, value):
		# TODO: add section/otion if it does not exist
		strval = "%s" % (value)
		self.hasChanged = True
		return self.set(section, option, strval)

	def readConfig(self):
		self.read(self.configFile)

	def writeConfig(self, fp=None):
		if self.hasChanged:
			fp = open(self.configFile, 'w')
			self.write(fp)
		return
