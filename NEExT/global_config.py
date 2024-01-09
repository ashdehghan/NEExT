"""

"""

# External libraries
import json

# # Internal libraries
# from ugaf.singleton_template import Singleton

# @Singleton
class Global_Config:

	def __init__(self):
		self.config = None
		self.quiet_mode = False


	def set_output_mode(self, quiet_mode):
		if quiet_mode == "on":
			self.quiet_mode = True
		else:
			self.quiet_mode = False


	def load_config(self, config, config_type):
		"""
			This method will simply load the global configuration
			file.
		"""
		if config_type == "file":
			with open(config, "r") as config_file:
				self.config = dict(json.load(config_file))
		elif config_type == "object":
			self.config = config
		else:
			raise ValueError("Selected config type is not supported.")
		# Set verbose setting
		if self.config["quiet_mode"] == "yes":
			self.quiet_mode = True
		else:
			self.quiet_mode = False


