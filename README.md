# Projet_L3MI

Dans les scripts python : 
	from dotenv import load_dotenv
	import os

	load_dotenv("config.env")

	ngspice_cmd = os.getenv("NGSPICE_CMD")
	data_dir = os.getenv("DATA_DIR")
	epochs = int(os.getenv("DEFAULT_EPOCHS"))
	lr = float(os.getenv("LEARNING_RATE")) 

