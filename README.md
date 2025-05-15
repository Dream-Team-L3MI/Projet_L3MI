# Projet_L3MI
Lien Google Drive du projet : 
	https://drive.google.com/drive/folders/1gWhNId5zFIcSLjKakIikl-2JZ9rxEhIo
	q=sharedwith:public%20parent:1gWhNId5zFIcSLjKakIikl-2JZ9rxEhIo



Dans les scripts python : 
	from dotenv import load_dotenv
	import os

	load_dotenv("config.env")

	ngspice_cmd = os.getenv("NGSPICE_CMD")
	data_dir = os.getenv("DATA_DIR")
	epochs = int(os.getenv("DEFAULT_EPOCHS"))
	lr = float(os.getenv("LEARNING_RATE")) 