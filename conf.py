##Configuration
conf = {
	"dataset_used": "cifar10", #type of dataset, mnist, cifar10, cifar100, fmnist
	
	# 1: BinaryCrossEntropy
	# 2: CategoricalCrossEntropy,
	# 3: FedLCalibratedLoss (calibrated loss function)

	'retrain_loss_criterion': 3,
 
	# Run for Client 
	"train_load_data":False, 
	"train_contrastive_learning":False, # Implement Contrastive loss
	"train_loss_criterion": 3,  #3: FedLCalibratedLoss

	## Data Load and implement Loss 
	"eval_load_data":False, 
	"eval_contrastive_learning":False,
	"eval_loss_criterion": 3,  #3: FedLCalibratedLoss

	# Run for Server
	"test_load_data":False, # Load Test dataset in pairs # to run `visualize.py` put it False
	"test_contrastive_learning":False, # Implement Contrastive Loss
	"test_loss_criterion": 3, #3: FedLCalibratedLoss

	#Type of data，tabular, image
	"data_type" : "image",
		
	#Model selection: mlp, cnn 
	"model_name" : "cnn",
	
	#prediction method
	"classification_type": "multi", #binary or multi

	# if using binary class True, else for multi class False
	# "loss_criterion_binary": True, # True, False
	
	# it is a hyperparameter for contrastive learning
	# '--tau':6.45,
	
	#Classes
	"num_classes": 10

	#number of parties
	"num_parties":30, 


	#Data processing method
	"no-iid": "fl-fcr",

	# client_optimizer used
    "client_optimizer": "SGD"

	#re_train_optimizer used
	"re_train_optimizer": "SGD"

	#Global epoch
	"global_epochs" : 50, 

	#Local epoch
	"local_epochs" : 10,

	#dirichlet distribution
	"beta" : 0.05, 
	"batch_size" : 1024,
	"weight_decay":1e-6,

    #learning rate
	"lr" : 0.01,
	"momentum" : 0.9,

    #Model aggregation
	"is_init_avg": True,

    #Local val test ratio
	"split_ratio": 0.2,
 
	# Synthetic Data Generation
    "gamma": 0.8,                 # Controls the distribution of weights (adjustable)
    "iid": False,                  # Set True for IID data, False for non-IID data
    "dimension": 3072,            # Feature dimension (e.g., 32x32x3 = 3072 for CIFAR-like data)
    # You can define additional parameters if needed for different settings
    "synthetic_samples_mean": 5,  # Mean for lognormal sample generation
    "synthetic_samples_std": 3,   # Standard deviation for lognormal sample generation

    #Label name
	"label_column": "label",

	#Data name
	"data_column": "file",

    # Test dataset , 
	"test_dataset": "./data/dataset/test/test.csv",

    #Train dataset
	"train_dataset" : "./data/dataset/train/train.csv", 

    #Where to save the model:
	"model_dir":"./save_model/",

    #Model name:
	"model_file":"model.pth",
	#Retrained Model name:
	"retrain_model_file":"retrained_model.pth",

	#save training epoch info
	"save_epochs_info" :{
		# make dir to save info in csv
		"dir_name" : "./save_info/",
		# for training_server_&_client epochs .csv file 
		"train_info_file" :"train_info.csv",
		# for re_training epochs .csv file 
		"re_train_info_file" :"re_training_info.csv",
		# for re_training epochs .csv file 
		"only_global_epochs_file" :"only_global_epochs.csv",
	},

	"retrain":{
		"epoch": 100,
		"lr": 0.01,
		"weight_decay": 1e-6,
		"num_vr":2000
	},

	
}

