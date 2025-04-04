import sys
sys.path.append("/vol/miltank/users/kaiserj/Individual_Privacy_Accounting")

from LSI.experiments.LSI_functionality import run_experiment
from LSI.utils.utils_parser import get_parser
from LSI.Datasets.dataset_helper import get_dataset, set_path_compressed, get_dataset_compressed
from LSI.utils.utils_data import compress_dataset, get_compressed_path
from LSI.models.models import get_model
from LSI.Datasets.dataset_compressed import CompressedDataset
import torch

# Set the device to GPU if available, otherwise fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments
parser = get_parser()
args = parser.parse_args()

# Check if the compressed dataset already exists
full_dataset_path, short_path = get_dataset_compressed(args.dataset, args.data_path, args.json_path)
if full_dataset_path is None:
    # If the compressed dataset does not exist, load the original dataset
    data_set_class, data_path = get_dataset(args.dataset, base_path=args.data_path, json_path=args.json_path)
    dataset = data_set_class(data_path, train=True)
    
    # Create a DataLoader for the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Get the model class for compression and initialize it
    compress_model_class = get_model(args.compress_model)
    # The number here is arbitrary, as the last layer of the compression model is removed
    compress_model = compress_model_class(10) 
    
    # Generate paths for the compressed dataset
    short_path, full_dataset_path, data_path, target_path = get_compressed_path(args.model, args.dataset, "train", args.data_path)
    
    # Compress the dataset using the compression model
    compress_dataset(compress_model, dataloader, full_dataset_path, data_path, target_path, DEVICE)
    
    # Save the path of the compressed dataset for future use
    set_path_compressed(args.dataset, short_path, args.json_path)
    
    # Load the compressed dataset
    dataset = CompressedDataset(full_dataset_path, train=True)
    print("dataset compressed")
    args.dataset = args.dataset + "_compressed"
else:
    # If the compressed dataset already exists, load it directly
    args.dataset = args.dataset + "_compressed"
    dataset = CompressedDataset(full_dataset_path, train=True)

# Run the experiment using the provided arguments and the dataset
run_experiment(args, dataset)


