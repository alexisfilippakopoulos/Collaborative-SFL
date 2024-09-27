from models import ClientModel, CollaborativeModel, ServerModel
import torch
import numpy as np
from yaml import safe_load
from tqdm import tqdm
from torch.utils.data import DataLoader

SEED = 32

def create_dict(num_clients):
    data_dict = {}
    for i in range(num_clients):
        data_dict[i] = {"weak_model": None, "strong_model": None, "server_model": None,
                        "weak_optim": None, "strong_optim": None, "server_optim": None,
                        "dataloader": None, "datasize": None}
        
    return data_dict

def set_all_seeds(seed) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(subset_path, batch_size, shuffle):
        subset = torch.load(f=subset_path)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True), len(subset)

def populate_dict(data_dict, seed, lr):
    print(f"[+] Instanciating models and optimizers")
    for cid in tqdm(data_dict.keys()):
        torch.manual_seed(seed)
        data_dict[cid]["weak_model"] = ClientModel()
        data_dict[cid]["weak_optim"] = torch.optim.SGD(params=data_dict[cid]["weak_model"].parameters(), lr=lr)
        torch.manual_seed(seed)
        data_dict[cid]["strong_model"] = CollaborativeModel()
        data_dict[cid]["strong_optim"] = torch.optim.SGD(params=data_dict[cid]["strong_model"].parameters(), lr=lr)
        torch.manual_seed(seed)
        data_dict[cid]["server_model"] = ServerModel()
        data_dict[cid]["server_optim"] = torch.optim.SGD(params=data_dict[cid]["server_model"].parameters(), lr=lr)
        data_dict[cid]["dataloader"], data_dict[cid]["datasize"] = load_data(subset_path=f"/home/alex/Desktop/ASOEE/split_yannis/Collaboirative-SFL/subset_data/sub_{cid}.pth", batch_size=32, shuffle=True)
    return data_dict



def read_config(file_path):
    with open(file_path, 'r') as file:
        cfg = safe_load(file)
    return cfg

def main():
    cfg = read_config("/home/alex/Desktop/ASOEE/split_yannis/Collaboirative-SFL/src/config.yaml")
    num_clients = cfg["weak_clients_num"] + cfg["strong_clients_num"]
    data_dict = create_dict(num_clients=num_clients)
    print(data_dict)
    set_all_seeds(cfg["seed"])
    data_dict = populate_dict(data_dict=data_dict, seed=cfg["seed"], lr=cfg["lr"])
    print(data_dict)

if __name__ == "__main__":
    main()