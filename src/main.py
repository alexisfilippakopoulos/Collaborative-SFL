from models import ClientModel, CollaborativeModel, ServerModel
import torch
import numpy as np
from yaml import safe_load
from tqdm import tqdm

SEED = 32

def set_all_seeds(seed) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_models_and_optims(num_clients, seed, lr):
    weak_client_models, weak_optims  = [], []
    collaborative_models, collaborative_optims = [], []
    server_models, server_optims = [], []
    print(f"[+] Instanciating models and optimizers")
    for _ in tqdm(range(num_clients)):
        torch.manual_seed(seed)
        weak_model = ClientModel()
        weak_optim = torch.optim.SGD(params=weak_model.parameters(), lr=lr)
        torch.manual_seed(seed)
        collab_model = CollaborativeModel()
        collab_optim = torch.optim.SGD(params=collab_model.parameters(), lr=lr)
        torch.manual_seed(seed)
        server_model = ServerModel()
        server_optim = torch.optim.SGD(params=server_model.parameters(), lr=lr)

        weak_client_models.append(weak_model), weak_optims.append(weak_optim)
        collaborative_models.append(collab_model), collaborative_optims.append(collab_optim) 
        server_models.append(server_model), server_optims.append(server_optim) 

    return weak_client_models, weak_optims, collaborative_models, collaborative_optims, server_models, server_optim

def read_config(file_path):
    with open(file_path, 'r') as file:
        cfg = safe_load(file)
    return cfg

def main():
    cfg = read_config("/home/alex/Desktop/ASOEE/split_yannis/Collaboirative-SFL/src/config.yaml")
    print(cfg)
    set_all_seeds(cfg["seed"])
    weak_client_models, weak_optims, collaborative_models, collaborative_optims, server_models, server_optims = create_models_and_optims(num_clients=cfg["weak_clients_num"] + cfg["strong_clients_num"], seed=cfg["seed"], lr=cfg["lr"])

if __name__ == "__main__":
    main()