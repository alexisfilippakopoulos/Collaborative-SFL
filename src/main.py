from models import ClientModel, CollaborativeModel, ServerModel
import torch
import numpy as np
from yaml import safe_load
from tqdm import tqdm
from torch.utils.data import DataLoader
import threading
from torch import nn
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


SEED = 32

def create_dict(num_clients):
    data_dict = {}
    for i in range(num_clients):
        data_dict[i] = {"weak_model": None, "strong_model": None, "server_model": None,
                        "weak_optim": None, "strong_optim": None, "server_optim": None,
                        "dataloader": None, "data_iter": None, "datasize": None, "num_batches": None}
        
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
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True), int(len(subset)), int(np.ceil(len(subset) / batch_size))

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
        data_dict[cid]["dataloader"], data_dict[cid]["datasize"], data_dict[cid]["num_batches"] = load_data(subset_path=f"/home/alex/Desktop/ASOEE/split_yannis/Collaboirative-SFL/subset_data/sub_{cid}.pth", batch_size=32, shuffle=True)
    return data_dict



def read_config(file_path):
    with open(file_path, 'r') as file:
        cfg = safe_load(file)
    return cfg

def forward_pass_clients(weak_model, strong_model, iterator, criterion, device, server_inputs, losses, idx, list_of_targets):
    inputs, targets = next(iterator)
    list_of_targets[idx] = targets
    inputs, targets = inputs.to(device), targets.to(device)
    weak_model, strong_model = weak_model.to(device), strong_model.to(device)
    split_outputs = weak_model(inputs)
    client_outputs = strong_model(split_outputs)
    server_inputs[idx] = split_outputs.clone().detach()
    loss = criterion(client_outputs, targets)
    losses[idx] = loss
    return

def forward_pass_server(server_model: nn.Module, device: torch.device, inputs: torch.tensor, targets: torch.tensor, criterion: nn, losses: list, idx: int):
    server_model.to(device)
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = server_model(inputs)
    loss = criterion(outputs, targets)
    losses[idx] = loss
    return

def federated_averaging(models: list[dict], datasizes: list[int], device: torch.device):
    avg_weights = {}
    total_data = sum(datasizes)
    for i, model in enumerate(models):
        for layer, weights in model.items():
            weights = weights.to(device)
            if layer not in avg_weights.keys():
                avg_weights[layer] = weights * (datasizes[i] / total_data)
            else:
                avg_weights[layer] += weights * (datasizes[i] / total_data)
    return avg_weights


def client_aggregation(data_dict: dict[dict], device):
    strong_model_weights = [None] * len(data_dict.keys())
    weak_model_weights = [None] *  len(data_dict.keys())
    datasizes = [None] *  len(data_dict.keys())

    for client, metadata in data_dict.items():
        strong_model_weights[client] = deepcopy(metadata["strong_model"].state_dict())
        weak_model_weights[client] = deepcopy(metadata["weak_model"].state_dict())
        datasizes[client] = metadata["datasize"]

    aggr_weak_model = federated_averaging(weak_model_weights, datasizes, device)
    aggr_strong_model = federated_averaging(strong_model_weights, datasizes, device)

    # load aggregated weights on each model instance
    for client, metadata in data_dict.items():
        metadata["strong_model"].load_state_dict(aggr_strong_model)
        metadata["weak_model"].load_state_dict(aggr_weak_model)

    print("\t[+] Aggregated Client-Side Models")
    return data_dict    

def server_aggregation(data_dict: dict[dict], device):
    datasizes = [None] *  len(data_dict.keys())
    server_model_weights = [None] * len(data_dict.keys())

    for client, metadata in data_dict.items():
        server_model_weights[client] = deepcopy(metadata["server_model"].state_dict())
        datasizes[client] = metadata["datasize"]

    aggr_server_model = federated_averaging(server_model_weights, datasizes, device)

    # load aggregated weights on each model instance
    for client, metadata in data_dict.items():
        metadata["server_model"].load_state_dict(aggr_server_model)

    print("\t[+] Aggregated Server-Side Models")
    return data_dict

def train_one_epoch(data_dict: dict[dict], criterion: nn, device: torch.device):
    curr_clients_mean_loss = 0.
    curr_server_mean_loss = 0.
    for _ in range(data_dict[0]["num_batches"]):
            # List holding the client-side's outputs
            server_inputs = [None] * len(data_dict.keys())
            client_targets = [None] * len(data_dict.keys())
            losses = [None] * len(data_dict.keys())
            threads = []
            # for each batch
            for client, metadata in data_dict.items():
                # clear the gradients in all optimizers
                metadata['weak_optim'].zero_grad(), metadata['strong_optim'].zero_grad(), metadata['server_optim'].zero_grad()
                # client-side forward propagation in parallel
                thread = threading.Thread(target=forward_pass_clients, args=(metadata["weak_model"], metadata["strong_model"], metadata["data_iter"], criterion, device, server_inputs, losses, client, client_targets))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
            
            # compute and BP the strong and weak models based on all clients' mean loss for this batch
            mean_client_loss = torch.stack(losses).mean()
            mean_client_loss.backward()
            curr_clients_mean_loss += mean_client_loss.item()

            for client, metadata in data_dict.items():
                metadata['weak_optim'].step(), metadata['strong_optim'].step()

            losses = [None] * len(data_dict.keys())

            # server-side forward propagation in parallel
            threads = []
            for idx, (inputs, targets) in enumerate(zip(server_inputs, client_targets)):
                thread = threading.Thread(target=forward_pass_server, args=(data_dict[idx]["server_model"], device, inputs, targets, criterion, losses, idx))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # compute and BP the server models based on all models' mean loss for this batch
            mean_server_loss = torch.stack(losses).mean()
            mean_server_loss.backward()
            curr_server_mean_loss += mean_server_loss.item()

            for client, metadata in data_dict.items():
                metadata['server_optim'].step()

    avg_client_mean_loss = round(curr_clients_mean_loss / data_dict[0]["num_batches"], 4)
    avg_server_mean_loss = round(curr_server_mean_loss / data_dict[0]["num_batches"], 4)

    return data_dict, avg_client_mean_loss, avg_server_mean_loss


def train_both_parties(data_dict, epochs, criterion, device, fed_avg_freq):
    for e in tqdm(range(epochs)):
        print(f"[+] Epoch: {e + 1}")
        # create a dataloader iterator for current epoch
        for _, metadata in data_dict.items():
            metadata["data_iter"] = iter(metadata["dataloader"])
        # train for one epoch
        data_dict, avg_client_mean_loss, avg_server_mean_loss = train_one_epoch(data_dict=data_dict, criterion=criterion, device=device)

        print(f"\t[+] Average Client-Side Mean Loss: {avg_client_mean_loss}")
        print(f"\t[+] Average Server-Side Mean Loss: {avg_server_mean_loss}")

        # if it's time to aggregate do
        if (e + 1) % fed_avg_freq == 0:
            data_dict = client_aggregation(data_dict=data_dict, device=device)
            data_dict = server_aggregation(data_dict=data_dict, device=device)

    return data_dict



def main():
    cfg = read_config("/home/alex/Desktop/ASOEE/split_yannis/Collaboirative-SFL/src/config.yaml")
    num_clients = cfg["weak_clients_num"] + cfg["strong_clients_num"]
    data_dict = create_dict(num_clients=num_clients)
    device = torch.device(cfg["device"])
    set_all_seeds(cfg["seed"])
    data_dict = populate_dict(data_dict=data_dict, seed=cfg["seed"], lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    data_dict = train_both_parties(data_dict=data_dict, epochs=cfg["epochs"], criterion=criterion, device=device, fed_avg_freq=cfg["fed_avg_freq"])

if __name__ == "__main__":
    main()