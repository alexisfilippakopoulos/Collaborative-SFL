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
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

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
        data_dict[cid]["dataloader"], data_dict[cid]["datasize"], data_dict[cid]["num_batches"] = load_data(subset_path=f"subset_data/sub_{cid}.pth", batch_size=32, shuffle=True)
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

    print("\t[+] Aggregated Client-Side Models")
    return aggr_strong_model, aggr_weak_model    

def server_aggregation(data_dict: dict[dict], device):
    # lists to save number of data samples and model weights for each client
    datasizes = [None] *  len(data_dict.keys())
    server_model_weights = [None] * len(data_dict.keys())

    for client, metadata in data_dict.items():
        server_model_weights[client] = deepcopy(metadata["server_model"].state_dict())
        datasizes[client] = metadata["datasize"]

    # aggregate
    aggr_server_model = federated_averaging(server_model_weights, datasizes, device)

    print("\t[+] Aggregated Server-Side Models")
    return aggr_server_model

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
                metadata["weak_model"].train(), metadata["strong_model"].train()
                # clear the gradients in all optimizers
                metadata['weak_optim'].zero_grad(), metadata['strong_optim'].zero_grad(), metadata['server_optim'].zero_grad()
                # client-side forward propagation in parallel
                thread = threading.Thread(target=forward_pass_clients, args=(metadata["weak_model"], metadata["strong_model"], metadata["data_iter"], criterion, device, server_inputs, losses, client, client_targets))
                threads.append(thread)
                thread.start()

            # wait all forward propagations
            for thread in threads:
                thread.join()
            
            # compute and BP the strong and weak models based on all clients' mean loss for this batch
            mean_client_loss = torch.stack(losses).mean()
            mean_client_loss.backward()
            curr_clients_mean_loss += mean_client_loss.item()
            # update client-side models
            for client, metadata in data_dict.items():
                metadata['weak_optim'].step(), metadata['strong_optim'].step()
                # offload models to cpu
                metadata["weak_model"], metadata["strong_model"] = metadata["weak_model"].to(torch.device("cpu")), metadata["strong_model"].to(torch.device("cpu"))
            # server-side forward propagation in parallel
            losses = [None] * len(data_dict.keys())
            threads = []
            for idx, (inputs, targets) in enumerate(zip(server_inputs, client_targets)):
                data_dict[idx]["server_model"].train()
                thread = threading.Thread(target=forward_pass_server, args=(data_dict[idx]["server_model"], device, inputs, targets, criterion, losses, idx))
                threads.append(thread)
                thread.start()

            # wait all forward propagations
            for thread in threads:
                thread.join()

            # compute and BP the server models based on all models' mean loss for this batch
            mean_server_loss = torch.stack(losses).mean()
            mean_server_loss.backward()
            curr_server_mean_loss += mean_server_loss.item()

            # update server-side models
            for client, metadata in data_dict.items():
                metadata['server_optim'].step()
                # offload models to cpu
                metadata['server_model'] = metadata['server_model'].to(torch.device("cpu"))

    avg_client_mean_loss = round(curr_clients_mean_loss / data_dict[0]["num_batches"], 4)
    avg_server_mean_loss = round(curr_server_mean_loss / data_dict[0]["num_batches"], 4)

    return data_dict, avg_client_mean_loss, avg_server_mean_loss


def train_both_parties(data_dict, epochs, criterion, device, fed_avg_freq):
    stats_df = pd.DataFrame(columns=['epoch', "avg_client_mean_loss", "avg_server_mean_loss", 'client_precision', 'server_precision', 'client_recall', "server_recall", "client_f1_score", "server_f1_score"])
    for e in tqdm(range(epochs)):
        print(f"[+] Epoch: {e + 1}")
        # create a dataloader iterator for current epoch
        for _, metadata in data_dict.items():
            metadata["data_iter"] = iter(metadata["dataloader"])
        # train for one epoch
        data_dict, avg_client_mean_loss, avg_server_mean_loss = train_one_epoch(data_dict=data_dict, criterion=criterion, device=device)
        
        print(f"\t[+] Average Client-Side Mean Loss: {avg_client_mean_loss}")
        print(f"\t[+] Average Server-Side Mean Loss: {avg_server_mean_loss}")
        server_accuracy, server_precision, server_recall, server_f1_score = None, None, None, None
        client_accuracy, client_precision, client_recall, client_f1_score = None, None, None, None
        # if it's time to aggregate do
        if (e + 1) % fed_avg_freq == 0:
            aggr_strong_model, aggr_weak_model = client_aggregation(data_dict=data_dict, device=device)
            aggr_server_model = server_aggregation(data_dict=data_dict, device=device)

            # load aggregated weights on each model instance
            for client, metadata in data_dict.items():
                metadata["strong_model"].load_state_dict(aggr_strong_model)
                metadata["weak_model"].load_state_dict(aggr_weak_model)
                metadata["server_model"].load_state_dict(aggr_server_model)

            server_accuracy, server_precision, server_recall, server_f1_score, client_accuracy, client_precision, client_recall, client_f1_score = evaluate_aggr_models(weak_model=data_dict[0]["weak_model"], strong_model=data_dict[0]["strong_model"], server_model=data_dict[0]["server_model"], test_dl_path="subset_data/sub_test.pth", device=device)

        stats_df.loc[len(stats_df)] = {'epoch': e + 1, 'avg_client_mean_loss': avg_client_mean_loss, 'avg_server_mean_loss': avg_server_mean_loss, 'client_acc': client_accuracy, 'server_acc': server_accuracy, 'client_precision': client_precision, 'server_precision': server_precision, 'client_recall': client_recall, 'server_recall': server_recall, 'client_f1_score': client_f1_score, 'server_f1_score': server_f1_score}
        stats_df.to_csv("results.csv")
    return data_dict

def evaluate_aggr_models(weak_model: nn, strong_model: nn, server_model: nn, test_dl_path: str, device: torch.device):
    dataloader = DataLoader(dataset=torch.load(test_dl_path), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    weak_model.to(device), strong_model.to(device), server_model.to(device)
    weak_model.eval(), strong_model.eval(), server_model.eval()
    Y_true, Y_client, Y_server = [], [], []
    client_correct, server_correct, total = 0, 0, 0
    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # inference
            weak_outputs = weak_model(inputs)
            client_outputs = strong_model(weak_outputs)
            server_outputs = server_model(weak_outputs)
            # get both side preds
            _, client_preds = torch.max(client_outputs, dim=1)
            _, server_preds = torch.max(server_outputs, dim=1)
            # calculate how many are correct
            client_correct += (client_preds == targets).sum().item()
            server_correct += (server_preds == targets).sum().item()
            total += targets.size(0)
            # save preds for each data sample in the dataloader
            Y_true.extend(targets.detach().cpu().numpy())
            Y_client.extend(client_preds.detach().cpu().numpy())
            Y_server.extend(server_preds.detach().cpu().numpy())

    server_accuracy = round((server_correct / total) * 100, 2)
    client_accuracy = round((client_correct / total) * 100, 2)
    server_precision, server_recall, server_f1_score, _ = precision_recall_fscore_support(Y_true, Y_server, average='weighted', zero_division=0.0)
    client_precision, client_recall, client_f1_score, _ = precision_recall_fscore_support(Y_true, Y_client, average='weighted', zero_division=0.0)

    print(f"\t[+] Server-Side Evaluation:\n\t\tAccuracy: {server_accuracy}\n\t\tPrecision: {server_precision: .4f}\n\t\tRecall: {server_recall: .4f}\n\t\tF1-Score: {server_f1_score: .4f}")
    print(f"\t[+] Client-Side Evaluation:\n\t\tAccuracy: {client_accuracy}\n\t\tPrecision: {client_precision: .4f}\n\t\tRecall: {client_recall: .4f}\n\t\tF1-Score: {client_f1_score: .4f}")

    return server_accuracy, server_precision, server_recall, server_f1_score, client_accuracy, client_precision, client_recall, client_f1_score
            



def main():
    cfg = read_config("src/config.yaml")
    num_clients = cfg["weak_clients_num"] + cfg["strong_clients_num"]
    data_dict = create_dict(num_clients=num_clients)
    device = torch.device(cfg["device"])
    set_all_seeds(cfg["seed"])
    data_dict = populate_dict(data_dict=data_dict, seed=cfg["seed"], lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    data_dict = train_both_parties(data_dict=data_dict, epochs=cfg["epochs"], criterion=criterion, device=device, fed_avg_freq=cfg["fed_avg_freq"])

if __name__ == "__main__":
    main()