# Collaboirative-SFL

We use a dictionary ***data_dict*** where each key-value pair is a client index and another dictionary ***data_dict[client_idx]***.<br><br>This secondary dictionary holds the following data for each client:
<br>
{***weak_model***: weak client model instance,<br>
***strong_model***: collaborative model instance (the offloaded/collab model that the strong client would hold)<br>
***server_model***: server model instance (what server model instance that would train with this client),<br>
***weak_optim***: optimizer for the weak model instance,<br>
***strong_optim***: optimizer for the collab/offloaded model instance,<br>
***server_optim***: optimizer for the server model instance,<br>
***data_iter***: dataloader iterator<br>
***dataloader***: dataloader object<br>
***datasize***: number of training data samples,<br>
***num_batches***: number of batches (ceil(len(dataloader) / batch_size))}

All models are initialized with the same weights

SIMADIKO:
**Kanoume assume oti oloi exoun ton idio arithmo batches!!!!!!!!!!!!**