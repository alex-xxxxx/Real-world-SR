import torch
import torch.utils.data

def create_dataloader(dataset, phase, batch_size, workers):

    #phase = dataset_opt['phase']
    if phase == 'train':
        num_workers = workers #6 * 4  # dataset_opt['n_workers'] * len(opt['gpu_ids'])
        batch_size =  batch_size #10 #8 #16 * 4  # dataset_opt['batch_size']
        shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=2, pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)
