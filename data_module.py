import numpy as np
import pytorch_lightning as pl


class KssData(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()

    def collate(self, dataset):
        
        def collate_fn(batch, indicies=None):
            len_arr = np.array([d["text"].shape[0] for d in batch])
            index_arr = np.argsort(-len_arr)
            batchsize = len(batch)
            real_batchsize = int(math.sqrt(batchsize))

            cut_list = list()
            for i in range(real_batchsize):
                if self.sort:
                    cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])
                else:
                    cut_list.append(np.arange(i*real_batchsize, (i+1)*real_batchsize))
            
            output = list()
            for i in range(real_batchsize):
                output.append(self.reprocess(batch, cut_list[i]))        
        
        dataset = dataset.map(
            get_examples_from_episode,
            batched=True,
            with_indices=True,
            num_proc=cfg.num_workers,
            remove_columns=dataset['train'].column_names,
        )
        


        return output        
