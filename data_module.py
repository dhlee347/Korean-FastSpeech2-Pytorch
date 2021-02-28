import pytorch_lightning as pl




class KssData(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()

        
