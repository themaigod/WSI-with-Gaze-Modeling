from wsi.data.image_producer import GridImageDataset

dataset_tumor_train = GridImageDataset("/home/omnisky/ajmq/PATCHES_TUMOR_TRAIN",
                                       "/home/omnisky/ajmq/NCRF/jsons/train_epoch",
                                       768,
                                       256,
                                       crop_size=224)
