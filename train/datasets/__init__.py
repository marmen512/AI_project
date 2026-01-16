# Datasets РґР»СЏ TRM
from train.datasets.trm_dataset import TRMDataset
from train.datasets.splits import split_by_doc_id, validate_split_integrity

__all__ = ['TRMDataset', 'split_by_doc_id', 'validate_split_integrity']
