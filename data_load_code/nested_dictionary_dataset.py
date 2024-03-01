# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch
from torch.utils.data.dataloader import default_collate

from .unicore_dataset import UnicoreDataset


def _flatten(dico, prefix=None):
    """Flatten a nested dictionary."""
    new_dico = OrderedDict()
    if isinstance(dico, dict):
        prefix = prefix + "." if prefix is not None else ""
        for k, v in dico.items():
            if v is None:
                continue
            new_dico.update(_flatten(v, prefix + k))
    elif isinstance(dico, list):
        for i, v in enumerate(dico):
            new_dico.update(_flatten(v, prefix + ".[" + str(i) + "]"))
    else:
        new_dico = OrderedDict({prefix: dico})
    return new_dico


def _unflatten(dico):
    """Unflatten a flattened dictionary into a nested dictionary."""
    new_dico = OrderedDict()
    for full_k, v in dico.items():
        full_k = full_k.split(".")
        node = new_dico
        for k in full_k[:-1]:
            if k.startswith("[") and k.endswith("]"):
                k = int(k[1:-1])
            if k not in node:
                node[k] = OrderedDict()
            node = node[k]
        node[full_k[-1]] = v
    return new_dico


class NestedDictionaryDataset(UnicoreDataset):
    def __init__(self, defn):
        super().__init__()
        self.defn = _flatten(defn)
        first = None
        for v in self.defn.values():
            if not isinstance(
                v,
                (
                    UnicoreDataset,
                    torch.utils.data.Dataset,
                ),
            ):
                raise ValueError("Expected Dataset but found: {}".format(v.__class__))
            first = first or v
            if len(v) > 0:
                assert len(v) == len(first), "dataset lengths must match"

        self._len = len(first)
        print("self._len: ",self._len)

    def __getitem__(self, index):
        return OrderedDict((k, ds[index]) for k, ds in self.defn.items())

    def __len__(self):
        return self._len

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        print("samples_al: ",type(samples))
        if len(samples) == 0:
            return {}
        sample = OrderedDict()
        for k, ds in self.defn.items():
            try:
                sample[k] = ds.collater([s[k] for s in samples])

            except NotImplementedError:
                sample[k] = default_collate([s[k] for s in samples])
        # print("aa1_sample: ",sample)
        # print("type(_unflatten(sample)): ",type(_unflatten(sample)))
        # print("_unflatten(sample).keys(): ", _unflatten(sample).keys())  # odict_keys(['net_input', 'target', 'smi_name'])
        # print("_unflatten(sample)['net_input'].keys(): ", _unflatten(sample)['net_input'].keys())  # odict_keys(['src_tokens', 'src_coord', 'src_distance', 'src_edge_type'])
        # print("_unflatten(sample)['smi_name'][0]: ", _unflatten(sample)['smi_name'][0])
        # print("_unflatten(sample)['target']['finetune_target'][0]: ", _unflatten(sample)['target']['finetune_target'][0])
        # print("_unflatten(sample)['net_input']['src_tokens'][0]: ", _unflatten(sample)['net_input']['src_tokens'][0].size())   # 长度有变化
        # print("_unflatten(sample)['net_input']['src_coord'][0]: ", _unflatten(sample)['net_input']['src_coord'][0])
        # print("_unflatten(sample)['net_input']['src_distance'][0]: ", _unflatten(sample)['net_input']['src_distance'][0])
        # print("_unflatten(sample)['net_input']['src_edge_type'][0]: ",_unflatten(sample)['net_input']['src_edge_type'][0])
        
        

        return _unflatten(sample)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return any(ds.supports_prefetch for ds in self.defn.values())

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        for ds in self.defn.values():
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch(indices)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return all(ds.can_reuse_epoch_itr_across_epochs for ds in self.defn.values())

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.defn.values():
            ds.set_epoch(epoch)