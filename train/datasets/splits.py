"""
Р‘РµР·РїРµС‡РЅРµ СЂРѕР·РґС–Р»РµРЅРЅСЏ dataset РЅР° train/eval РїРѕ doc_id РґР»СЏ Р·Р°РїРѕР±С–РіР°РЅРЅСЏ eval leakage.
"""

from typing import Dict, List, Set, Tuple
from torch.utils.data import Dataset, Subset
import hashlib


def split_by_doc_id(
    dataset: Dataset,
    train_ratio: float = 0.9,
    doc_id_key: str = 'doc_id'
) -> Tuple[Dataset, Dataset]:
    """
    Р РѕР·РґС–Р»РёС‚Рё РґР°С‚Р°СЃРµС‚ РЅР° train/eval РїРѕ doc_id (РЅРµ РІРёРїР°РґРєРѕРІРѕ).
    
    Р“Р°СЂР°РЅС‚СѓС” С‰Рѕ Р¶РѕРґРµРЅ РґРѕРєСѓРјРµРЅС‚ РЅРµ Р·'СЏРІР»СЏС”С‚СЊСЃСЏ РІ РѕР±РѕС… splits.
    
    Args:
        dataset: Р”Р°С‚Р°СЃРµС‚ С‰Рѕ РїРѕРІРµСЂС‚Р°С” (input_ids, labels, doc_id, segment_id) Р°Р±Рѕ dict
        train_ratio: Р§Р°СЃС‚РєР° РґРѕРєСѓРјРµРЅС‚С–РІ РґР»СЏ train (0.0-1.0)
        doc_id_key: РљР»СЋС‡ РґР»СЏ doc_id РІ dataset item
        
    Returns:
        (train_dataset, eval_dataset) СЏРє Subset datasets
        
    Raises:
        ValueError: РЇРєС‰Рѕ dataset РЅРµ РїС–РґС‚СЂРёРјСѓС” doc_id Р°Р±Рѕ train_ratio РЅРµРєРѕСЂРµРєС‚РЅРёР№
    """
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio РјР°С” Р±СѓС‚Рё РјС–Р¶ 0 С‚Р° 1, РѕС‚СЂРёРјР°РЅРѕ {train_ratio}")
    
    # Р—С–Р±СЂР°С‚Рё РІСЃС– doc_id
    doc_to_indices: Dict[str, List[int]] = {}
    
    for idx in range(len(dataset)):
        try:
            item = dataset[idx]
            
            # РџС–РґС‚СЂРёРјРєР° СЂС–Р·РЅРёС… С„РѕСЂРјР°С‚С–РІ РїРѕРІРµСЂРЅРµРЅРЅСЏ
            if isinstance(item, tuple):
                # РњРѕР¶Р»РёРІРѕ (input_ids, labels, doc_id, segment_id) Р°Р±Рѕ (input_ids, labels)
                if len(item) >= 3:
                    doc_id = item[2]
                else:
                    # РЇРєС‰Рѕ doc_id РЅРµРјР°С”, СЃС‚РІРѕСЂРёС‚Рё Р· С–РЅРґРµРєСЃСѓ (РЅРµС–РґРµР°Р»СЊРЅРѕ, Р°Р»Рµ РїСЂР°С†СЋС”)
                    doc_id = f"default_{idx}"
            elif isinstance(item, dict):
                doc_id = item.get(doc_id_key, item.get('doc_id', None))
                if doc_id is None:
                    doc_id = f"default_{idx}"
            else:
                # РќРµРѕС‡С–РєСѓРІР°РЅРёР№ С„РѕСЂРјР°С‚, СЃС‚РІРѕСЂРёС‚Рё doc_id
                doc_id = f"default_{idx}"
            
            # РќРѕСЂРјР°Р»С–Р·СѓРІР°С‚Рё doc_id РґРѕ string
            if isinstance(doc_id, (int, float)):
                doc_id = str(doc_id)
            elif not isinstance(doc_id, str):
                doc_id = str(hash(doc_id))
            
            if doc_id not in doc_to_indices:
                doc_to_indices[doc_id] = []
            doc_to_indices[doc_id].append(idx)
        except Exception as e:
            # РЇРєС‰Рѕ РїРѕРјРёР»РєР° РїСЂРё РѕС‚СЂРёРјР°РЅРЅС– item, РїСЂРѕРїСѓСЃС‚РёС‚Рё
            print(f"Warning: РџРѕРјРёР»РєР° РїСЂРё РѕР±СЂРѕР±С†С– С–РЅРґРµРєСЃСѓ {idx}: {e}")
            continue
    
    if not doc_to_indices:
        raise ValueError("РќРµ РІРґР°Р»РѕСЃСЏ РІРёС‚СЏРіРЅСѓС‚Рё doc_id Р· dataset")
    
    # Р РѕР·РґС–Р»РёС‚Рё РґРѕРєСѓРјРµРЅС‚Рё (РЅРµ С–РЅРґРµРєСЃРё!)
    doc_ids = list(doc_to_indices.keys())
    num_train_docs = int(len(doc_ids) * train_ratio)
    
    train_doc_ids = set(doc_ids[:num_train_docs])
    eval_doc_ids = set(doc_ids[num_train_docs:])
    
    # РџРµСЂРµРєРѕРЅР°С‚РёСЃСЏ С‰Рѕ РЅРµРјР°С” overlap
    assert train_doc_ids.isdisjoint(eval_doc_ids), "Overlap РјС–Р¶ train С‚Р° eval doc_ids!"
    
    # Р—С–Р±СЂР°С‚Рё С–РЅРґРµРєСЃРё РґР»СЏ РєРѕР¶РЅРѕРіРѕ split
    train_indices = []
    eval_indices = []
    
    for doc_id, indices in doc_to_indices.items():
        if doc_id in train_doc_ids:
            train_indices.extend(indices)
        elif doc_id in eval_doc_ids:
            eval_indices.extend(indices)
        else:
            # Р¦Рµ РЅРµ РїРѕРІРёРЅРЅРѕ СЃС‚Р°С‚РёСЃСЏ
            raise RuntimeError(f"doc_id {doc_id} РЅРµ РІ train Р°Р±Рѕ eval")
    
    # РЎС‚РІРѕСЂРёС‚Рё Subset datasets
    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)
    
    print(f"вњ… Р РѕР·РґС–Р»РµРЅРѕ dataset: {len(train_indices)} train samples ({len(train_doc_ids)} docs), {len(eval_indices)} eval samples ({len(eval_doc_ids)} docs)")
    
    return train_dataset, eval_dataset


def validate_split_integrity(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    doc_id_key: str = 'doc_id'
) -> bool:
    """
    РџРµСЂРµРІС–СЂРёС‚Рё С‰Рѕ РЅРµРјР°С” overlap doc_id РјС–Р¶ train С‚Р° eval.
    
    Args:
        train_dataset: Train dataset
        eval_dataset: Eval dataset
        doc_id_key: РљР»СЋС‡ РґР»СЏ doc_id
        
    Returns:
        True СЏРєС‰Рѕ РІСЃРµ РћРљ, False СЏРєС‰Рѕ Р·РЅР°Р№РґРµРЅРѕ overlap
        
    Raises:
        ValueError: РЇРєС‰Рѕ РЅРµ РІРґР°Р»РѕСЃСЏ РІРёС‚СЏРіРЅСѓС‚Рё doc_id
    """
    train_doc_ids = set()
    eval_doc_ids = set()
    
    # Р—С–Р±СЂР°С‚Рё doc_id Р· train
    for idx in range(len(train_dataset)):
        try:
            if isinstance(train_dataset, Subset):
                item = train_dataset.dataset[train_dataset.indices[idx]]
            else:
                item = train_dataset[idx]
            
            if isinstance(item, tuple) and len(item) >= 3:
                doc_id = str(item[2])
            elif isinstance(item, dict):
                doc_id = str(item.get(doc_id_key, item.get('doc_id', None)))
                if doc_id == 'None':
                    continue
            else:
                continue
            
            train_doc_ids.add(doc_id)
        except Exception:
            continue
    
    # Р—С–Р±СЂР°С‚Рё doc_id Р· eval
    for idx in range(len(eval_dataset)):
        try:
            if isinstance(eval_dataset, Subset):
                item = eval_dataset.dataset[eval_dataset.indices[idx]]
            else:
                item = eval_dataset[idx]
            
            if isinstance(item, tuple) and len(item) >= 3:
                doc_id = str(item[2])
            elif isinstance(item, dict):
                doc_id = str(item.get(doc_id_key, item.get('doc_id', None)))
                if doc_id == 'None':
                    continue
            else:
                continue
            
            eval_doc_ids.add(doc_id)
        except Exception:
            continue
    
    # РџРµСЂРµРІС–СЂРёС‚Рё overlap
    overlap = train_doc_ids & eval_doc_ids
    
    if overlap:
        print(f"вљ пёЏ  Р—РЅР°Р№РґРµРЅРѕ overlap doc_ids: {len(overlap)} РґРѕРєСѓРјРµРЅС‚С–РІ")
        print(f"   РџСЂРёРєР»Р°РґРё: {list(overlap)[:5]}")
        return False
    
    print(f"вњ… РќРµРјР°С” overlap: {len(train_doc_ids)} train docs, {len(eval_doc_ids)} eval docs")
    return True
