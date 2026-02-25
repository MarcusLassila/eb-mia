import unittest
from collections import defaultdict

import torch

from mia.run_audit import get_entity_audit_table


class DummyEntityDataset:

    def __init__(self, entity_ids):
        self.entity_ids = torch.tensor(entity_ids, dtype=torch.long)

    def get_entity_index_table(self):
        table = defaultdict(list)
        for idx, entity_id in enumerate(self.entity_ids.tolist()):
            table[entity_id].append(idx)
        return table

    def __getitem__(self, index):
        return int(index)

    def __len__(self):
        return len(self.entity_ids)


class TestGetEntityAuditTable(unittest.TestCase):
    def test_filters_entity_sizes_and_balances_entities(self):
        dataset = DummyEntityDataset([
            0, 0,
            1, 1, 1,
            2,
            3, 3,
            4, 4, 4, 4,
            5, 5,
        ])
        target_train_index = torch.tensor([1, 3, 5], dtype=torch.long)  # target entities: 0, 1, 2

        audit_table = get_entity_audit_table(
            dataset,
            target_train_index,
            mode="all",
            min_samples_per_entity=1,
            max_samples_per_entity=3,
        )

        self.assertEqual(list(audit_table.keys()), [0, 1, 3, 5])
        self.assertEqual(audit_table[0].indices, [0, 1])
        self.assertEqual(audit_table[1].indices, [2, 3, 4])
        self.assertEqual(audit_table[3].indices, [6, 7])
        self.assertEqual(audit_table[5].indices, [12, 13])

        target_entities = {0, 1, 2}
        n_target = sum(entity_id in target_entities for entity_id in audit_table.keys())
        self.assertEqual(n_target, len(audit_table) // 2)

    def test_exclude_train_mode_uses_target_entity_labels_for_balancing(self):
        dataset = DummyEntityDataset([
            0, 0,
            1,
            2,
        ])
        target_train_index = torch.tensor([0], dtype=torch.long)  # target entity: 0

        audit_table = get_entity_audit_table(
            dataset,
            target_train_index,
            mode="exclude_train",
            min_samples_per_entity=1,
            max_samples_per_entity=1,
        )

        self.assertEqual(list(audit_table.keys()), [0, 1])
        self.assertEqual(audit_table[0].indices, [1])
        self.assertEqual(audit_table[1].indices, [2])

    def test_exact_samples_per_entity_preserves_kept_target_sample_in_max_one_train_mode(self):
        dataset = DummyEntityDataset([
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2,
            3, 3, 3,
            4,
        ])
        target_train_index = torch.tensor([2, 3, 4], dtype=torch.long)  # target entities: 0, 1

        audit_table = get_entity_audit_table(
            dataset,
            target_train_index,
            mode="max_one_train_sample",
            n_audit_samples_per_entity=2,
        )

        self.assertEqual(list(audit_table.keys()), [0, 1, 2, 3])
        self.assertEqual(len(audit_table[0].indices), 2)
        self.assertEqual(len(audit_table[1].indices), 2)
        self.assertEqual(len(audit_table[2].indices), 2)
        self.assertEqual(len(audit_table[3].indices), 2)
        self.assertNotIn(4, audit_table)
        self.assertIn(2, audit_table[0].indices)
        self.assertIn(4, audit_table[1].indices)


if __name__ == "__main__":
    unittest.main()
