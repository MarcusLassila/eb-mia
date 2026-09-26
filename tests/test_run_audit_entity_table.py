import unittest
from collections import defaultdict

import torch

from mia.utils import select_entity_audit_indices


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


class TestSelectEntityAuditIndices(unittest.TestCase):
    def test_filters_entity_sizes_and_balances_entities(self):
        torch.manual_seed(0)
        dataset = DummyEntityDataset([
            0, 0,
            1, 1, 1,
            2,
            3, 3,
            4, 4, 4, 4,
            5, 5,
        ])
        target_train_index = torch.tensor([1, 3, 5], dtype=torch.long)  # target entities: 0, 1, 2
        target_train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        target_train_mask[target_train_index] = True

        audit_table = select_entity_audit_indices(
            dataset.get_entity_index_table(),
            target_train_mask,
            mode="all",
            min_samples_per_entity=1,
            max_samples_per_entity=3,
        )

        self.assertEqual(list(audit_table.keys()), [0, 1, 2, 3, 4, 5])
        self.assertEqual(audit_table[0], [0, 1])
        self.assertEqual(audit_table[1], [2, 3, 4])
        self.assertEqual(audit_table[2], [5])
        self.assertEqual(audit_table[3], [6, 7])
        self.assertEqual(audit_table[4], [8, 9, 11])
        self.assertEqual(audit_table[5], [12, 13])

        target_entities = {0, 1, 2}
        n_target = sum(entity_id in target_entities for entity_id in audit_table.keys())
        self.assertEqual(n_target, len(audit_table) // 2)

    def test_exclude_train_mode_uses_target_entity_labels_for_balancing(self):
        torch.manual_seed(0)
        dataset = DummyEntityDataset([
            0, 0,
            1,
            2,
        ])
        target_train_index = torch.tensor([0], dtype=torch.long)  # target entity: 0
        target_train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        target_train_mask[target_train_index] = True

        audit_table = select_entity_audit_indices(
            dataset.get_entity_index_table(),
            target_train_mask,
            mode="exclude_train",
            min_samples_per_entity=1,
            max_samples_per_entity=1,
        )

        self.assertEqual(list(audit_table.keys()), [0, 1])
        self.assertEqual(audit_table[0], [1])
        self.assertEqual(audit_table[1], [2])

    def test_exact_samples_per_entity_preserves_kept_target_sample_in_max_one_train_mode(self):
        torch.manual_seed(0)
        dataset = DummyEntityDataset([
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2,
            3, 3, 3,
            4,
        ])
        target_train_index = torch.tensor([2, 3, 4], dtype=torch.long)  # target entities: 0, 1
        target_train_mask = torch.zeros(len(dataset), dtype=torch.bool)
        target_train_mask[target_train_index] = True

        audit_table = select_entity_audit_indices(
            dataset.get_entity_index_table(),
            target_train_mask,
            mode="max_one_train",
            min_samples_per_entity=2,
            max_samples_per_entity=2,
        )

        self.assertEqual(list(audit_table.keys()), [0, 1, 2, 3])
        self.assertEqual(audit_table[0], [0, 2])
        self.assertEqual(audit_table[1], [4, 5])
        self.assertEqual(audit_table[2], [8, 9])
        self.assertEqual(audit_table[3], [11, 12])
        self.assertNotIn(4, audit_table)
        self.assertIn(2, audit_table[0])
        self.assertIn(4, audit_table[1])

    def test_balancing_is_random_reproducible_and_independent_of_table_order(self):
        table = {entity: [2 * entity, 2 * entity + 1] for entity in range(20)}
        membership = torch.zeros(40, dtype=torch.bool)
        membership[0] = True
        selections = set()
        for seed in range(8):
            generator = torch.Generator().manual_seed(seed)
            audit = select_entity_audit_indices(table, membership, "exclude_train", 1, 1, generator=generator)
            torch.rand(100)
            reversed_table = dict(reversed(list(table.items())))
            repeated = select_entity_audit_indices(
                reversed_table, membership, "exclude_train", 1, 1,
                generator=torch.Generator().manual_seed(seed),
            )
            self.assertEqual(audit, repeated)
            self.assertEqual(len(audit), 2)
            self.assertEqual(audit[0], [1])
            selections.update(entity for entity in audit if entity != 0)
        self.assertGreater(len(selections), 1)


if __name__ == "__main__":
    unittest.main()
