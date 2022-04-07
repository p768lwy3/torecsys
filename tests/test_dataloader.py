import unittest

from parameterized import parameterized

import torecsys.data.dataloader as trs_data


class DataloaderTestCase(unittest.TestCase):
    @parameterized.expand([
        (0, 0, 1, [[1, 2, 3, 4, 5], [0, 2, 5, 7], [1, 3, 5, 6, 8]])
    ])
    def test_index_field(self, unk_index, unk_token, new_unk_token, dataset):
        index_field = trs_data.fields.IndexField(unk_index, unk_token)

        self.assertEqual(index_field.unk_index, unk_index)
        self.assertEqual(index_field.unk_token, unk_token)
        self.assertEqual(index_field.current_max_token, unk_token)

        index_field.unk_token = new_unk_token
        self.assertEqual(index_field.unk_token, new_unk_token)
        self.assertEqual(index_field.current_max_token, new_unk_token)

        index_field.build_vocab(dataset=dataset)

        row_0 = dataset[0]
        row_0_tokens = index_field.indices(row_0)
        row_0_indices = index_field.tokens(row_0_tokens)
        self.assertEqual(row_0, row_0_indices)

    @parameterized.expand([
        (["Hello world", "Python zen", "python is beautiful"], 7, 6)
    ])
    def test_sentence_field(self, dataset, total_words, total_lower_words):
        sentence_field_default = trs_data.fields.SentenceField()
        self.assertEqual(len(sentence_field_default), 2)

        sentence_field_default.build_vocab(dataset=dataset)
        self.assertEqual(len(sentence_field_default), total_words + 2)

        sentence_field_custom = trs_data.fields.SentenceField(
            tokenize=lambda sent: [w.lower() for w in sent.split()]
        )

        self.assertEqual(len(sentence_field_custom), 2)

        sentence_field_custom.build_vocab(dataset=dataset)
        self.assertEqual(len(sentence_field_custom), total_lower_words + 2)

    @parameterized.expand([
        ({}, [])
    ])
    def test_collate_fn(self, schema, batch_data):
        dataloader = trs_data.CollateFunction(schema=schema)
        outputs = dataloader.to_tensor(batch_data=batch_data)
        self.assertIsInstance(outputs, dict)


if __name__ == '__main__':
    unittest.main()
