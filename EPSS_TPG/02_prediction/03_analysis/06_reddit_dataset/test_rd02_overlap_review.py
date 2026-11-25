import json
from pathlib import Path
import tempfile
import unittest
from review_rd02_overlap import structure, table


class ReviewTests(unittest.TestCase):
    def test_metadata_does_not_export_words(self):
        left='private vendor wording containing secret example'
        right=left+' additional'
        result=structure(left,right,'five_word_jaccard')
        serialized=json.dumps(result)
        for word in ('private','vendor','secret','additional'):
            self.assertNotIn(word,serialized)
        self.assertEqual(result['change_blocks'][0]['right_tokens'],1)
        self.assertFalse(result['normalized_identical'])

    def test_structural_classes_not_duplicate_cve_judgments(self):
        self.assertEqual(structure('Product elevation of privilege','Product elevation of privilege',
                                  'normalized_exact')['classification'],'short_product_vulnerability_title')
        text=' '.join(['word']*25)
        self.assertEqual(structure(text,text,'normalized_exact')['classification'],'shared_mechanism_description')

    def test_canonical_csv_newlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'table.csv'; rows=[{'kind':'exact','roles':'train,validation'}]
            table(path,rows); first=path.read_bytes(); table(path,rows)
            self.assertEqual(first,path.read_bytes()); self.assertNotIn(b'\r',first)


if __name__=='__main__': unittest.main()
