import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

import streamlit_app as app


class ModelComparisonTests(unittest.TestCase):
    def test_saved_exports_are_discovered_and_matched(self):
        available, errors = app.discover_exports()
        self.assertEqual(set(available), {'baseline', 'proposed'})
        self.assertFalse(errors)
        self.assertTrue(app.exports_match(available))
        for path, metadata in available.values():
            self.assertEqual(app.sha256_file(path), metadata['model_sha256'])

    def test_missing_exports_keep_uploader_and_both_result_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict('os.environ', {'PETCLASSNET_MODELS_DIR': folder}):
                page = AppTest.from_file(str(app.APP_DIR / 'streamlit_app.py')).run(timeout=30)
        self.assertFalse(page.exception)
        self.assertEqual(len(page.get('file_uploader')), 1)
        self.assertEqual(len(page.dataframe), 2)
        for table in page.dataframe:
            self.assertEqual(len(table.value), app.NUM_CLASSES)
            self.assertTrue(table.value['Model score (%)'].isna().all())

    def test_uploaded_image_runs_both_real_models_and_results_persist(self):
        # Use a saved demonstration image, never the held-out test partition.
        image_path = next((app.APP_DIR / 'PetClassNet_12_Class_Prototype_25_baseline'
                           / 'gradcam').rglob('input.png'))
        uploaded = io.BytesIO(image_path.read_bytes())
        with patch.object(st, 'file_uploader', return_value=uploaded):
            page = AppTest.from_file(str(app.APP_DIR / 'streamlit_app.py')).run(timeout=60)
            self.assertFalse(page.exception)
            self.assertEqual(len(page.dataframe), 4)
            page.button[0].click().run(timeout=120)
            self.assertFalse(page.exception)
            self.assertFalse(page.error)
            predictions = page.session_state['prediction_result']['predictions']
            self.assertEqual(set(predictions), {'baseline', 'proposed'})
            for result in predictions.values():
                scores = result['probabilities']
                self.assertEqual(scores.shape, (12,))
                self.assertTrue(np.isfinite(scores).all())
                self.assertAlmostEqual(float(scores.sum()), 1, places=5)
            for table in list(page.dataframe)[2:]:
                self.assertEqual(len(table.value), 12)
                self.assertTrue(table.value['Model score (%)'].is_monotonic_decreasing)
            page.run(timeout=30)
            self.assertFalse(page.exception)
            self.assertEqual(len(page.metric), 2)
        # A changed photograph must not show scores from the previous image.
        replacement = io.BytesIO()
        Image.new('RGB', (40, 40), 'blue').save(replacement, format='PNG')
        with patch.object(st, 'file_uploader', return_value=replacement):
            page.run(timeout=30)
            self.assertFalse(page.exception)
            self.assertEqual(len(page.metric), 0)
            for table in list(page.dataframe)[2:]:
                self.assertTrue(table.value['Model score (%)'].isna().all())

    def test_invalid_image_reports_error_without_hiding_panels(self):
        with patch.object(st, 'file_uploader', return_value=io.BytesIO(b'not an image')):
            page = AppTest.from_file(str(app.APP_DIR / 'streamlit_app.py')).run(timeout=30)
        self.assertFalse(page.exception)
        self.assertEqual(len(page.error), 1)
        self.assertEqual(len(page.dataframe), 4)


if __name__ == '__main__':
    unittest.main()
