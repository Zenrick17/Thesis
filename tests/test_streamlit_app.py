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
            self.assertEqual(len(table.value), 3)
            self.assertEqual(table.value['Rank'].tolist(), [1, 2, 3])
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
                self.assertEqual(len(result['maps']), 3)
                for heatmap in result['maps']:
                    values = heatmap['values']
                    self.assertEqual(values.ndim, 2)
                    self.assertTrue(np.isfinite(values).all())
                    self.assertTrue(((values >= 0) & (values <= 1)).all())
                    rgb = app.decode_uploaded_image(uploaded.getvalue())
                    self.assertEqual(app.heatmap_overlay(rgb, values, heatmap['flat']).size, rgb.size)
            top3_tables = [table.value for table in page.dataframe if 'Rank' in table.value.columns]
            self.assertEqual(len(top3_tables), 2)
            for role, table in zip(app.ROLE_LABELS, top3_tables):
                self.assertEqual(table['Rank'].tolist(), [1, 2, 3])
                self.assertEqual(table['Category'].tolist(),
                                 [app.CLASS_NAMES[int(i)] for i in predictions[role]['top3']])
                self.assertTrue(table['Model score (%)'].is_monotonic_decreasing)
            self.assertEqual(len(page.dataframe), 6)
            page.run(timeout=30)
            self.assertFalse(page.exception)
            self.assertEqual(len(page.metric), 2)
            # Grad-CAM can be disabled while top-three predictions remain available.
            page.checkbox[0].uncheck().run(timeout=30)
            self.assertEqual(len(page.metric), 0)
            page.button[0].click().run(timeout=120)
            self.assertFalse(page.exception)
            self.assertFalse(page.error)
            without_heatmaps = page.session_state['prediction_result']['predictions']
            for role, result in without_heatmaps.items():
                self.assertFalse(result['maps'])
                np.testing.assert_allclose(result['probabilities'], predictions[role]['probabilities'])
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
