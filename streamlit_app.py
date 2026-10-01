r"""PetClassNet: outline-defense disease classification prototype.

HOW TO RUN ON WINDOWS (Python 3.11 or 3.12, 64-bit)
------------------------------------------------
1. Keep the baseline and enhanced prototype folders beside this app.
   Their exports/models/ folders contain the trained models and metadata.
   Alternatively, place both export pairs in a root models/ folder.
2. Open PowerShell in this folder and run:
       py -3.12 -m venv .venv
       .\.venv\Scripts\python.exe -m pip install -r requirements.txt
       .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
   For Python 3.11, replace -3.12 with -3.11 in the first command.
3. Open http://localhost:8501 if the browser does not open automatically.
4. Upload a clear cat/dog affected-area photograph and click Classify image.
   Press Ctrl+C in PowerShell to stop the app.

macOS/Linux: python3.12 -m venv .venv; .venv/bin/python -m pip install -r
requirements.txt; .venv/bin/python -m streamlit run streamlit_app.py.

The UI displays saved validation results and runs both models on each image.
This app does inference only. It never trains on or stores uploaded photographs.
The proposed model changes training augmentation/loss, not the B0 backbone.
"""
from pathlib import Path
import hashlib
import io
import json
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
import tensorflow as tf

CLASS_NAMES = ['Cat - Dental Disease', 'Cat - Ear Mites', 'Cat - Eye Infection', 'Cat - Ringworm', 'Cat - Scabies', 'Cat - Flea Allergy', 'Dog - Dental Disease', 'Dog - Eye Infection', 'Dog - Fungal Infection', 'Dog - Hot Spots', 'Dog - Mange', 'Dog - Flea Allergy']
NUM_CLASSES = len(CLASS_NAMES)
PREPROCESSING_ID = 'exif_rgb_bilinear224_float32_0_255_b0_internal'
ROLE_LABELS = {'baseline': 'Baseline · G-CE', 'proposed': 'Proposed · T-CBF'}
APP_DIR = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 10 * 1024 * 1024
EXPORT_DIRS = [APP_DIR / folder / 'exports' / 'models' for folder in (
    'PetClassNet_12_Class_Prototype_25_baseline',
    'PetClassNet_12_Class_Prototype_25_enhanced',
    'PetClassNet_12_Class_Prototype_25')]


def discover_exports(model_dir=None):
    """Find each role independently, retaining validation errors for diagnosis."""
    directories = [Path(model_dir).expanduser()] if model_dir else [APP_DIR / 'models', *EXPORT_DIRS]
    available, errors = {}, {}
    for role in ROLE_LABELS:
        problems = []
        for directory in directories:
            try:
                available[role] = read_export_metadata(directory, role)
                break
            except (OSError, ValueError, KeyError, TypeError) as error:
                problems.append(f'{directory}: {error}')
        if role not in available:
            errors[role] = '\n'.join(problems)
    return available, errors


def exports_match(available):
    return len(available) == 2 and all(
        available['baseline'][1][key] == available['proposed'][1][key]
        for key in ['protocol_fingerprint', 'data_fingerprint', 'manifest_sha256'])


def prediction_table(probabilities=None):
    if probabilities is None:
        return pd.DataFrame({'Category': CLASS_NAMES, 'Model score (%)': [None] * NUM_CLASSES})
    order = np.argsort(-probabilities, kind='stable')
    return pd.DataFrame({'Category': [CLASS_NAMES[int(i)] for i in order],
                         'Model score (%)': np.round(probabilities[order] * 100, 2)})


def top3_prediction_table(probabilities=None):
    if probabilities is None:
        return pd.DataFrame({'Rank': [1, 2, 3], 'Category': [None] * 3,
                             'Model score (%)': [None] * 3})
    return prediction_table(probabilities).head(3).assign(Rank=[1, 2, 3])[
        ['Rank', 'Category', 'Model score (%)']]


def prediction_key(image_data, available, show_heatmaps):
    models = {}
    for role, (path, metadata) in available.items():
        stat = path.stat()
        models[role] = [str(path.resolve()), metadata['model_sha256'],
                        stat.st_size, stat.st_mtime_ns]
    return hashlib.sha256(image_data).hexdigest() + json.dumps(
        {'models': models, 'heatmaps': show_heatmaps}, sort_keys=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_export_metadata(model_dir, role):
    model_path = Path(model_dir) / f'{role}.keras'
    metadata_path = Path(model_dir) / f'{role}.json'
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f'Extract the {role} model ZIP here. Both {role}.keras and {role}.json are required.')
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    expected_config = {'baseline': 'G-CE', 'proposed': 'T-CBF'}[role]
    if (meta.get('schema_version') != 2 or meta.get('class_count') != NUM_CLASSES or meta.get('role') != role
            or meta.get('configuration') != expected_config or meta.get('class_names') != CLASS_NAMES
            or meta.get('input_shape') != [224, 224, 3]
            or meta.get('preprocessing_id') != PREPROCESSING_ID):
        raise ValueError('This model/metadata does not match the twelve-class prototype. Re-export it from the supplied notebook.')
    if meta.get('epochs_completed') != 25 or meta.get('seed') != 42:
        raise ValueError('This app expects completed 25-epoch seed-42 prototype exports.')
    if not 1 <= int(meta.get('selected_epoch', 0)) <= 25:
        raise ValueError('The selected checkpoint epoch is invalid.')
    if meta.get('recipe', {}).get('warmup_epochs') != 5 or meta.get('recipe', {}).get('finetune_epochs') != 20:
        raise ValueError('Expected five warm-up and 20 fine-tuning epochs.')
    if not isinstance(meta.get('model_sha256'), str) or len(meta['model_sha256']) != 64:
        raise ValueError('The export is missing its model checksum.')
    for key in ['protocol_fingerprint', 'data_fingerprint', 'manifest_sha256']:
        if not isinstance(meta.get(key), str) or len(meta[key]) != 64:
            raise ValueError(f'The export is missing {key}.')
    return model_path, meta


@st.cache_resource(show_spinner='Loading the trained model…')
def load_export(model_path_string, model_sha256, file_size, modified_ns):
    # File size/time are part of the cache key; replacing a model triggers a reload.
    path = Path(model_path_string)
    if sha256_file(path) != model_sha256:
        raise ValueError('The model file and metadata checksum differ. Extract the original model ZIP again.')
    model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    if model.input_shape != (None, 224, 224, 3) or model.output_shape != (None, NUM_CLASSES):
        raise ValueError('The saved model has an unexpected input or output shape.')
    activation = model.get_layer('top_activation')
    logits = model.get_layer('class_logits')
    if logits.units != NUM_CLASSES or tf.keras.activations.serialize(logits.activation) != 'linear':
        raise ValueError('The model must expose the twelve linear logits for Grad-CAM.')
    model.get_layer('rescaling')  # B0 preprocessing lives inside the exported model.
    gradient_model = tf.keras.Model(model.inputs[0], [activation.output, logits.output])
    return model, gradient_model


def decode_uploaded_image(data):
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Please upload an image smaller than 10 MB.')
    with Image.open(io.BytesIO(data)) as source:
        if getattr(source, 'n_frames', 1) != 1:
            raise ValueError('Use a single photograph rather than an animated image.')
        source.load()
        rgb = ImageOps.exif_transpose(source).convert('RGB').copy()
    return rgb


def prepare_image(rgb):
    # Same EXIF/RGB and Pillow bilinear resize as training and validation.
    # DO NOT divide by 255: EfficientNet-B0 already includes rescaling.
    resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
    return tf.convert_to_tensor(np.asarray(resized, dtype=np.float32)[None, ...])


def predict_with_gradcam(model, gradient_model, rgb, show_heatmaps=True):
    batch = prepare_image(rgb)
    probabilities = np.asarray(model(batch, training=False)[0], dtype=np.float64)
    if (probabilities.shape != (NUM_CLASSES,) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.isclose(probabilities.sum(), 1, atol=1e-5, rtol=0)):
        raise ValueError('The model returned invalid twelve-class probabilities.')
    top = np.argsort(-probabilities, kind='stable')[:3]
    maps = []
    if show_heatmaps:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(batch)  # Enables attribution with a frozen backbone too.
            activation, logits = gradient_model(batch, training=False)
            scores = tf.unstack(tf.gather(logits[0], top))
        for score in scores:
            gradients = tape.gradient(score, activation)
            if gradients is None:
                raise RuntimeError('The class score is disconnected from the feature map.')
            weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
            values = np.asarray(tf.nn.relu(tf.reduce_sum(activation * weights, axis=-1))[0], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError('The heatmap contains invalid values.')
            lo, hi = float(values.min()), float(values.max())
            flat = hi <= 0 or hi - lo <= 1e-12
            normalized = np.zeros_like(values) if flat else (values - lo) / (hi - lo)
            maps.append({'values': normalized, 'flat': flat})
        del tape
    return {'probabilities': probabilities, 'top3': top, 'maps': maps}


def heatmap_overlay(rgb, values, flat=False):
    if flat:
        return rgb.copy()
    heat = np.asarray(Image.fromarray(values.astype(np.float32)).resize(rgb.size, Image.Resampling.BILINEAR))
    # A simple blue→cyan→yellow→red color ramp, with no plotting dependency.
    heat = np.clip(heat, 0, 1)
    colors = np.stack([np.clip(1.5 - np.abs(4 * heat - 3), 0, 1),
                       np.clip(1.5 - np.abs(4 * heat - 2), 0, 1),
                       np.clip(1.5 - np.abs(4 * heat - 1), 0, 1)], axis=-1)
    colored = Image.fromarray(np.uint8(colors * 255))
    return Image.blend(rgb, colored, alpha=0.4)


def render_prediction(role, result, rgb, metadata):
    top = result['top3']
    probabilities = result['probabilities']
    st.write('Highest-scoring category')
    st.write(f'**{CLASS_NAMES[int(top[0])]}**')
    st.metric('Model score', f'{probabilities[top[0]]:.2%}')
    st.markdown('**Top 3 predictions**')
    st.dataframe(top3_prediction_table(probabilities), hide_index=True, width='stretch')
    st.caption('Scores retain their original twelve-class probabilities; the top three may total less than 100%.')
    with st.expander('All twelve model scores'):
        st.dataframe(prediction_table(probabilities), hide_index=True, width='stretch', height=460)
    if result['maps']:
        st.markdown('**Grad-CAM · Top 3 predictions**')
        st.caption('Warmer colors show image regions that contributed more to each category’s score.')
        for rank, (class_id, heatmap) in enumerate(zip(top, result['maps']), 1):
            st.image(heatmap_overlay(rgb, heatmap['values'], heatmap['flat']),
                     caption=f'{rank}. {CLASS_NAMES[int(class_id)]} · {probabilities[class_id]:.2%} · Grad-CAM',
                     width='stretch')
            if heatmap['flat']:
                st.caption('No positive, varying heatmap was produced for this class; the original image is shown.')
        st.caption('Heatmaps show areas influencing a class score. They do not verify lesion location or disease severity.')
    payload = {'model': metadata['configuration'], 'selected_epoch': metadata['selected_epoch'],
               'scope': 'outline-defense prototype; not a confirmed diagnosis',
               'scores': {n: float(v) for n, v in zip(CLASS_NAMES, probabilities)}}
    st.download_button('Download prediction JSON', json.dumps(payload, indent=2),
                       file_name=f'petclassnet_{role}_prediction.json', mime='application/json', key=f'json_{role}')


def main():
    st.set_page_config(page_title='PetClassNet · Model Comparison', layout='wide')
    st.title('PetClassNet')
    st.write('Compare baseline and proposed models on the same cat or dog photograph.')
    st.caption('Twelve supported categories covering skin, eye, ear and dental conditions. '
               'Model scores are not confirmed diagnoses.')
    with st.sidebar:
        st.header('Comparison settings')
        show_heatmaps = st.checkbox('Show Grad-CAM for top 3 predictions', value=True,
                                    help='Generate a separate heatmap for each of the three highest-scoring categories in both models.')
        st.caption('Baseline: geometric augmentation + cross-entropy.\n\n'
                   'Proposed: additional brightness, contrast and scale augmentation + class-balanced focal loss.')
        with st.expander('Model file settings'):
            model_dir = st.text_input('Model folder override', value=os.environ.get('PETCLASSNET_MODELS_DIR', ''),
                                      help='Leave blank to find the saved exports in the project folders automatically.')
    available, errors = discover_exports(model_dir.strip() or None)
    matched = exports_match(available)
    with st.sidebar.expander('Model file details'):
        for role, (path, _) in available.items():
            st.write(f'{ROLE_LABELS[role]}: {path.relative_to(APP_DIR) if path.is_relative_to(APP_DIR) else path}')
        for role, error in errors.items():
            st.text(f'{ROLE_LABELS[role]}: {error}')
    if len(available) == 2 and not matched:
        st.warning('The exports have different experiment data or settings. Their predictions are shown, '
                   'but their validation scores are not a matched experiment comparison.')

    st.subheader('Validation results')
    st.caption('Saved results from training; these evaluate the validation set, not the photograph you upload below.')
    for column, role in zip(st.columns(2), ROLE_LABELS):
        with column:
            st.markdown(f'**{ROLE_LABELS[role]}**')
            if role in available:
                meta = available[role][1]
                validation = meta.get('validation', {})
                st.dataframe(pd.DataFrame({
                    'Metric': ['Accuracy', 'Macro-F1', 'Top-3 accuracy', 'Validation images', 'Selected epoch'],
                    'Value': [f"{validation['top1_accuracy']:.2%}" if 'top1_accuracy' in validation else 'Unavailable',
                              f"{validation['macro_f1']:.4f}" if 'macro_f1' in validation else 'Unavailable',
                              f"{validation['top3_accuracy']:.2%}" if 'top3_accuracy' in validation else 'Unavailable',
                              str(validation.get('n', 'Unavailable')), str(meta['selected_epoch'])]}),
                    hide_index=True, width='stretch')
                st.caption(f"{meta['epochs_completed']} epochs completed · seed {meta['seed']}")
            else:
                st.info('The trained export is unavailable in this deployment. Add its model export to the project folders.')
    if matched:
        baseline = available['baseline'][1].get('validation', {})
        proposed = available['proposed'][1].get('validation', {})
        if 'top1_accuracy' in baseline and 'top1_accuracy' in proposed:
            delta = 100 * (proposed['top1_accuracy'] - baseline['top1_accuracy'])
            st.caption(f'Proposed − baseline validation accuracy: {delta:+.2f} percentage points. '
                       'These are single-seed prototype results.')

    st.subheader('Classify a photograph')
    uploaded = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
                                help='Use one clear photograph of the affected area, up to 10 MB.')
    rgb, result_key = None, None
    if uploaded is not None:
        try:
            image_data = uploaded.getvalue()
            rgb = decode_uploaded_image(image_data)
            result_key = prediction_key(image_data, available, show_heatmaps)
        except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            st.error(f'Unable to read the photograph: {error}')
    if rgb is not None:
        preview, instruction = st.columns([1, 2])
        with preview:
            st.image(rgb, caption='The same photograph is used for both models.', width='stretch')
        with instruction:
            st.write('Run the trained models to compare their predicted categories and scores.')
            classify = st.button('Classify image', type='primary', disabled=not available)
        if classify:
            predictions, inference_errors = {}, {}
            for role, (path, meta) in available.items():
                try:
                    stat = path.stat()
                    with st.spinner(f'Running {ROLE_LABELS[role]}…'):
                        model, gradient_model = load_export(str(path.resolve()), meta['model_sha256'],
                                                           stat.st_size, stat.st_mtime_ns)
                        predictions[role] = predict_with_gradcam(model, gradient_model, rgb, show_heatmaps)
                except (OSError, ValueError, RuntimeError, KeyError, tf.errors.OpError) as error:
                    inference_errors[role] = str(error)
            st.session_state['prediction_result'] = {
                'key': result_key, 'predictions': predictions, 'errors': inference_errors}
    saved = st.session_state.get('prediction_result', {})
    current = result_key is not None and saved.get('key') == result_key
    predictions = saved.get('predictions', {}) if current else {}
    inference_errors = saved.get('errors', {}) if current else {}
    st.subheader('Image results')
    for column, role in zip(st.columns(2), ROLE_LABELS):
        with column:
            st.markdown(f'**{ROLE_LABELS[role]}**')
            if role in predictions:
                render_prediction(role, predictions[role], rgb, available[role][1])
            else:
                if role in errors:
                    st.info('Trained model export unavailable.')
                elif role in inference_errors:
                    st.error(f'Unable to classify this image: {inference_errors[role]}')
                else:
                    st.caption('Upload a photograph and click Classify image to see the scores.')
                st.markdown('**Top 3 predictions**')
                st.dataframe(top3_prediction_table(), hide_index=True, width='stretch')
    if len(predictions) == 2:
        first = int(predictions['baseline']['top3'][0])
        second = int(predictions['proposed']['top3'][0])
        if first == second:
            st.info(f'Both models predict {CLASS_NAMES[first]}.')
        else:
            st.info(f'The models disagree: baseline predicts {CLASS_NAMES[first]}; '
                    f'proposed predicts {CLASS_NAMES[second]}.')
        st.caption('A known reference label is needed to determine which prediction is correct.')


if __name__ == '__main__':
    main()
