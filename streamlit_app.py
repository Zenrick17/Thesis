r"""PetClassNet: outline-defense disease classification prototype.

HOW TO RUN ON WINDOWS (Python 3.11 or 3.12, 64-bit)
------------------------------------------------
1. Extract the complete prototype package into a folder.
2. Run each separate Colab notebook for its 25 epochs, then download its model ZIP.
3. Extract BOTH model ZIPs into this same folder. Expected files:
       models/baseline.keras       models/baseline.json
       models/proposed.keras       models/proposed.json
   Keep each .keras file and its matching .json metadata together.
4. Open PowerShell in this folder and run:
       py -3.12 -m venv .venv
       .\.venv\Scripts\python.exe -m pip install -r requirements.txt
       .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
   For Python 3.11, replace -3.12 with -3.11 in the first command.
5. Open http://localhost:8501 if the browser does not open automatically.
6. Choose a model, upload a clear cat/dog affected-area photograph, and click Classify image.
   Press Ctrl+C in PowerShell to stop the app.

macOS/Linux: python3.12 -m venv .venv; .venv/bin/python -m pip install -r
requirements.txt; .venv/bin/python -m streamlit run streamlit_app.py.

No trained weights are bundled: models must be exported from the Colab notebooks.
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


@st.cache_resource(show_spinner='Loading the selected model…')
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
    st.subheader(ROLE_LABELS[role])
    top = result['top3']
    probabilities = result['probabilities']
    st.write('Highest-scoring category')
    st.write(f'**{CLASS_NAMES[int(top[0])]}**')
    st.metric('Model score', f'{probabilities[top[0]]:.2%}')
    table = pd.DataFrame({'Rank': [1, 2, 3], 'Category': [CLASS_NAMES[int(i)] for i in top],
                          'Model score (%)': [round(float(probabilities[i]) * 100, 2) for i in top]})
    st.dataframe(table, hide_index=True, width='stretch')
    st.caption('These retain the original twelve-class scores; the top three may total less than 100%.')
    if result['maps']:
        for rank, (class_id, heatmap) in enumerate(zip(top, result['maps']), 1):
            st.image(heatmap_overlay(rgb, heatmap['values'], heatmap['flat']),
                     caption=f'{rank}. {CLASS_NAMES[int(class_id)]} · class-specific heatmap', width='stretch')
            if heatmap['flat']:
                st.caption('No positive, varying heatmap was produced for this class; the original image is shown.')
        st.caption('Heatmaps show areas influencing a class score. They do not verify lesion location or disease severity.')
    with st.expander('All twelve model scores'):
        st.dataframe(pd.DataFrame({'Category': CLASS_NAMES, 'Model score (%)': np.round(probabilities * 100, 2)}),
                     hide_index=True, width='stretch')
    payload = {'model': metadata['configuration'], 'selected_epoch': metadata['selected_epoch'],
               'scope': 'outline-defense prototype; not a confirmed diagnosis',
               'scores': {n: float(v) for n, v in zip(CLASS_NAMES, probabilities)}}
    st.download_button('Download prediction JSON', json.dumps(payload, indent=2),
                       file_name=f'petclassnet_{role}_prediction.json', mime='application/json', key=f'json_{role}')


def main():
    st.set_page_config(page_title='PetClassNet · Skin Classifier', page_icon='🐾', layout='wide')
    st.title('🐾 PetClassNet')
    st.write('Cat and dog disease classification · outline-defense prototype')
    st.info('This prototype compares images with 12 supported categories covering skin, eye, ear and dental conditions. It cannot identify healthy pets '
            'or unsupported conditions. Model scores are not confirmed diagnoses; consult a veterinarian about health concerns.')
    with st.sidebar:
        st.header('Demo controls')
        default_dir = os.environ.get('PETCLASSNET_MODELS_DIR', str(APP_DIR / 'models'))
        model_dir = Path(st.text_input('Model folder', value=default_dir)).expanduser()
        show_heatmaps = st.checkbox('Show class heatmaps', value=True)
        st.caption('Baseline: geometric augmentation + cross-entropy.\n\n'
                   'Proposed: additional brightness, contrast and scale augmentation + class-balanced focal loss.')
        with st.expander('How to run this app'):
            st.markdown('Install Python 3.12 and extract the package. Train both Colab notebooks and extract '
                        'their model ZIPs into the app folder. Open PowerShell in that folder:')
            st.code('py -3.12 -m venv .venv\n'
                    '.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n'
                    '.\\.venv\\Scripts\\python.exe -m streamlit run streamlit_app.py', language='powershell')
            st.caption('Open http://localhost:8501. Stop with Ctrl+C. Full instructions are also at the top of this Python file and in README.md.')
    available, errors = {}, {}
    for role in ROLE_LABELS:
        try:
            available[role] = read_export_metadata(model_dir, role)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors[role] = str(error)
    if not available:
        st.warning('No trained model is ready yet. Run the Colab notebooks, then extract the model ZIPs into models/.')
        st.code('models/\n  baseline.keras\n  baseline.json\n  proposed.keras\n  proposed.json')
        with st.expander('Model setup details'):
            for role, error in errors.items():
                st.write(f'{ROLE_LABELS[role]}: {error}')
        st.stop()
    options = list(available)
    # Proposed seed42 was selected as the demonstration configuration in advance.
    options = sorted(options, key=lambda r: r != 'proposed')
    matched = len(available) == 2 and all(available['baseline'][1][k] == available['proposed'][1][k]
                                       for k in ['protocol_fingerprint', 'data_fingerprint', 'manifest_sha256'])
    if matched:
        options.append('compare')
    elif len(available) == 2:
        st.warning('These two exports use different experiment data or settings. Re-export a matched pair to enable side-by-side comparison.')
    selected = st.selectbox('Model', options, format_func=lambda r: 'Compare both models' if r == 'compare' else ROLE_LABELS[r])
    roles = list(available) if selected == 'compare' else [selected]
    with st.expander('Training details and validation results'):
        for role in roles:
            meta = available[role][1]
            st.write(f"**{ROLE_LABELS[role]}** · 25 epochs completed · selected epoch {meta['selected_epoch']} · seed 42")
            validation = meta.get('validation', {})
            if validation:
                st.write(f"Validation macro-F1: {validation['macro_f1']:.4f}; validation accuracy: "
                         f"{validation['top1_accuracy']:.2%}; validation images: {validation['n']}.")
        st.caption('Validation selects the checkpoint. These single-seed prototype results do not establish general improvement.')
    uploaded = st.file_uploader('Upload a cat or dog affected-area photograph', type=['jpg', 'jpeg', 'png', 'webp', 'bmp'])
    if uploaded is None:
        st.write('Upload an image to begin. Use a clear photograph of the affected area.')
        return
    try:
        rgb = decode_uploaded_image(uploaded.getvalue())
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        st.error(f'Unable to read the photograph: {error}')
        return
    left, right = st.columns([1, 2])
    with left:
        st.image(rgb, caption='Uploaded photograph', width='stretch')
    with right:
        result_key = hashlib.sha256(uploaded.getvalue()).hexdigest() + json.dumps({
            'models': {r: available[r][1]['model_sha256'] for r in roles}, 'heatmaps': show_heatmaps}, sort_keys=True)
        if st.button('Classify image', type='primary'):
            predictions = {}
            for role in roles:
                try:
                    path, meta = available[role]
                    stat = path.stat()
                    model, gradient_model = load_export(str(path.resolve()), meta['model_sha256'], stat.st_size, stat.st_mtime_ns)
                    with st.spinner('Classifying the photograph…'):
                        predictions[role] = predict_with_gradcam(model, gradient_model, rgb, show_heatmaps)
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    st.error(f'Unable to run {ROLE_LABELS[role]}: {error}')
                    st.caption('Re-extract the matching model ZIP and install requirements.txt in the app environment.')
            # Keep the result visible when download/expander controls rerun the app.
            st.session_state['prediction_result'] = {'key': result_key, 'predictions': predictions}
        saved = st.session_state.get('prediction_result', {})
        if saved.get('key') != result_key:
            st.caption('Click Classify image to run the selected model(s) on this photograph.')
            return
        columns = st.columns(len(roles))
        for column, role in zip(columns, roles):
            with column:
                if role in saved.get('predictions', {}):
                    render_prediction(role, saved['predictions'][role], rgb, available[role][1])


if __name__ == '__main__':
    main()
