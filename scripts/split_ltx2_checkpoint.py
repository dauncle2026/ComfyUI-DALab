import argparse
import json
import os
import sys

from safetensors.torch import save_file

comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, comfy_path)

def split_checkpoint(input_path: str, models_dir: str, dry_run: bool = False):
    print(f"Loading checkpoint: {input_path}")
    print(f"File size: {os.path.getsize(input_path) / 1024**3:.2f} GB")

    distill_prefix = "distill_" if "distill" in input_path else ""
    dtype_prefix = "fp8_" if "fp8" in input_path else "fp4_" if "fp4" in input_path else ""

    from safetensors import safe_open

    tensors = {}
    metadata = {}

    with safe_open(input_path, framework="pt", device="cpu") as f:
        metadata = dict(f.metadata()) if f.metadata() else {}
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    print(f"Total keys: {len(tensors)}")
    print(f"Metadata keys: {list(metadata.keys())}")

    config = json.loads(metadata.get('config', '{}'))

    splits = {
        'diffusion': {
            'prefixes': ['model.'],
            'exclude_prefixes': [
                'model.diffusion_model.video_embeddings_connector.',
                'model.diffusion_model.audio_embeddings_connector.',
            ],
            'output_dir': os.path.join(models_dir, 'diffusion_models', 'ltx2'),
            'output_file': f'ltx2_{distill_prefix}diffusion_{dtype_prefix}.safetensors',
            'key_transform': lambda k: k,
            'config_keys': ['transformer'],
            'quant_key_prefix': 'model.diffusion_model.',
        },
        'video_vae': {
            'prefixes': ['vae.'],
            'output_dir': os.path.join(models_dir, 'vae', 'ltx2'),
            'output_file': f'ltx2_{distill_prefix}video_vae_{dtype_prefix}.safetensors',
            'key_transform': lambda k: k.replace('vae.', '', 1),
            'config_keys': ['vae'],
            'quant_key_prefix': '',
        },
        'audio_vae': {
            'prefixes': ['audio_vae.', 'vocoder.'],
            'output_dir': os.path.join(models_dir, 'vae', 'ltx2'),
            'output_file': f'ltx2_{distill_prefix}audio_vae_{dtype_prefix}.safetensors',
            'key_transform': lambda k: k,
            'config_keys': ['audio_vae', 'vocoder'],
            'quant_key_prefix': '',
        },
        'text_proj': {
            'prefixes': [
                'text_embedding_projection.',
                'model.diffusion_model.video_embeddings_connector.',
                'model.diffusion_model.audio_embeddings_connector.',
            ],
            'output_dir': os.path.join(models_dir, 'text_encoders', 'ltx2'),
            'output_file': f'ltx2_{distill_prefix}text_proj_{dtype_prefix}.safetensors',
            'key_transform': lambda k: k.replace('model.diffusion_model.', '').replace(
                'text_embedding_projection.aggregate_embed.weight', 'text_embedding_projection.weight'
            ),
            'config_keys': [],
            'quant_key_prefix': '',
        },
    }

    for split_name, split_config in splits.items():
        print(f"\n{'='*50}")
        print(f"Processing: {split_name}")

        exclude_prefixes = split_config.get('exclude_prefixes', [])
        split_tensors = {}
        for key, tensor in tensors.items():
            excluded = False
            for exclude_prefix in exclude_prefixes:
                if key.startswith(exclude_prefix):
                    excluded = True
                    break
            if excluded:
                continue

            for prefix in split_config['prefixes']:
                if key.startswith(prefix):
                    new_key = split_config['key_transform'](key)
                    split_tensors[new_key] = tensor
                    break

        if not split_tensors:
            print(f"  No tensors found for {split_name}, skipping")
            continue

        total_size = sum(t.numel() * t.element_size() for t in split_tensors.values())
        print(f"  Keys: {len(split_tensors)}")
        print(f"  Size: {total_size / 1024**3:.2f} GB")
        print(f"  Sample keys: {list(split_tensors.keys())[:3]}")

        output_path = os.path.join(split_config['output_dir'], split_config['output_file'])

        if dry_run:
            print(f"  [DRY RUN] Would save to: {output_path}")
            continue

        os.makedirs(split_config['output_dir'], exist_ok=True)

        split_metadata = {}

        if split_config['config_keys']:
            split_config_data = {}
            for config_key in split_config['config_keys']:
                if config_key in config:
                    split_config_data[config_key] = config[config_key]
            if split_config_data:
                split_metadata['config'] = json.dumps(split_config_data)
                print(f"  Config keys included: {list(split_config_data.keys())}")

        if '_quantization_metadata' in metadata:
            try:
                quant_meta = json.loads(metadata['_quantization_metadata'])
                quant_key_prefix = split_config.get('quant_key_prefix', '')

                if 'layers' in quant_meta:
                    filtered_layers = {}
                    tensor_prefixes = set()
                    for tensor_key in split_tensors.keys():
                        parts = tensor_key.rsplit('.', 1)
                        if len(parts) == 2 and parts[1] in ('weight', 'bias', 'input_scale', 'weight_scale'):
                            layer_name = parts[0]
                            if quant_key_prefix and layer_name.startswith(quant_key_prefix):
                                layer_name = layer_name[len(quant_key_prefix):]
                            tensor_prefixes.add(layer_name)
                        tensor_prefixes.add(tensor_key)

                    for quant_key, quant_value in quant_meta['layers'].items():
                        stripped_key = quant_key
                        if quant_key_prefix and quant_key.startswith(quant_key_prefix):
                            stripped_key = quant_key[len(quant_key_prefix):]

                        if stripped_key in tensor_prefixes:
                            filtered_layers[stripped_key] = quant_value

                    if filtered_layers:
                        filtered_quant = {
                            'format_version': quant_meta.get('format_version', '1.0'),
                            'layers': filtered_layers
                        }
                        split_metadata['_quantization_metadata'] = json.dumps(filtered_quant)
                        print(f"  Quantization metadata layers: {len(filtered_layers)}")
                        if filtered_layers:
                            sample_key = list(filtered_layers.keys())[0]
                            print(f"  Sample quant key: {sample_key}")
                else:
                    filtered_quant = {}
                    for quant_key, quant_value in quant_meta.items():
                        if quant_key in split_tensors:
                            filtered_quant[quant_key] = quant_value

                    if filtered_quant:
                        split_metadata['_quantization_metadata'] = json.dumps(filtered_quant)
                        print(f"  Quantization metadata keys: {len(filtered_quant)}")
            except json.JSONDecodeError:
                split_metadata['_quantization_metadata'] = metadata['_quantization_metadata']

        print(f"  Saving to: {output_path}")
        save_file(split_tensors, output_path, metadata=split_metadata)

        saved_size = os.path.getsize(output_path)
        print(f"  Saved size: {saved_size / 1024**3:.2f} GB")

    print(f"\n{'='*50}")
    print("Split complete!")
    print(f"\nOutput files:")
    for split_name, split_config in splits.items():
        output_path = os.path.join(split_config['output_dir'], split_config['output_file'])
        if os.path.exists(output_path):
            print(f"  {output_path}: {os.path.getsize(output_path) / 1024**3:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description='Split LTX2 checkpoint into separate model files')
    parser.add_argument('--input', '-i', required=True, help='Input checkpoint path')
    parser.add_argument('--models-dir', '-m', required=True, help='ComfyUI models directory')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Dry run, do not save files')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    split_checkpoint(args.input, args.models_dir, args.dry_run)

if __name__ == '__main__':
    main()
