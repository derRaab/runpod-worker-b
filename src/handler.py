'''
Handler for the generation of a fine tuned lora model.
'''

import os
import shutil
import subprocess

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_download, upload_file_to_bucket

from rp_schema import INPUT_SCHEMA


def handler(job):

    job_input = job['input']

    if 'errors' in (job_input := validate(job_input, INPUT_SCHEMA)):
        return {'error': job_input['errors']}
    job_input = job_input['validated_input']

    # Download the zip file
    downloaded_input = rp_download.file(job_input['zip_url'])

    if not os.path.exists('./training'):
        os.mkdir('./training')
        os.mkdir('./training/img')
        os.mkdir(
            f"./training/img/{job_input['steps']}_{job_input['instance_name']} {job_input['class_name']}")
        os.mkdir('./training/model')
        os.mkdir('./training/logs')

    # Make clean data directory
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    flat_directory = f"./training/img/{job_input['steps']}_{job_input['instance_name']} {job_input['class_name']}"
    os.makedirs(flat_directory, exist_ok=True)

    for root, dirs, files in os.walk(downloaded_input['extracted_path']):
        # Skip __MACOSX folder
        if '__MACOSX' in root:
            continue

        for file in files:
            file_path = os.path.join(root, file)
            if os.path.splitext(file_path)[1].lower() in allowed_extensions:
                shutil.copy(
                    os.path.join(downloaded_input['extracted_path'], file_path),
                    flat_directory
                )

                # Input with default values
                learning_rate = job_input.get('learning_rate', "4e-07")
                lr_scheduler = job_input.get('lr_scheduler', "constant")
                lr_scheduler_num_cycles = job_input.get('lr_scheduler_num_cycles', 2)
                max_data_loader_n_workers = job_input.get('max_data_loader_num_workers', 0)
                max_train_steps = job_input.get('max_train_steps', 1000)
                mixed_precision = job_input.get('mixed_precision', "fp16")
                network_dim = job_input.get('network_dim', 16)
                optimizer_type = job_input.get('optimizer_type', "Adafactor")
                output_name = job_input.get('id', "a-traing-specific-name")
                save_precision = job_input.get('save_precision', "fp16")
                train_batch_size = job_input.get('train_batch_size', 1)
                unet_lr = job_input.get('unet_lr', 0.0001)

                # Accelerate launch arguments https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch
                args = []
                args.append('--num_cpu_threads_per_process="2"')
                args.append('./sdxl_train_network.py')

                args.append('--bucket_no_upscale')
                args.append('--bucket_reso_steps=64')
                args.append('--cache_latents')
                args.append('--keep_tokens="20"')
                args.append('--learning_rate="' + str(learning_rate) + '"')
                args.append('--lr_scheduler_num_cycles="' + str(lr_scheduler_num_cycles) + '"')
                args.append('--lr_scheduler="' + str(lr_scheduler) + '"')
                args.append('--max_data_loader_n_workers="' + str(max_data_loader_n_workers) + '"')
                args.append('--max_train_steps="' + str(max_train_steps) + '"')
                args.append('--mixed_precision="' + str(mixed_precision) + '"')
                args.append('--network_alpha="8"')
                args.append('--network_dim="' + str(network_dim) + '"')
                args.append('--network_module=networks.lora')
                args.append('--network_train_unet_only')
                args.append('--no_half_vae')
                args.append('--noise_offset=0.1')
                args.append('--optimizer_type="' + str(optimizer_type) + '"')
                args.append('--output_dir="./training/model"')
                args.append('--output_name="' + str(output_name) + '"')
                args.append('--pretrained_model_name_or_path="/model_cache/sd_xl_base_1.0.safetensors"')
                args.append('--resolution="1024,1024"')
                args.append('--save_every_n_epochs="1"')
                args.append('--save_model_as=safetensors')
                args.append('--save_precision="' + str(save_precision) + '"')
                args.append('--text_encoder_lr=5e-05')
                args.append('--train_batch_size="' + str(train_batch_size) + '"')
                args.append('--train_data_dir="./training/img"')
                args.append('--unet_lr="' + str(unet_lr) + '"')
                args.append('--xformers')

                subprocess.run(f"""accelerate launch {' '.join(args)}""", shell=True, check=True)

    job_s3_config = job.get('s3Config')

    uploaded_lora_url = upload_file_to_bucket(
        file_name=f"{job['id']}.safetensors",
        file_location=f"./training/model/{job['id']}.safetensors",
        bucket_creds=job_s3_config,
        bucket_name=None if job_s3_config is None else job_s3_config['bucketName'],
    )

    return {"lora": uploaded_lora_url}


runpod.serverless.start({"handler": handler})
