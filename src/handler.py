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

    # Clear content directories from previous runs
    shutil.rmtree('./job_files', ignore_errors=True)
    shutil.rmtree('./training', ignore_errors=True)


    # Ensure only allowed keys are present
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

    print(flat_directory)

    mc_args = []
    mc_args.append('--batch_size="1"')
    mc_args.append('--num_beams="1"')
    mc_args.append('--top_p="0.9"')
    mc_args.append('--max_length="75"')
    mc_args.append('--min_length="8"')
    mc_args.append('--beam_search')
    mc_args.append('--caption_extension=".txt" ".' + flat_directory + '"')
    mc_args.append('--caption_weights="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"')


    print('os.getcwd():' + os.getcwd())
    if (os.path.exists('./sd-scripts')):
        print('./sd-scripts exists so use option A')
        subprocess.run('python ./finetune/make_captions.py ' + ' '.join(mc_args), shell=True, check=True,cwd='./sd-scripts')
    elif (os.path.exists('../sd-scripts')): 
        print('../sd-scripts exists so use option B')
        subprocess.run('python ./finetune/make_captions.py ' + ' '.join(mc_args), shell=True, check=True)
    else:
        print('neither ./sd-scripts nor ../sd-scripts exists so use option C')
        subprocess.run('python ./finetune/make_captions.py ' + ' '.join(mc_args), shell=True, check=True)
        

    # subprocess.run('python ./finetune/make_captions.py ' + ' '.join(mc_args), shell=True, check=True,cwd='./sd-scripts')
    print(mc_args)
    #return {"error": "error"}



    # Input with default values
    output_name = job_input.get('id', "a-traing-specific-name")

    # Accelerate launch arguments
    # See https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch
    # And https://github.com/bmaltais/kohya_ss/wiki/LoRA-training-parameters
    args = []

    num_cpu_threads_per_process = job_input.get('num_cpu_threads_per_process', 0)
    if num_cpu_threads_per_process > 0:
        args.append('--num_cpu_threads_per_process=' + str(num_cpu_threads_per_process))

    # args.append('--num_cpu_threads_per_process=' + str(job_input.get('num_cpu_threads_per_process', 1)))
    args.append('./sdxl_train_network.py')

    args.append('--bucket_no_upscale')
    args.append('--bucket_reso_steps=64')
    args.append('--cache_latents')
    args.append('--keep_tokens="20"')
    args.append('--learning_rate="' + job_input['learning_rate'] + '"')
    args.append('--logging_dir="./training/logs"')
    args.append('--lr_scheduler_num_cycles="' + job_input['lr_scheduler_num_cycles'] + '"')
    args.append('--lr_scheduler="' + job_input['lr_scheduler'] + '"')
    args.append('--max_data_loader_n_workers="' + job_input['max_data_loader_n_workers'] + '"')
    args.append('--max_train_steps="' + job_input['max_train_steps'] + '"')
    args.append('--mixed_precision="' + job_input['mixed_precision'] + '"')
    args.append('--network_alpha="8"')
    args.append('--network_dim="' + job_input['network_dim'] + '"')
    args.append('--network_module=networks.lora')
    args.append('--network_train_unet_only')
    args.append('--no_half_vae')
    args.append('--noise_offset=0.1')
    args.append('--optimizer_type="' + job_input['optimizer_type'] + '"')
    args.append('--output_dir="./training/model"')
    args.append('--output_name="' + job_input['output_name'] + '"')
    args.append('--pretrained_model_name_or_path="/model_cache/sd_xl_base_1.0.safetensors"')
    args.append('--resolution="1024,1024"')
    args.append('--save_every_n_epochs="1"')
    args.append('--save_model_as=safetensors')
    args.append('--save_precision="' + job_input['save_precision'] + '"')
    args.append('--text_encoder_lr=5e-05')
    args.append('--train_batch_size="' + job_input['train_batch_size'] + '"')
    args.append('--train_data_dir="./training/img"')
    args.append('--unet_lr="' + job_input['unet_lr'] + '"')
    args.append('--xformers')

#     --network_dim=16 --output_name="last" --lr_scheduler_num_cycles="2" --no_half_vae --learning_rate="4e-07" --lr_scheduler="constant" --train_batch_size="1" --max_train_steps="1000" --save_every_n_epochs="1" --mixed_precision="fp16" --save_precision="fp16" --cache_latents --optimizer_type="Adafactor" --max_data_loader_n_workers="0" --keep_tokens="20" --bucket_reso_steps=64 --xformers --bucket_no_upscale --noise_offset=0.1 --network_train_unet_only

        # https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch
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
